"""comic-drama 子命令编排器：剧本拆解 → 图片 → 关键帧 → 视频合成。

对应 CDA.md 第 4 节四个模块 + 第 4.6 节 ProjectStore。
每个阶段产物落盘（storyboard.json / images / keyframes / shots / final.mp4），
支持断点续跑：某产物已存在则跳过，被用户手动替换则后续阶段自动采纳。
"""
import os
import re
import json
import logging
import subprocess as subp
from os import path
from typing import List, Dict, Optional

from .util import safe_mkdir, safe_remove, safe_rmdir, is_video, is_pic, write_text, read_text, ffmpeg_get_info
from .ffmpeg import ffmpeg_cat
from .comic_drama_agent import ComicDramaAgent
from .comic_drama_models import (
    Storyboard, CharacterAppearance, Scene, Shot,
    ImageAsset, KeyframePair, VideoResult,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def _run(cmd: str, cwd: str):
    """运行命令，字节捕获 + UTF-8 解码（避免 Windows 默认 GBK 解码崩溃）。"""
    logger.info(f'cmd: {cmd}')
    r = subp.run(
        cmd, cwd=cwd, shell=True,
        stdout=subp.PIPE, stderr=subp.STDOUT,
        encoding='utf-8', errors='ignore', text=True,
    )
    text = r.stdout or ''
    if r.returncode != 0:
        logger.warn(f'命令退出码 {r.returncode}：\n{text[-2000:]}')
    return r.returncode, text


class ComicDramaOrchestrator:
    """协调 CDA 四个阶段：ScriptBreaker → ImageGen → KeyframeGen → VideoGen。"""

    def __init__(self, args):
        self.args = args
        self.agent = ComicDramaAgent(args)
        self.pj_dir = self.agent.pj_dir
        # ProjectStore（CDA.md 4.6）：统一目录结构
        self.images_dir = path.join(self.pj_dir, 'images')
        self.keyframes_dir = path.join(self.pj_dir, 'keyframes')
        self.shots_dir = path.join(self.pj_dir, 'shots')
        safe_mkdir(self.images_dir)
        safe_mkdir(self.keyframes_dir)
        safe_mkdir(self.shots_dir)

    # ── ProjectStore 辅助 ─────────────────────────────────────

    def _storyboard_fname(self) -> str:
        return path.join(self.pj_dir, 'storyboard.json')

    def _save_storyboard(self, sb: Storyboard):
        json.dump(sb.model_dump(), open(self._storyboard_fname(), 'w', encoding='utf8'),
                  ensure_ascii=False, indent=2)

    def _load_storyboard(self) -> Optional[Storyboard]:
        f = self._storyboard_fname()
        if not path.isfile(f):
            return None
        try:
            return Storyboard(**json.load(open(f, encoding='utf8')))
        except Exception:
            return None

    @staticmethod
    def _all_shots(sb: Storyboard) -> List[Shot]:
        return [s for sc in sb.scenes for s in sc.shots]

    # ── 模块 1：剧本拆解（ScriptBreaker）──────────────────────

    def step_break(self, script: str) -> Storyboard:
        logger.info('[1] 剧本拆解')
        f = self._storyboard_fname()
        if path.isfile(f):
            logger.info(f'[1] 已存在 {f}，跳过拆解（如需重跑请删除）')
            return self._load_storyboard()
        sb = self.agent.break_script(
            script,
            style=self.args.style,
            duration=self.args.duration,
            width=self.args.width, height=self.args.height,
            project_id=self.args.project_id,
        )
        # 校验：角色外观锚点是否齐备（覆盖所有出现角色）
        for _ in range(self.args.check):
            prob = self._break_problem(sb)
            if not prob:
                logger.info('[1] 分镜校验通过')
                break
            logger.warn(f'[1] 分镜校验失败：\n{prob}')
            sb = self.agent.fix_storyboard(sb, prob)
        self._save_storyboard(sb)
        logger.info(f'[1] 共 {len(sb.scenes)} 场景 / {len(self._all_shots(sb))} 分镜')
        return sb

    @staticmethod
    def _break_problem(sb: Storyboard) -> str:
        """本地规则校验（无需 LLM）：分镜完整性、时长、角色锚点。

        只做硬结构检查；角色是否出现在英文 prompt 里交由 LLM 在 fix 轮判断
        （英文 prompt 常用拼音/译名，纯字面匹配会误报）。
        """
        prob = []
        shots = ComicDramaOrchestrator._all_shots(sb)
        if not shots:
            prob.append('没有任何分镜')
        total = sum(s.duration_seconds for s in shots)
        if total <= 0:
            prob.append('分镜总时长为 0')
        # 角色外观锚点不可为空
        for c in sb.characters:
            if not (c.appearance and c.clothing):
                prob.append(f'角色「{c.name}」的外观锚点不完整')
        # shot_id 连续
        ids = [s.shot_id for s in shots]
        if ids and ids != list(range(1, len(ids) + 1)):
            prob.append(f'shot_id 不连续：{ids}')
        # 每个 shot 必须有 image_prompt
        empty_prompt = [s.shot_id for s in shots if not s.image_prompt]
        if empty_prompt:
            prob.append(f'以下分镜缺少 image_prompt：{empty_prompt}')
        return '\n'.join(prob)

    # ── 模块 2：图片素材（ImageGen）────────────────────────────

    def step_images(self, sb: Storyboard) -> List[ImageAsset]:
        logger.info('[2] 生成 / 登记图片素材')
        user_images: Dict[int, str] = self._load_user_images()
        assets: List[ImageAsset] = []
        for shot in self._all_shots(sb):
            rel = f'images/shot_{shot.shot_id:03d}.png'
            full = path.join(self.pj_dir, rel)
            if path.isfile(full) and path.getsize(full):
                # 断点续跑 / 用户已替换
                assets.append(ImageAsset(shot_id=shot.shot_id, image_path=rel, source='existing'))
                logger.info(f'  shot {shot.shot_id:03d} 已存在 {rel}')
                continue
            if shot.shot_id in user_images:
                # 用户素材：直接登记（可选编辑）
                src = user_images[shot.shot_id]
                open(full, 'wb').write(open(src, 'rb').read())
                assets.append(ImageAsset(shot_id=shot.shot_id, image_path=rel, source='user_provided'))
                logger.info(f'  shot {shot.shot_id:03d} 采用用户素材 {src}')
                continue
            # 文生图
            data = self.agent.gen_image_bytes(shot.image_prompt)
            if data is None:
                logger.warn(f'  shot {shot.shot_id:03d} 无文生图模型，跳过（需配置 tti 模型）')
                continue
            open(full, 'wb').write(data)
            assets.append(ImageAsset(shot_id=shot.shot_id, image_path=rel, source='generated'))
            logger.info(f'  shot {shot.shot_id:03d} 已生成 {rel}')
        json.dump([a.model_dump() for a in assets],
                  open(path.join(self.pj_dir, 'images.json'), 'w', encoding='utf8'),
                  ensure_ascii=False, indent=2)
        return assets

    def _load_user_images(self) -> Dict[int, str]:
        """读取用户素材映射（可选）。

        支持两种：--user-images "1=a.png,2=b.png" 或目录自动匹配 shot_001.png。
        """
        m: Dict[int, str] = {}
        raw = getattr(self.args, 'user_images', '')
        if raw:
            for pair in raw.split(','):
                pair = pair.strip()
                if '=' in pair:
                    k, v = pair.split('=', 1)
                    m[int(k)] = v
        return m

    # ── 模块 3：关键帧（KeyframeGen）───────────────────────────

    def step_keyframes(self, sb: Storyboard, assets: List[ImageAsset]) -> List[KeyframePair]:
        logger.info('[3] 生成关键帧')
        asset_map = {a.shot_id: a for a in assets}
        pairs: List[KeyframePair] = []
        for shot in self._all_shots(sb):
            start_rel = f'keyframes/shot_{shot.shot_id:03d}_start.png'
            end_rel = f'keyframes/shot_{shot.shot_id:03d}_end.png'
            start_full = path.join(self.pj_dir, start_rel)
            end_full = path.join(self.pj_dir, end_rel)
            pair = self.agent.gen_keyframe_pair(sb, shot)
            # 起始帧：若该 shot 有图片素材，直接沿用（减少不一致）
            a = asset_map.get(shot.shot_id)
            if a and path.isfile(path.join(self.pj_dir, a.image_path)):
                open(start_full, 'wb').write(open(path.join(self.pj_dir, a.image_path), 'rb').read())
            else:
                d = self.agent.gen_image_bytes(pair.start_frame_prompt)
                if d:
                    open(start_full, 'wb').write(d)
                elif not path.isfile(start_full):
                    logger.warn(f'  shot {shot.shot_id:03d} 起始帧无图（无 tti 模型）')
            # 结束帧
            d = self.agent.gen_image_bytes(pair.end_frame_prompt)
            if d:
                open(end_full, 'wb').write(d)
            pairs.append(KeyframePair(
                shot_id=shot.shot_id, start_frame=start_rel, end_frame=end_rel,
                start_frame_prompt=pair.start_frame_prompt,
                end_frame_prompt=pair.end_frame_prompt,
                motion_description=pair.motion_description,
            ))
        json.dump([p.model_dump() for p in pairs],
                  open(path.join(self.pj_dir, 'keyframes.json'), 'w', encoding='utf8'),
                  ensure_ascii=False, indent=2)
        return pairs

    # ── 模块 4：视频合成（VideoGen）────────────────────────────

    def step_video(self, sb: Storyboard, pairs: List[KeyframePair]) -> List[VideoResult]:
        logger.info('[4] 生成分镜视频并拼接')
        shot_map = {s.shot_id: s for s in self._all_shots(sb)}
        results: List[VideoResult] = []
        # 视频后端
        backend = self.args.video_backend
        shot_videos: List[str] = []
        for pair in pairs:
            shot = shot_map[pair.shot_id]
            rel = f'shots/shot_{pair.shot_id:03d}.mp4'
            full = path.join(self.pj_dir, rel)
            if path.isfile(full) and path.getsize(full):
                shot_videos.append(full)
                results.append(VideoResult(shot_id=pair.shot_id, video_path=rel,
                                            duration_seconds=shot.duration_seconds, model_used='cached'))
                continue
            if backend == 'ffmpeg':
                # 默认后端：用关键帧 + ffmpeg 做简单过渡（起始帧定帧 → 结束帧）
                self._ffmpeg_keyframe_video(shot, pair, full)
            else:
                # 外部视频后端（Sora / Runway 等）：提交 + 轮询，此处留 TODO 占位
                prompt = self.agent.gen_video_prompt(shot, pair.motion_description, int(shot.duration_seconds))
                logger.warn(f'  shot {pair.shot_id:03d} 视频后端「{backend}」未接入，'
                            f'已生成 prompt：\n{prompt}\n请手动渲染后放到 {rel}')
                continue
            if path.isfile(full) and path.getsize(full):
                shot_videos.append(full)
                results.append(VideoResult(shot_id=pair.shot_id, video_path=rel,
                                            duration_seconds=shot.duration_seconds, model_used=backend))
        # 拼接成片
        out_final = path.join(self.pj_dir, 'final.mp4')
        if shot_videos:
            data = ffmpeg_cat([open(v, 'rb').read() for v in shot_videos])
            open(out_final, 'wb').write(data)
            logger.info(f'[*] 成片已生成：{out_final}')
        else:
            logger.warn('[*] 无可拼接的分镜视频（视频后端未生成 mp4）')
        json.dump([r.model_dump() for r in results],
                  open(path.join(self.pj_dir, 'shots.json'), 'w', encoding='utf8'),
                  ensure_ascii=False, indent=2)
        return results

    def _ffmpeg_keyframe_video(self, shot: Shot, pair: KeyframePair, out_full: str):
        """用两帧关键帧做 ffmpeg 定帧过渡（ffmpeg 后端）。

        起始帧保持前 60% 时长，结束帧保持后 40%。纯帧级拼接，无真实运动。
        """
        dur = max(shot.duration_seconds, 0.5)
        d1 = dur * 0.6
        d2 = dur - d1
        start_full = path.join(self.pj_dir, pair.start_frame)
        end_full = path.join(self.pj_dir, pair.end_frame)
        # 两张静图 → 两段视频 → 拼接
        tmp = path.join(self.pj_dir, 'shots', f'_tmp_{shot.shot_id:03d}')
        safe_mkdir(tmp)
        v1 = path.join(tmp, 'a.mp4')
        v2 = path.join(tmp, 'b.mp4')
        _run(
            f'ffmpeg -loop 1 -i "{start_full}" -t {d1:.3f} -r 30 -vf scale={self.args.width}:{self.args.height} '
            f'-c:v libx264 -pix_fmt yuv420p "{v1}" -y',
            self.pj_dir,
        )
        _run(
            f'ffmpeg -loop 1 -i "{end_full}" -t {d2:.3f} -r 30 -vf scale={self.args.width}:{self.args.height} '
            f'-c:v libx264 -pix_fmt yuv420p "{v2}" -y',
            self.pj_dir,
        )
        if path.isfile(v1) and path.isfile(v2):
            _run(f'ffmpeg -i "{v1}" -i "{v2}" -filter_complex "[0:v][1:v]concat=n=2:v=1[a]" '
                 f'-map "[a]" -c:v libx264 -pix_fmt yuv420p "{out_full}" -y', self.pj_dir)
        else:
            logger.warn(f'  shot {shot.shot_id:03d} ffmpeg 关键帧视频生成失败')
        safe_remove(path.join(tmp, 'a.mp4'))
        safe_remove(path.join(tmp, 'b.mp4'))
        safe_rmdir(tmp)

    # ── 主流程 ─────────────────────────────────────────────────

    def run(self):
        logger.info(self.args)
        # project_id 缺省用 out 目录名
        if not self.args.project_id:
            self.args.project_id = path.basename(path.abspath(self.args.out)) or 'comic'
        safe_mkdir(self.pj_dir)
        sb = self.step_break(self.args.script)
        assets = self.step_images(sb)
        pairs = self.step_keyframes(sb, assets)
        self.step_video(sb, pairs)
        logger.info('[*] 已完成，产物在 %s', self.pj_dir)


def comic_drama(args):
    """入口函数：创建编排器并运行。"""
    if args.debug:
        logger.setLevel(logging.DEBUG)
    ComicDramaOrchestrator(args).run()


def reg_subparser(subparsers):
    parser = subparsers.add_parser('comic-drama', help='漫剧 Agent：剧本 → 分镜 → 图片 → 关键帧 → 成片')
    parser.add_argument('script', help='剧本 / 故事梗概文本')
    parser.add_argument('-o', '--out', default='.', help='输出目录')
    parser.add_argument('--project-id', default='', help='项目标识（默认用 out 目录名）')
    parser.add_argument('--style', default='animated comic drama, cinematic lighting, high detail',
                        help='整体风格预设（英文）')
    parser.add_argument('--duration', type=int, default=30, help='目标时长（秒）')
    parser.add_argument('-W', '--width', type=int, default=1280, help='画布宽度（像素）')
    parser.add_argument('--height', type=int, default=720, help='画布高度（像素）')
    parser.add_argument('-c', '--check', type=int, default=2, help='分镜校验 / 修复轮数')
    parser.add_argument('--user-images', default='',
                        help='用户素材映射 "shot_id=path,..."，或留空用 images/ 目录自动匹配')
    parser.add_argument('--video-backend', default='ffmpeg',
                        help='视频后端：ffmpeg（本地关键帧过渡）/ sora / runway（留待接入）')
    parser.add_argument('-D', '--debug', action='store_true', help='调试模式')
    parser.set_defaults(func=comic_drama)
