"""copy 子命令编排器：视频复刻（八阶段流水线）。

对应任务书《视频复刻》第 5 节模块划分。
画面路线：S1 抽帧 → S2 pHash 关键帧 → S3 逆向文生图提示词 → S4 Seedance 文生图
文案路线：S5 SenseVoice 提取文案 → S6 大模型改写文案
合流：S7 Seedance 图生视频（图片 + 改写文案）→ S8 ffmpeg 字幕/BGM/节奏 → 输出

支持断点续跑（state.yaml）、干跑模式（--dry-run 只到 S3+S6）、强制重跑（--force）。
画面路线与文案路线状态独立记录，便于后续并行化。
"""
import os
import re
import json
import hashlib
import logging
import time
from os import path
from typing import List, Optional

import cv2
from PIL import Image
import imagehash

from .util import (
    safe_mkdir, write_text, read_text,
    read_yaml_model, write_yaml_model,
    ffmpeg_get_info, get_video_imgs,
)
from .copy_models import (
    VideoMeta, Keyframe, CopyState,
    ImagePromptResult, ImageResult, Transcript, RewriteResult,
    SeedanceImageTask, SeedanceVideoTask, PostConfig,
)
from .copy_pmt import (
    REVERSE_IMAGE_SYSTEM_PMT, POLISH_SYSTEM_PMT, POLISH_USER_PMT,
    REWRITE_SYSTEM_PMT, REWRITE_USER_PMT, DEFAULT_REWRITE_REQUIREMENT,
    render_prompt, build_vision_content,
)
from .openai import OpenAIClient
from .seedance import (
    SeedanceAuthError, SeedanceTimeoutError, SeedanceError,
    create_client_from_args,
)
from .sensevoice import transcribe_video, SenseVoiceError
from .ffmpeg_post import post_process

logger = logging.getLogger(__name__)


# ── 退出码（任务书第 8 节） ───────────────────────────────────────

EXIT_OK = 0
EXIT_ERROR = 2            # 视频不存在 / 未配置视觉模型 / 无关键帧
EXIT_VISION_NO_IMAGE = 3  # 视觉模型不支持图片输入
EXIT_AUTH = 4             # Seedance 鉴权失败（401/403）
EXIT_POLL_TIMEOUT = 5     # 轮询超时
EXIT_DOWNLOAD_FAIL = 6    # 下载失败
EXIT_ASR_FAIL = 7         # SenseVoice 无音频 / ASR 失败
EXIT_POST_FAIL = 8        # ffmpeg 后期失败


# ── 路径辅助 ──────────────────────────────────────────────────────

def _ensure_dir(d: str) -> str:
    """递归创建目录（os.makedirs，exist_ok），返回原路径。"""
    os.makedirs(d, exist_ok=True)
    return d


def _copy_data_dir() -> str:
    """复刻数据根目录：data/copy/（规范化绝对路径）。"""
    return path.normpath(path.join(path.dirname(__file__), '..', 'data', 'copy'))


def _video_hash(video_path: str) -> str:
    """计算视频文件 sha256 前 16 位，作缓存键。"""
    h = hashlib.sha256()
    with open(video_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def _stage_dir(video_hash: str) -> str:
    return path.join(_copy_data_dir(), video_hash)


def _state_path(video_hash: str) -> str:
    return path.join(_stage_dir(video_hash), 'state.yaml')


def _frames_dir(video_hash: str) -> str:
    return path.join(_stage_dir(video_hash), 'frames')


def _frame_path(video_hash: str, index: int) -> str:
    return path.join(_frames_dir(video_hash), f'frame_{index:06d}.jpg')


def _keyframes_dir(video_hash: str) -> str:
    return path.join(_stage_dir(video_hash), 'keyframes')


def _keyframe_path(video_hash: str, index: int, time_sec: float) -> str:
    return path.join(_keyframes_dir(video_hash), f'kf_{index:03d}_t{time_sec:.2f}s.jpg')


def _reversed_prompt_path(video_hash: str) -> str:
    return path.join(_stage_dir(video_hash), 'reversed_image_prompt.txt')


def _images_dir(video_hash: str) -> str:
    return path.join(_stage_dir(video_hash), 'images')


def _image_task_path(video_hash: str) -> str:
    return path.join(_stage_dir(video_hash), 'image_task.json')


def _transcript_path(video_hash: str) -> str:
    return path.join(_stage_dir(video_hash), 'transcript.txt')


def _rewritten_path(video_hash: str) -> str:
    return path.join(_stage_dir(video_hash), 'rewritten_script.txt')


def _video_task_path(video_hash: str) -> str:
    return path.join(_stage_dir(video_hash), 'video_task.json')


# ── 输出文件名 ────────────────────────────────────────────────────

def _out_fname(out_dir: str, video_hash: str, suffix: str) -> str:
    ts = int(time.time())
    return path.join(out_dir, f'copy_{video_hash}_{ts}{suffix}')


# ── 断点续跑：state 读写 ─────────────────────────────────────────

def _load_state(video_hash: str) -> Optional[CopyState]:
    return read_yaml_model(_state_path(video_hash), CopyState)


def _save_state(state: CopyState) -> None:
    _ensure_dir(_stage_dir(state.video_hash))
    write_yaml_model(_state_path(state.video_hash), state)


def _load_sample_imgs(video_hash: str) -> List:
    """从 frames/ 目录加载样本帧（S2 读取）。"""
    fdir = _frames_dir(video_hash)
    if not path.isdir(fdir):
        return []
    fnames = sorted(
        f for f in os.listdir(fdir) if f.startswith('frame_') and f.endswith('.jpg')
    )
    imgs = []
    for fn in fnames:
        img = cv2.imread(path.join(fdir, fn))
        if img is not None:
            imgs.append(img)
    return imgs


# ══════════════════════════════════════════════════════════════════
# 编排器
# ══════════════════════════════════════════════════════════════════

class CopyOrchestrator:
    """协调八阶段流水线：画面路线 S1→S4、文案路线 S5→S6、合流 S7→S8。"""

    def __init__(self, args):
        self.args = args
        if not path.isfile(args.VIDEO):
            logger.error(f'视频文件不存在：{args.VIDEO}')
            raise SystemExit(EXIT_ERROR)
        self.video_hash = _video_hash(args.VIDEO)
        self.state = _load_state(self.video_hash)
        self.client = OpenAIClient.from_args(args)
        self.force = bool(getattr(args, 'force', False))

    # ── 阶段跳过判断 ─────────────────────────────────────────

    def _done(self, *stages: str) -> bool:
        """state.stage 是否已完成给定阶段之一（用于粗粒度跳过）。"""
        return self.state is not None and self.state.stage in stages

    def _fresh_state(self) -> CopyState:
        return CopyState(
            video_hash=self.video_hash, stage='init',
            meta=VideoMeta(path=self.args.VIDEO, fps=0.0, total_frames=0,
                           duration=0.0, width=0, height=0),
        )

    # ── S1 抽帧 ─────────────────────────────────────────────

    def step_s1(self) -> CopyState:
        if self.state is None:
            self.state = self._fresh_state()
        if not self.force and self._done('done_s1', 'done_s2', 'done_s3', 'done_s4',
                                         'done_s5', 'done_s6', 'done_s7', 'done_s8'):
            logger.info('[S1] cache hit，跳过抽帧')
            return self.state

        t0 = time.time()
        video_path = self.args.VIDEO
        info = ffmpeg_get_info(video_path)
        fps = info.get('fps', 0.0) or 0.0
        duration = info.get('duration', 0.0) or 0.0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f'无法打开视频：{video_path}')
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()

        logger.info(f'[S1] 视频元信息：fps={fps:.2f}, 总帧={total_frames}, 时长={duration:.1f}s, {width}x{height}')
        self.state.meta = VideoMeta(path=video_path, fps=fps, total_frames=total_frames,
                                    duration=duration, width=width, height=height)

        sample_fps = float(getattr(self.args, 'sample_fps', 1.0) or 1.0)
        logger.info(f'[S1] 按 sample_fps={sample_fps} 抽帧')
        imgs, _ = get_video_imgs(video_path, fps=sample_fps)

        fdir = _frames_dir(self.video_hash)
        _ensure_dir(fdir)
        for i, img in enumerate(imgs):
            cv2.imwrite(_frame_path(self.video_hash, i), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        logger.info(f'[S1] 抽取样本帧数：{len(imgs)}，耗时 {time.time() - t0:.1f}s')
        self.state.stage = 'done_s1'
        _save_state(self.state)
        return self.state

    # ── S2 pHash 关键帧 ─────────────────────────────────────

    def step_s2(self) -> CopyState:
        if not self.force and self._done('done_s2', 'done_s3', 'done_s4',
                                         'done_s5', 'done_s6', 'done_s7', 'done_s8'):
            logger.info('[S2] cache hit，跳过关键帧提取')
            return self.state

        t0 = time.time()
        sample_imgs = _load_sample_imgs(self.video_hash)
        if not sample_imgs:
            raise RuntimeError('S1 未抽取样本帧，无法进行 S2')

        hash_size = int(getattr(self.args, 'hash_size', 8))
        threshold = int(getattr(self.args, 'phash_threshold', 10))
        max_keyframes = int(getattr(self.args, 'max_keyframes', 10))
        sample_fps = float(getattr(self.args, 'sample_fps', 1.0) or 1.0)

        keyframes: List[Keyframe] = []
        prev_phash = None
        _ensure_dir(_keyframes_dir(self.video_hash))

        for i, img in enumerate(sample_imgs):
            ph = imagehash.phash(
                Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), hash_size=hash_size)
            if i == 0 or prev_phash is None:
                is_key, dist = True, None
            else:
                dist = ph - prev_phash
                is_key = dist > threshold
            if is_key:
                time_sec = i / sample_fps
                kf_path = _keyframe_path(self.video_hash, len(keyframes), time_sec)
                cv2.imwrite(kf_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                keyframes.append(Keyframe(index=len(keyframes), time=round(time_sec, 2),
                                          path=kf_path, phash=str(ph)))
                prev_phash = ph

        if len(keyframes) > max_keyframes:
            idxs = [int(round(j * (len(keyframes) - 1) / (max_keyframes - 1)))
                    for j in range(max_keyframes)]
            keyframes = [keyframes[k] for k in idxs]
            for i, kf in enumerate(keyframes):
                kf.index = i

        if not keyframes:
            logger.error('[S2] 未筛选出任何关键帧。建议调低 --phash-threshold（当前 %d）或提高 --sample-fps', threshold)
            raise SystemExit(EXIT_ERROR)

        self.state.keyframes = keyframes
        logger.info(f'[S2] 关键帧数：{len(keyframes)}（样本帧 {len(sample_imgs)}），耗时 {time.time() - t0:.1f}s')
        self.state.stage = 'done_s2'
        _save_state(self.state)
        return self.state

    # ── S3 逆向文生图提示词 ─────────────────────────────────

    @staticmethod
    def _is_vision_unsupported(ex: Exception) -> bool:
        msg = str(ex).lower()
        return ('image_url' in msg) or ('image' in msg and 'not support' in msg) \
            or ('multimodal' in msg) or ('vision' in msg and 'not support' in msg)

    def step_s3(self) -> CopyState:
        if not self.force and self._done('done_s3', 'done_s4', 'done_s5',
                                         'done_s6', 'done_s7', 'done_s8'):
            logger.info('[S3] cache hit，跳过逆向提示词')
            return self.state

        t0 = time.time()
        vmodel = getattr(self.args, 'vmodel', None) or getattr(self.args, 'model', None)
        if not vmodel:
            logger.error('未配置视觉模型：请传入 -vm/--vmodel（或回退的 -m/--model）')
            raise SystemExit(EXIT_ERROR)

        detail = getattr(self.args, 'vision_detail', 'low')
        content = build_vision_content(self.state.keyframes, self.state.meta.duration, detail)
        msgs = [
            {'role': 'system', 'content': REVERSE_IMAGE_SYSTEM_PMT},
            {'role': 'user', 'content': content},
        ]
        logger.info(f'[S3] 调用视觉模型 {vmodel} 逆向文生图提示词，关键帧 {len(self.state.keyframes)} 帧')
        try:
            reversed_prompt = self.client.call_llm_retry(
                msgs, vmodel, retry=getattr(self.args, 'retry', 10),
                temp=getattr(self.args, 'temp', 1),
                max_tokens=getattr(self.args, 'max_tokens', None),
                extra_body=getattr(self.args, 'extra_body', None),
            )
        except Exception as ex:
            if self._is_vision_unsupported(ex):
                logger.error(f'[S3] 视觉模型「{vmodel}」不支持图片输入，请更换支持多模态的模型（-vm）')
                raise SystemExit(EXIT_VISION_NO_IMAGE)
            raise

        reversed_prompt = reversed_prompt.strip()

        # 可选润色（-m 文本模型，--no-polish 关闭）
        polished = reversed_prompt
        tmodel = getattr(self.args, 'model', None)
        if not getattr(self.args, 'no_polish', False) and tmodel:
            logger.info(f'[S3] 调用文本模型 {tmodel} 润色提示词')
            polish_user = render_prompt(
                POLISH_USER_PMT, reversed_prompt=reversed_prompt,
                image_ratio=getattr(self.args, 'image_ratio', '16:9'),
                image_resolution=getattr(self.args, 'image_resolution', '1080p'))
            try:
                polished = self.client.call_llm_retry(
                    [{'role': 'system', 'content': POLISH_SYSTEM_PMT},
                     {'role': 'user', 'content': polish_user}],
                    tmodel, retry=getattr(self.args, 'retry', 10),
                    temp=getattr(self.args, 'temp', 1),
                    max_tokens=getattr(self.args, 'max_tokens', None),
                    extra_body=getattr(self.args, 'extra_body', None),
                ).strip()
            except Exception as ex:
                logger.warning(f'[S3] 润色失败，使用原始提示词：{ex}')

        self.state.image_prompt = ImagePromptResult(
            reversed_prompt=reversed_prompt, polished_prompt=polished,
            image_ratio=getattr(self.args, 'image_ratio', '16:9'),
            image_resolution=getattr(self.args, 'image_resolution', '1080p'))
        write_text(_reversed_prompt_path(self.video_hash), polished)
        logger.info(f'[S3] 逆向文生图提示词已生成（{len(polished)} 字），耗时 {time.time() - t0:.1f}s')
        self.state.stage = 'done_s3'
        _save_state(self.state)
        return self.state

    # ── S4 Seedance 文生图 ──────────────────────────────────

    def step_s4(self) -> CopyState:
        if not self.force and self.state.image_result and \
                path.isfile(self.state.image_result.local_path) and \
                path.getsize(self.state.image_result.local_path) > 0:
            logger.info('[S4] cache hit，复用已生成图片')
            if self.state.stage in ('init', 'done_s1', 'done_s2', 'done_s3'):
                self.state.stage = 'done_s4'
                _save_state(self.state)
            return self.state

        t0 = time.time()
        prompt = (self.state.image_prompt.polished_prompt
                  or self.state.image_prompt.reversed_prompt)
        ratio = getattr(self.args, 'image_ratio', '16:9')
        resolution = getattr(self.args, 'image_resolution', '1080p')
        image_model = getattr(self.args, 'image_model', None) \
            or os.environ.get('SEEDANCE_IMAGE_MODEL', 'seedance-image')

        client = create_client_from_args(self.args)
        try:
            # 续跑：已有 image_task 先查状态
            task_id = self.state.image_task.task_id if self.state.image_task else None
            image_url = None
            if task_id and not self.force:
                logger.info(f'[S4] 发现已有文生图任务 {task_id}，先查状态')
                try:
                    image_url = client.wait_image_task(task_id, timeout=60)
                except Exception as ex:
                    logger.warning(f'[S4] 查询已有任务失败，将重新创建：{ex}')
                    task_id = None
            if not task_id:
                logger.info(f'[S4] 创建文生图任务：model={image_model}, ratio={ratio}, resolution={resolution}')
                task_id = client.create_image_task(prompt, ratio=ratio, resolution=resolution, model=image_model)
                image_url = client.wait_image_task(task_id)

            img_path = path.join(_images_dir(self.video_hash), 'seedance_image_0.jpg')
            _ensure_dir(path.dirname(img_path))
            client.download_image(image_url, img_path)

            self.state.image_task = SeedanceImageTask(
                task_id=task_id, status='succeeded', image_url=image_url,
                created_at=time.time() - t0, updated_at=time.time(),
                prompt=prompt, ratio=ratio, resolution=resolution, model=image_model)
            self.state.image_result = ImageResult(
                task_id=task_id, image_url=image_url, local_path=img_path)
            write_text(_image_task_path(self.video_hash),
                       json.dumps(self.state.image_task.model_dump(), ensure_ascii=False, indent=2))
            logger.info(f'[S4] 文生图完成：{img_path}，耗时 {time.time() - t0:.1f}s')
        except SeedanceAuthError as ex:
            logger.error(f'[S4] {ex}')
            raise SystemExit(EXIT_AUTH)
        except SeedanceTimeoutError as ex:
            logger.error(f'[S4] {ex}')
            self.state.stage = 'done_s4'
            _save_state(self.state)
            raise SystemExit(EXIT_POLL_TIMEOUT)
        except SeedanceError as ex:
            logger.error(f'[S4] {ex}')
            raise SystemExit(EXIT_ERROR)

        self.state.stage = 'done_s4'
        _save_state(self.state)
        return self.state

    # ── S5 SenseVoice 提取文案 ──────────────────────────────

    def step_s5(self) -> CopyState:
        if not self.force and self.state.transcript and self.state.transcript.text:
            logger.info('[S5] cache hit，跳过 ASR')
            if self.state.stage in ('init', 'done_s1', 'done_s2', 'done_s3', 'done_s4'):
                self.state.stage = 'done_s5'
                _save_state(self.state)
            return self.state

        logger.info('[S5] SenseVoice 提取文案')
        try:
            transcript = transcribe_video(
                self.args.VIDEO,
                language=getattr(self.args, 'asr_language', 'zh'),
                model=getattr(self.args, 'sensevoice_model', None)
                    or os.environ.get('SENSEVOICE_MODEL', 'sensevoice-small'),
                key=getattr(self.args, 'sensevoice_key', None),
                host=getattr(self.args, 'sensevoice_host', None),
            )
        except SenseVoiceError as ex:
            logger.error(f'[S5] {ex}')
            raise SystemExit(EXIT_ASR_FAIL)

        self.state.transcript = transcript
        write_text(_transcript_path(self.video_hash), transcript.text)
        logger.info(f'[S5] 文案提取完成（{len(transcript.text)} 字）')
        self.state.stage = 'done_s5'
        _save_state(self.state)
        return self.state

    # ── S6 大模型改写文案 ───────────────────────────────────

    def step_s6(self) -> CopyState:
        if not self.force and self.state.rewrite_result and self.state.rewrite_result.rewritten_text:
            logger.info('[S6] cache hit，跳过文案改写')
            if self.state.stage in ('init', 'done_s1', 'done_s2', 'done_s3', 'done_s4', 'done_s5'):
                self.state.stage = 'done_s6'
                _save_state(self.state)
            return self.state

        # --no-rewrite：直接用原始文案
        if getattr(self.args, 'no_rewrite', False):
            original = self.state.transcript.text
            self.state.rewrite_result = RewriteResult(original_text=original, rewritten_text=original)
            write_text(_rewritten_path(self.video_hash), original)
            logger.info('[S6] --no-rewrite，直接使用原始文案')
            self.state.stage = 'done_s6'
            _save_state(self.state)
            return self.state

        tmodel = getattr(self.args, 'model', None)
        if not tmodel:
            logger.warning('[S6] 未配置文本模型（-m），跳过改写，使用原始文案')
            original = self.state.transcript.text
            self.state.rewrite_result = RewriteResult(original_text=original, rewritten_text=original)
            write_text(_rewritten_path(self.video_hash), original)
            self.state.stage = 'done_s6'
            _save_state(self.state)
            return self.state

        original = self.state.transcript.text
        requirement = self._load_rewrite_requirement()
        user = render_prompt(REWRITE_USER_PMT, original_text=original, rewrite_requirement=requirement)
        logger.info(f'[S6] 调用文本模型 {tmodel} 改写文案')
        rewritten = self.client.call_llm_retry(
            [{'role': 'system', 'content': REWRITE_SYSTEM_PMT},
             {'role': 'user', 'content': user}],
            tmodel, retry=getattr(self.args, 'retry', 10),
            temp=getattr(self.args, 'temp', 1),
            max_tokens=getattr(self.args, 'max_tokens', None),
            extra_body=getattr(self.args, 'extra_body', None),
        ).strip()

        self.state.rewrite_result = RewriteResult(original_text=original, rewritten_text=rewritten)
        write_text(_rewritten_path(self.video_hash), rewritten)
        logger.info(f'[S6] 文案改写完成（{len(rewritten)} 字）')
        self.state.stage = 'done_s6'
        _save_state(self.state)
        return self.state

    def _load_rewrite_requirement(self) -> str:
        """加载改写要求：--rewrite-requirement 可为文件路径或直接文本，缺省用内置模板。"""
        raw = getattr(self.args, 'rewrite_requirement', None)
        if not raw:
            return DEFAULT_REWRITE_REQUIREMENT
        if path.isfile(raw):
            return read_text(raw)
        return raw

    # ── S7 Seedance 图生视频 ────────────────────────────────

    def step_s7(self) -> CopyState:
        if not self.force and self.state.video_task and self.state.video_task.video_url:
            logger.info('[S7] cache hit，复用已生成视频 URL')
            if self.state.stage in ('init', 'done_s1', 'done_s2', 'done_s3', 'done_s4', 'done_s5', 'done_s6'):
                self.state.stage = 'done_s7'
                _save_state(self.state)
            return self.state

        t0 = time.time()
        if not self.state.image_result or not path.isfile(self.state.image_result.local_path):
            raise RuntimeError('S4 未生成图片，无法进行 S7 图生视频')

        prompt = self.state.rewrite_result.rewritten_text if self.state.rewrite_result \
            else self.state.transcript.text
        duration = int(getattr(self.args, 'gen_duration', 8))
        resolution = getattr(self.args, 'gen_resolution', '1080p')
        ratio = getattr(self.args, 'gen_ratio', '16:9')
        audio = bool(getattr(self.args, 'gen_audio', True))
        model = getattr(self.args, 'seedance_model', None) or 'seedance-2.0'

        client = create_client_from_args(self.args)
        try:
            task_id = self.state.video_task.task_id if self.state.video_task else None
            video_url = None
            if task_id and not self.force:
                logger.info(f'[S7] 发现已有图生视频任务 {task_id}，先查状态')
                try:
                    video_url = client.wait_video_task(task_id, timeout=60)
                except Exception as ex:
                    logger.warning(f'[S7] 查询已有任务失败，将重新创建：{ex}')
                    task_id = None
            if not task_id:
                logger.info(f'[S7] 创建图生视频任务：duration={duration}s, ratio={ratio}')
                task_id = client.create_video_task(
                    self.state.image_result.local_path, prompt,
                    duration=duration, resolution=resolution, ratio=ratio, audio=audio, model=model)
                video_url = client.wait_video_task(task_id)

            self.state.video_task = SeedanceVideoTask(
                task_id=task_id, status='succeeded', video_url=video_url,
                created_at=time.time() - t0, updated_at=time.time(),
                prompt=prompt, duration=duration, resolution=resolution,
                ratio=ratio, model=model, audio=audio)
            write_text(_video_task_path(self.video_hash),
                       json.dumps(self.state.video_task.model_dump(), ensure_ascii=False, indent=2))
            logger.info(f'[S7] 图生视频完成：{video_url}，耗时 {time.time() - t0:.1f}s')
        except SeedanceAuthError as ex:
            logger.error(f'[S7] {ex}')
            raise SystemExit(EXIT_AUTH)
        except SeedanceTimeoutError as ex:
            logger.error(f'[S7] {ex}')
            self.state.stage = 'done_s7'
            _save_state(self.state)
            raise SystemExit(EXIT_POLL_TIMEOUT)
        except SeedanceError as ex:
            logger.error(f'[S7] {ex}')
            raise SystemExit(EXIT_ERROR)

        self.state.stage = 'done_s7'
        _save_state(self.state)
        return self.state

    # ── S8 ffmpeg 后期 ──────────────────────────────────────

    def step_s8(self) -> CopyState:
        out_dir = getattr(self.args, 'out', 'output/copy')
        _ensure_dir(out_dir)
        final_out = _out_fname(out_dir, self.video_hash, '.mp4')

        if not self.force and self.state.output_path and \
                path.isfile(self.state.output_path) and path.getsize(self.state.output_path) > 0:
            logger.info('[S8] cache hit，最终视频已存在')
            return self.state

        # --no-post 整体跳过：直接下载原始生成视频
        if getattr(self.args, 'no_post', False):
            logger.info('[S8] --no-post，跳过 ffmpeg 后期，直接下载生成视频')
            self._download_raw(final_out)
            self.state.output_path = final_out
            self.state.stage = 'done_s8'
            _save_state(self.state)
            return self.state

        # 无任何后期参数：也直接下载
        subtitle = getattr(self.args, 'subtitle', None)
        bgm = getattr(self.args, 'bgm', None)
        rhythm = self._load_rhythm()
        if not subtitle and not bgm and not rhythm:
            logger.info('[S8] 无后期参数，直接下载生成视频')
            self._download_raw(final_out)
            self.state.output_path = final_out
            self.state.stage = 'done_s8'
            _save_state(self.state)
            return self.state

        # 先下载原始生成视频，再做后期
        raw_path = _out_fname(out_dir, self.video_hash, '.raw.mp4')
        self._download_raw(raw_path)
        self.state.post_config = PostConfig(
            subtitle_path=subtitle, bgm_path=bgm,
            bgm_volume=float(getattr(self.args, 'bgm_volume', 0.25)), rhythm=rhythm)
        try:
            post_process(raw_path, subtitle_path=subtitle, bgm_path=bgm,
                         bgm_volume=float(getattr(self.args, 'bgm_volume', 0.25)),
                         rhythm=rhythm, output_video=final_out)
        except Exception as ex:
            logger.error(f'[S8] ffmpeg 后期失败：{ex}，保留中间视频 {raw_path}')
            raise SystemExit(EXIT_POST_FAIL)

        self.state.output_path = final_out
        self.state.stage = 'done_s8'
        _save_state(self.state)
        logger.info(f'[S8] 后期完成：{final_out}')
        return self.state

    def _load_rhythm(self) -> Optional[dict]:
        raw = getattr(self.args, 'rhythm', None)
        if not raw:
            return None
        if path.isfile(raw):
            return json.loads(read_text(raw))
        return json.loads(raw)

    def _download_raw(self, dest: str):
        """下载 S7 生成视频到 dest（已存在且非空则跳过）。"""
        if path.isfile(dest) and path.getsize(dest) > 0:
            logger.info(f'[S8] 视频已存在，跳过下载：{dest}')
            return
        client = create_client_from_args(self.args)
        try:
            client.download_video(self.state.video_task.video_url, dest, max_retries=3)
        except SeedanceError as ex:
            logger.error(f'[S8] 下载失败：{ex}')
            raise SystemExit(EXIT_DOWNLOAD_FAIL)

    # ── 主流程 ─────────────────────────────────────────────

    def run(self):
        logger.info(self.args)
        if not path.isfile(self.args.VIDEO):
            logger.error(f'视频文件不存在：{self.args.VIDEO}')
            raise SystemExit(EXIT_ERROR)
        vmodel = getattr(self.args, 'vmodel', None) or getattr(self.args, 'model', None)
        if not vmodel:
            logger.error('未配置视觉模型：请传入 -vm/--vmodel（或回退的 -m/--model）')
            raise SystemExit(EXIT_ERROR)

        if self.state is None:
            self.state = self._fresh_state()
            logger.info('[缓存未命中] 从头开始')
        elif not self.force:
            logger.info(f'[缓存命中] 已有 state，stage={self.state.stage}')
        else:
            logger.info('[--force] 强制重跑')

        t_all0 = time.time()

        # 画面路线 S1→S2→S3→S4
        self.step_s1()
        self.step_s2()
        self.step_s3()

        # 文案路线 S5→S6
        self.step_s5()
        self.step_s6()

        # 干跑：只到 S3 + S6，输出提示词/文案后停止
        if getattr(self.args, 'dry_run', False):
            self._emit_dry_run()
            logger.info(f'[*] dry-run 完成，总耗时 {time.time() - t_all0:.1f}s')
            return

        # 合流 S4→S7→S8
        self.step_s4()
        self.step_s7()
        self.step_s8()

        logger.info(f'[*] 复刻完成：{self.state.output_path}，总耗时 {time.time() - t_all0:.1f}s')

    def _emit_dry_run(self):
        out_dir = getattr(self.args, 'out', 'output/copy')
        _ensure_dir(out_dir)
        prompt_out = _out_fname(out_dir, self.video_hash, '.image_prompt.txt')
        write_text(prompt_out, self.state.image_prompt.polished_prompt
                   or self.state.image_prompt.reversed_prompt)
        transcript_out = _out_fname(out_dir, self.video_hash, '.transcript.txt')
        if self.state.transcript:
            write_text(transcript_out, self.state.transcript.text)
        rewritten_out = _out_fname(out_dir, self.video_hash, '.rewritten.txt')
        if self.state.rewrite_result:
            write_text(rewritten_out, self.state.rewrite_result.rewritten_text)
        logger.info(f'[dry-run] 文生图提示词：{prompt_out}')
        logger.info(f'[dry-run] 提示词内容：\n{self.state.image_prompt.polished_prompt or self.state.image_prompt.reversed_prompt}')
        if self.state.rewrite_result:
            logger.info(f'[dry-run] 改写文案：{rewritten_out}')
            logger.info(f'[dry-run] 改写文案内容：\n{self.state.rewrite_result.rewritten_text}')


# ── 入口 ──────────────────────────────────────────────────────────

def copy(args) -> None:
    """copy 子命令入口：创建编排器并运行。"""
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    CopyOrchestrator(args).run()


def reg_subparser(subparsers):
    """注册 copy 子命令。"""
    parser = subparsers.add_parser(
        'copy',
        help='视频复刻：抽帧 → 关键帧 → 逆向文生图提示词 → Seedance 文生图/图生视频 → ffmpeg 后期',
    )
    parser.add_argument('VIDEO', help='参考视频路径')
    parser.add_argument('-o', '--out', default='output/copy', help='输出目录（默认 output/copy）')
    # S1/S2
    parser.add_argument('--sample-fps', type=float, default=1.0, help='初筛抽帧频率（帧/秒，默认 1.0）')
    parser.add_argument('--phash-threshold', type=int, default=10, help='pHash 汉明距离阈值（默认 10）')
    parser.add_argument('--max-keyframes', type=int, default=10, help='送视觉模型的关键帧上限（默认 10）')
    parser.add_argument('--hash-size', type=int, default=8, help='pHash 尺寸（默认 8）')
    parser.add_argument('--vision-detail', choices=['low', 'high'], default='low', help='OpenAI 图片 detail（默认 low）')
    parser.add_argument('--keep-frames', action='store_true', help='保留全部采样帧（日志详情）')
    # S4 文生图
    parser.add_argument('--image-model', default='seedance-image', help='Seedance 文生图模型名（默认 seedance-image）')
    parser.add_argument('--image-ratio', default='16:9', choices=['16:9', '9:16', '1:1', '4:3', '3:4'], help='文生图画幅（默认 16:9）')
    parser.add_argument('--image-resolution', default='1080p', choices=['480p', '720p', '1080p'], help='文生图分辨率（默认 1080p）')
    # S5 ASR
    parser.add_argument('--sensevoice-key', default=None, help='SenseVoice API Key（默认环境变量 SENSEVOICE_API_KEY）')
    parser.add_argument('--sensevoice-host', default=None, help='SenseVoice Base URL（默认环境变量 SENSEVOICE_BASE_URL）')
    parser.add_argument('--sensevoice-model', default='sensevoice-small', help='SenseVoice 模型名/本地目录（默认 sensevoice-small）')
    parser.add_argument('--asr-language', default='zh', help='ASR 语言（默认 zh）')
    # S6 改写
    parser.add_argument('--rewrite-requirement', default=None, help='文案改写要求（文件路径或直接文本，缺省用内置模板）')
    parser.add_argument('--no-rewrite', action='store_true', help='跳过 S6，直接使用原始文案')
    parser.add_argument('--no-polish', action='store_true', help='关闭 S3 提示词润色')
    # S7 图生视频
    parser.add_argument('--gen-duration', type=int, default=8, choices=[4, 8, 12], help='图生视频时长（秒，默认 8）')
    parser.add_argument('--gen-resolution', default='1080p', choices=['480p', '720p', '1080p'], help='图生视频分辨率（默认 1080p）')
    parser.add_argument('--gen-ratio', default='16:9', choices=['16:9', '9:16', '1:1', '4:3', '3:4'], help='图生视频画幅（默认 16:9）')
    parser.add_argument('--gen-audio', action='store_true', default=True, help='是否生成音频（默认开启）')
    parser.add_argument('--no-gen-audio', action='store_false', dest='gen_audio', help='关闭音频生成')
    parser.add_argument('--seedance-key', default=None, help='Seedance API Key（默认环境变量 SEEDANCE_API_KEY）')
    parser.add_argument('--seedance-host', default=None, help='Seedance Base URL（默认环境变量 SEEDANCE_BASE_URL）')
    parser.add_argument('--seedance-model', default='seedance-2.0', help='图生视频模型名（默认 seedance-2.0）')
    parser.add_argument('--poll-timeout', type=int, default=600, help='轮询超时秒（默认 600）')
    # S8 后期
    parser.add_argument('--subtitle', default=None, help='字幕文件路径（.srt/.ass）')
    parser.add_argument('--bgm', default=None, help='BGM 文件路径')
    parser.add_argument('--bgm-volume', type=float, default=0.25, help='BGM 音量（默认 0.25）')
    parser.add_argument('--rhythm', default=None, help='节奏配置 JSON（文件路径或 JSON 文本）')
    parser.add_argument('--no-post', action='store_true', help='跳过 S8 ffmpeg 后期')
    # 通用
    parser.add_argument('--dry-run', action='store_true', help='只执行 S1–S3 与 S5–S6，输出提示词/文案后停止')
    parser.add_argument('--force', action='store_true', help='忽略缓存强制重跑')
    parser.set_defaults(func=copy)