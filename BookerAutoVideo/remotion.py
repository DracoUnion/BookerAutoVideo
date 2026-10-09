"""remotion 子命令编排器：脚手架 → 生成代码 → 校验修复 → 渲染视频。"""
import os
import re
import json
import logging
import subprocess as subp
from os import path
from typing import List

from .util import safe_mkdir, safe_remove, is_video, write_text
from .remotion_agent import RemotionAgent
from .remotion_models import RemotionFile

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def _run(cmd: str, cwd: str):
    """运行命令（Windows 下 npx/npm 需 shell=True），记录日志。

    以字节捕获输出，用 UTF-8（回退 GBK）解码，避免子进程输出含 CJK/emoji
    时因 Windows 默认 GBK 解码抛 UnicodeDecodeError。
    """
    logger.info(f'cmd: {cmd}')
    r = subp.run(
        cmd, cwd=cwd, shell=True,
        stdout=subp.PIPE, stderr=subp.STDOUT,
        encoding='utf-8', errors='ignore',
        text=True,
    )
    text = (r.stdout or '')
    if r.returncode != 0:
        logger.warn(f'命令退出码 {r.returncode}：\n{text[-2000:]}')
    return r.returncode, text


class RemotionOrchestrator:
    """协调 Remotion 脚手架、LLM 代码生成、编译校验与渲染。"""

    def __init__(self, args):
        self.args = args
        self.pj_dir = path.abspath(args.dir)
        self.agent = RemotionAgent(args)

    # ── 步骤 1：脚手架 ─────────────────────────────────────────

    def step_scaffold(self):
        logger.info('[1] 脚手架 Remotion 工程')
        safe_mkdir(self.pj_dir)
        if path.isfile(path.join(self.pj_dir, 'package.json')):
            logger.info('[1] 已存在 package.json，跳过 create-video')
        else:
            rc, _ = _run('npx create-video@latest --yes --blank --no-tailwind .', self.pj_dir)
            if not path.isfile(path.join(self.pj_dir, 'package.json')):
                raise RuntimeError(
                    f'create-video 脚手架失败（目录 {self.pj_dir} 未生成 package.json）。'
                    '请确认 Node.js / npx 可用且目录可写。'
                )
        _run('npm i', self.pj_dir)
        safe_mkdir(path.join(self.pj_dir, 'public'))

    # ── 步骤 2：生成代码文件 ───────────────────────────────────

    def step_gen_files(self, plan, composition: str) -> List[RemotionFile]:
        logger.info('[2] 生成根组件')
        files = self.agent.gen_root(plan, composition)
        for f in files:
            self._write_file(f)
        logger.info('[3] 生成场景组件')
        for i, scene in enumerate(plan.scenes):
            f = self.agent.gen_scene(plan, scene)
            self._write_file(f)
            files.append(f)
        return files

    def _write_file(self, f: RemotionFile):
        """按相对路径写入工程目录（限制在工程内，防止路径逃逸）。"""
        rel = f.fname.replace('\\', '/').lstrip('/')
        full = path.join(self.pj_dir, rel)
        # 防路径逃逸
        if not full.startswith(self.pj_dir):
            logger.warn(f'[!] 忽略非法路径 {f.fname}')
            return
        safe_mkdir(path.dirname(full))
        write_text(full, f.code)
        logger.info(f'  写入 {rel}')

    # ── 步骤 4：编译校验 + 修复 ────────────────────────────────

    def step_validate(self, files: List[RemotionFile], plan, composition: str) -> List[RemotionFile]:
        logger.info('[4] 编译校验')
        err_map = {f.fname: f.code for f in files}
        for i in range(self.args.check):
            rc, out = _run('npx tsc --noEmit', self.pj_dir)
            if rc == 0:
                logger.info('[4] 类型检查通过')
                return files
            logger.info(f'[4] 第 {i+1} 轮类型检查未通过，修复')
            fixed = False
            # 只修复报错信息里提到的文件，避免误改无关文件
            touched = [f for f in err_map if f.split('/')[-1] in out]
            if not touched:
                logger.info('[4] 报错未定位到具体文件，停止修复')
                break
            for fname in touched:
                err = [l for l in out.splitlines() if fname.split('/')[-1] in l] or [out]
                r = self.agent.fix_file(fname, err_map[fname], '\n'.join(err), plan)
                err_map[r.fname] = r.code
                fixed = True
            if not fixed:
                break
            # 写回修复后的文件
            for fname, code in err_map.items():
                self._write_file(RemotionFile(fname=fname, code=code))
            # 更新 files 以便返回
            files = [RemotionFile(fname=k, code=v) for k, v in err_map.items()]
        logger.warn('[4] 类型检查未通过，保留最后代码，继续尝试渲染')
        return files

    # ── 步骤 5：渲染视频 ───────────────────────────────────────

    def step_render(self, composition: str, out_fname: str):
        logger.info('[5] 渲染视频')
        if self.args.still:
            _run(f'npx remotion still {composition} out/preview.png', self.pj_dir)
        out = path.join(self.pj_dir, out_fname)
        _run(f'npx remotion render {composition} {out_fname}', self.pj_dir)
        if is_video(out):
            logger.info(f'[*] 视频已生成：{out}')
        else:
            logger.warn(f'[*] 未找到输出文件 {out}，渲染可能失败')

    # ── 主流程 ─────────────────────────────────────────────────

    def run(self):
        logger.info(self.args)
        safe_mkdir(self.pj_dir)
        self.step_scaffold()
        plan = self.agent.plan(
            self.args.topic,
            width=self.args.width, height=self.args.height,
            fps=self.args.fps, duration=self.args.duration,
        )
        logger.info(f'分镜：{plan.title}，{len(plan.scenes)} 个场景')
        files = self.step_gen_files(plan, self.args.composition)
        files = self.step_validate(files, plan, self.args.composition)
        self.step_render(self.args.composition, self.args.out)
        logger.info('[*] 已完成，产物在 %s', self.pj_dir)


def remotion(args):
    """入口函数：创建编排器并运行。"""
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    RemotionOrchestrator(args).run()


def reg_subparser(subparsers):
    parser = subparsers.add_parser('remotion', help='用大模型编写 Remotion 代码并渲染视频')
    parser.add_argument('topic', help='视频主题 / 文案')
    parser.add_argument('-d', '--dir', default='.', help='目标工程目录（默认当前目录）')
    parser.add_argument('-W', '--width', type=int, default=1280, help='画布宽度（像素）')
    parser.add_argument('--height', type=int, default=720, dest='height', help='画布高度（像素）')
    parser.add_argument('--fps', type=int, default=30, help='帧率')
    parser.add_argument('--duration', type=int, default=10, help='视频时长（秒）')
    parser.add_argument('-c', '--check', type=int, default=3, help='编译校验 / 修复轮数')
    parser.add_argument('-o', '--out', default='out.mp4', help='输出视频相对路径')
    parser.add_argument('--composition', default='Main', help='Composition id')
    parser.add_argument('-sl', '--still', action='store_true', help='渲染前出一帧预览 PNG')
    parser.add_argument('-D', '--debug', action='store_true', help='调试模式')
    parser.set_defaults(func=remotion)
