"""ffmpeg 后期封装（S8）：字幕烧录 / BGM 混音 / 节奏处理。

对应任务书第 7.8 节。只新增函数，不修改 ffmpeg.py 现有签名。
字幕/BGM/节奏拆成独立小函数，便于组合与测试。
命令用 list 形式（shell=False），规避 Windows 下引号转义问题。
"""
import os
import json
import shutil
import logging
import subprocess as subp
import tempfile
import uuid
from os import path

logger = logging.getLogger(__name__)


def _run(cmd, cwd=None):
    """运行命令（list 形式），UTF-8 解码避免 Windows GBK 崩溃。"""
    logger.info(f'cmd: {cmd}')
    r = subp.run(
        cmd, cwd=cwd,
        stdout=subp.PIPE, stderr=subp.STDOUT,
        encoding='utf-8', errors='ignore', text=True,
    )
    if r.returncode != 0:
        logger.warning(f'命令退出码 {r.returncode}：\n{(r.stdout or "")[-2000:]}')
    return r.returncode, (r.stdout or '')


def _tmpdir() -> str:
    d = path.join(tempfile.gettempdir(), uuid.uuid4().hex)
    os.makedirs(d, exist_ok=True)
    return d


def _cleanup(tmpdir: str):
    try:
        shutil.rmtree(tmpdir)
    except Exception:
        pass


def _stage_input(input_video: str, tmp: str, name: str = 'in.mp4') -> str:
    """把输入视频拷到临时目录，返回临时文件名。"""
    dst = path.join(tmp, name)
    open(dst, 'wb').write(open(input_video, 'rb').read())
    return dst


def burn_subtitle(input_video: str, subtitle_path: str, output_video: str) -> str:
    """烧录字幕（subtitles 滤镜，默认烧录）。

    Args:
        input_video: 输入视频路径
        subtitle_path: 字幕文件路径（.srt/.ass）
        output_video: 输出视频路径
    """
    if not path.isfile(subtitle_path):
        raise FileNotFoundError(f'字幕文件不存在：{subtitle_path}')

    tmp = _tmpdir()
    in_name = _stage_input(input_video, tmp, 'in.mp4')
    ext = path.splitext(subtitle_path)[1]
    sub_name = path.join(tmp, 'sub' + ext)
    open(sub_name, 'wb').write(open(subtitle_path, 'rb').read())
    out_name = path.join(tmp, 'out.mp4')

    vf = f"subtitles=sub{ext}:force_style='FontName=黑体,FontSize=24,OutlineColour=&H80000000,BorderStyle=3'"
    rc, _ = _run(
        ['ffmpeg', '-i', 'in.mp4', '-vf', vf, '-c:a', 'copy', 'out.mp4', '-y'],
        cwd=tmp,
    )
    if rc != 0 or not path.isfile(out_name) or path.getsize(out_name) == 0:
        raise RuntimeError('字幕烧录失败')
    open(output_video, 'wb').write(open(out_name, 'rb').read())
    _cleanup(tmp)
    logger.info(f'字幕烧录完成：{output_video}')
    return output_video


def mix_bgm(input_video: str, bgm_path: str, volume: float, output_video: str) -> str:
    """混音 BGM（amix + volume，BGM 循环铺满）。

    Args:
        input_video: 输入视频路径
        bgm_path: BGM 文件路径
        volume: BGM 音量（0.0~1.0）
        output_video: 输出视频路径
    """
    if not path.isfile(bgm_path):
        raise FileNotFoundError(f'BGM 文件不存在：{bgm_path}')

    tmp = _tmpdir()
    _stage_input(input_video, tmp, 'in.mp4')
    bgm_ext = path.splitext(bgm_path)[1]
    bgm_name = path.join(tmp, 'bgm' + bgm_ext)
    open(bgm_name, 'wb').write(open(bgm_path, 'rb').read())
    out_name = path.join(tmp, 'out.mp4')

    # 原音（若有）与 BGM 混音；BGM 循环铺满，音量可调
    filter_complex = (
        f'[1:a]volume={volume}[bg];'
        f'[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[a]'
    )
    rc, _ = _run(
        ['ffmpeg', '-i', 'in.mp4', '-stream_loop', '-1', '-i', f'bgm{bgm_ext}',
         '-filter_complex', filter_complex,
         '-map', '0:v', '-map', '[a]', '-c:v', 'copy', '-c:a', 'aac', 'out.mp4', '-y'],
        cwd=tmp,
    )
    if rc != 0 or not path.isfile(out_name) or path.getsize(out_name) == 0:
        raise RuntimeError('BGM 混音失败')
    open(output_video, 'wb').write(open(out_name, 'rb').read())
    _cleanup(tmp)
    logger.info(f'BGM 混音完成：{output_video}')
    return output_video


def _render_segment(in_name, tmp, idx, seg):
    """渲染单个片段（trim + speed），返回片段文件路径或 None。"""
    seg_out = path.join(tmp, f'seg_{idx}.mp4')
    start = seg.get('start', 0)
    end = seg.get('end')
    speed = seg.get('speed', 1.0)

    cmd = ['ffmpeg', '-i', in_name]
    if end is not None:
        cmd += ['-ss', str(start), '-to', str(end)]
    else:
        cmd += ['-ss', str(start)]
    if speed and speed != 1.0:
        cmd += ['-vf', f'setpts={1.0 / speed:.4f}*PTS', '-af', f'atempo={speed}']
    cmd += ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', seg_out, '-y']
    rc, _ = _run(cmd, cwd=tmp)
    return seg_out if rc == 0 and path.isfile(seg_out) else None


def apply_rhythm(input_video: str, rhythm_config: dict, output_video: str) -> str:
    """节奏处理：支持 trim（截取）、speed（变速）、concat（多段拼接）。

    Args:
        input_video: 输入视频路径
        rhythm_config: 节奏配置 JSON
        output_video: 输出视频路径
    """
    if not rhythm_config:
        raise ValueError('rhythm_config 为空')

    tmp = _tmpdir()
    in_name = _stage_input(input_video, tmp, 'in.mp4')

    # concat 列表：多段截取/变速后拼接
    concat = rhythm_config.get('concat')
    if concat:
        seg_files = [f for f in (_render_segment(in_name, tmp, i, seg)
                                 for i, seg in enumerate(concat)) if f]
        if not seg_files:
            raise RuntimeError('节奏 concat 处理失败')

        list_name = path.join(tmp, 'list.txt')
        open(list_name, 'w', encoding='utf8').write(
            '\n'.join(f"file '{f}'" for f in seg_files))
        out_name = path.join(tmp, 'out.mp4')
        rc, _ = _run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'list.txt',
                      '-c', 'copy', 'out.mp4', '-y'], cwd=tmp)
        if rc != 0 or not path.isfile(out_name):
            raise RuntimeError('节奏 concat 拼接失败')
        open(output_video, 'wb').write(open(out_name, 'rb').read())
        _cleanup(tmp)
        logger.info(f'节奏 concat 处理完成：{output_video}')
        return output_video

    # 单段 trim + speed
    trim = rhythm_config.get('trim', [None, None])
    seg = {'start': trim[0] or 0, 'end': trim[1], 'speed': rhythm_config.get('speed', 1.0)}
    seg_out = _render_segment(in_name, tmp, 0, seg)
    if not seg_out:
        raise RuntimeError('节奏 trim/speed 处理失败')
    open(output_video, 'wb').write(open(seg_out, 'rb').read())
    _cleanup(tmp)
    logger.info(f'节奏 trim/speed 处理完成：{output_video}')
    return output_video


def post_process(
    input_video: str,
    subtitle_path: str = None,
    bgm_path: str = None,
    bgm_volume: float = 0.25,
    rhythm: dict = None,
    output_video: str = None,
) -> str:
    """S8 后期总入口：按需组合字幕/BGM/节奏，逐步处理。

    无 subtitle / bgm / rhythm 时对应步骤跳过。

    Returns:
        最终输出视频路径
    """
    if output_video is None:
        output_video = input_video

    cur = input_video
    tmp = _tmpdir()
    try:
        if rhythm:
            cur = apply_rhythm(cur, rhythm, path.join(tmp, 'after_rhythm.mp4'))
        if subtitle_path:
            cur = burn_subtitle(cur, subtitle_path, path.join(tmp, 'after_sub.mp4'))
        if bgm_path:
            cur = mix_bgm(cur, bgm_path, bgm_volume, path.join(tmp, 'after_bgm.mp4'))

        if cur != output_video:
            open(output_video, 'wb').write(open(cur, 'rb').read())
    finally:
        _cleanup(tmp)

    logger.info(f'[S8] 后期完成：{output_video}')
    return output_video