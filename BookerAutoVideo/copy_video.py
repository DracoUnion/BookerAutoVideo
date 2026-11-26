"""copy 子命令编排器：视频逆向还原提示词 → Seedance 复刻生成。

对应任务书《视频复刻》第 5 节模块划分。
五阶段流水线：S1 抽帧 → S2 pHash 关键帧 → S3 逆向提示词 → S4 Seedance 生成 → S5 下载。
支持断点续跑（state.yaml）、干跑模式（--dry-run）、强制重跑（--force）。
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
    safe_mkdir, safe_remove, write_text, read_text,
    read_yaml_model, write_yaml_model,
    ffmpeg_get_info, get_video_imgs,
)
from .copy_models import (
    VideoMeta, Keyframe, CopyState, SeedanceTask,
)
from .copy_pmt import (
    REVERSE_SYSTEM_PMT, REVERSE_USER_PMT,
    KEYFRAME_DETAIL_TMPL, POLISH_SYSTEM_PMT, POLISH_USER_PMT,
    render_prompt, build_keyframe_details,
)
from .openai import OpenAIClient
from .seedance import (
    SeedanceClient, SeedanceAuthError, SeedanceTimeoutError, SeedanceError,
    create_client_from_args,
)

logger = logging.getLogger(__name__)


# ── 退出码 ────────────────────────────────────────────────────────

EXIT_OK = 0
EXIT_ERROR = 2        # 视频不存在 / 未配置视觉模型 / 无关键帧
EXIT_VISION_NO_IMAGE = 3   # 视觉模型不支持图片输入
EXIT_POLL_TIMEOUT = 5      # 轮询超时
EXIT_DOWNLOAD_FAIL = 6     # 下载失败


# ── 路径辅助 ──────────────────────────────────────────────────────

def _copy_data_dir() -> str:
    """复刻数据根目录：data/copy/"""
    return path.join(path.abspath(path.dirname(__file__)), '..', 'data', 'copy')


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
    """S1 抽取的样本帧目录（持久化，供 S2 断点续跑读取）。"""
    return path.join(_stage_dir(video_hash), 'frames')


def _keyframes_dir(video_hash: str) -> str:
    return path.join(_stage_dir(video_hash), 'keyframes')


def _keyframe_path(video_hash: str, index: int, time_sec: float) -> str:
    return path.join(
        _keyframes_dir(video_hash),
        f'kf_{index:03d}_t{time_sec:.2f}s.jpg',
    )


def _frame_path(video_hash: str, index: int) -> str:
    return path.join(_frames_dir(video_hash), f'frame_{index:06d}.jpg')


def _reversed_prompt_path(video_hash: str) -> str:
    return path.join(_stage_dir(video_hash), 'reversed_prompt.txt')


def _seedance_task_path(video_hash: str) -> str:
    return path.join(_stage_dir(video_hash), 'seedance_task.json')


# ── 辅助：输出文件名 ──────────────────────────────────────────────

def _output_fname(out_dir: str, video_hash: str) -> str:
    ts = int(time.time())
    return path.join(out_dir, f'copy_{video_hash}_{ts}.mp4')


def _prompt_out_path(out_dir: str, video_hash: str) -> str:
    ts = int(time.time())
    return path.join(out_dir, f'copy_{video_hash}_{ts}.prompt.txt')


# ── S1：抽帧 ──────────────────────────────────────────────────────

def _stage1_extract_frames(args, video_hash: str, state: CopyState) -> CopyState:
    """S1：按 sample_fps 从视频抽帧，记录 VideoMeta。

    优先复用 util.get_video_imgs（cv2.VideoCapture 按时间戳采样）。
    样本帧持久化到 frames/ 目录，供 S2 断点续跑读取。
    """
    t0 = time.time()
    video_path = args.VIDEO
    if not path.isfile(video_path):
        raise FileNotFoundError(f'视频文件不存在：{video_path}')

    # 获取视频元信息（ffmpeg）
    info = ffmpeg_get_info(video_path)
    fps = info.get('fps', 0.0) or 0.0
    duration = info.get('duration', 0.0) or 0.0

    # OpenCV 获取总帧 / 宽高
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'无法打开视频：{video_path}')
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    logger.info(f'[S1] 视频元信息：fps={fps:.2f}, 总帧={total_frames}, 时长={duration:.1f}s, {width}x{height}')

    meta = VideoMeta(
        path=video_path,
        fps=fps,
        total_frames=total_frames,
        duration=duration,
        width=width,
        height=height,
    )
    state.meta = meta

    # 按 sample_fps 抽帧（util.get_video_imgs 按 1/sample_fps 秒采样）
    sample_fps = float(getattr(args, 'sample_fps', 1.0) or 1.0)
    logger.info(f'[S1] 按 sample_fps={sample_fps} 抽帧')

    imgs, _ = get_video_imgs(video_path, fps=sample_fps)

    # 持久化样本帧到 frames/（供 S2 断点续跑读取）
    fdir = _frames_dir(video_hash)
    safe_mkdir(fdir)
    for i, img in enumerate(imgs):
        fp = _frame_path(video_hash, i)
        cv2.imwrite(fp, img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # 将样本帧数量临时挂在 state 上（私有属性，不入 YAML 序列化）
    object.__setattr__(state, '_sample_frame_count', len(imgs))
    object.__setattr__(state, '_sample_imgs_dir', fdir)

    logger.info(f'[S1] 抽取样本帧数：{len(imgs)}，耗时 {time.time() - t0:.1f}s')

    state.stage = 'done_s1'
    return state


def _load_sample_imgs(video_hash: str, sample_fps: float) -> List:
    """从 frames/ 目录加载样本帧（S2 断点续跑读取）。"""
    fdir = _frames_dir(video_hash)
    if not path.isdir(fdir):
        return []
    fnames = sorted(
        f for f in os.listdir(fdir) if f.startswith('frame_') and f.endswith('.jpg')
    )
    imgs = []
    for fn in fnames:
        fp = path.join(fdir, fn)
        img = cv2.imread(fp)
        if img is not None:
            imgs.append(img)
    return imgs


# ── S2：pHash 关键帧提取 ─────────────────────────────────────────

def _stage2_keyframes(args, video_hash: str, state: CopyState) -> CopyState:
    """S2：pHash 比对筛选关键帧，写盘。

    - 首帧必为关键帧
    - 与上一关键帧汉明距离 > threshold 记为关键帧
    - 超 max_keyframes 均匀采样截断
    """
    t0 = time.time()

    # 优先从 S1 持久化的 frames/ 读取（断点续跑）
    sample_imgs = _load_sample_imgs(video_hash, float(getattr(args, 'sample_fps', 1.0) or 1.0))
    # 如果 S1 本次刚执行且 state 上挂了样本帧目录，优先用那个
    sample_dir = getattr(state, '_sample_imgs_dir', None)
    if sample_dir and path.isdir(sample_dir):
        sample_imgs = _load_sample_imgs(video_hash, float(getattr(args, 'sample_fps', 1.0) or 1.0))

    if not sample_imgs:
        raise RuntimeError('S1 未抽取样本帧，无法进行 S2')

    hash_size = int(getattr(args, 'hash_size', 8))
    threshold = int(getattr(args, 'phash_threshold', 10))
    max_keyframes = int(getattr(args, 'max_keyframes', 10))

    keyframes: List[Keyframe] = []
    prev_phash = None

    kf_dir = _keyframes_dir(video_hash)
    safe_mkdir(kf_dir)

    keep_frames = getattr(args, 'keep_frames', False)

    for i, img in enumerate(sample_imgs):
        ph = imagehash.phash(
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
            hash_size=hash_size,
        )
        ph_hex = str(ph)

        if i == 0 or prev_phash is None:
            is_key = True
            dist = None
        else:
            dist = ph - prev_phash
            is_key = dist > threshold

        if is_key:
            sample_fps = float(getattr(args, 'sample_fps', 1.0) or 1.0)
            time_sec = i / sample_fps
            kf_path = _keyframe_path(video_hash, len(keyframes), time_sec)

            safe_mkdir(path.dirname(kf_path))
            cv2.imwrite(kf_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])

            keyframes.append(Keyframe(
                index=len(keyframes),
                time=round(time_sec, 2),
                path=kf_path,
                phash=ph_hex,
            ))
            if keep_frames and dist is not None:
                logger.info(f'  关键帧 {len(keyframes)-1}: t={time_sec:.2f}s, hamming_dist={dist}')
            prev_phash = ph

    # 若关键帧过多，均匀采样截断
    if len(keyframes) > max_keyframes:
        indices = [
            int(round(j * (len(keyframes) - 1) / (max_keyframes - 1)))
            for j in range(max_keyframes)
        ]
        keyframes = [keyframes[idx] for idx in indices]
        for i, kf in enumerate(keyframes):
            kf.index = i

    if not keyframes:
        logger.error(
            '[S2] 未筛选出任何关键帧。建议调低 --phash-threshold（当前 %d）'
            ' 或提高 --sample-fps', threshold
        )
        raise SystemExit(EXIT_ERROR)

    state.keyframes = keyframes
    logger.info(
        f'[S2] 关键帧数：{len(keyframes)}（样本帧 {len(sample_imgs)}, '
        f'threshold={threshold}, hash_size={hash_size}），耗时 {time.time() - t0:.1f}s'
    )
    state.stage = 'done_s2'
    return state


# ── S3：逆向提示词 ────────────────────────────────────────────────

def _vision_supports_image_error(ex: Exception) -> bool:
    """判断错误是否因“视觉模型不支持图片输入”。"""
    msg = str(ex).lower()
    return ('image_url' in msg) or ('image' in msg and 'not support' in msg) \
        or ('multimodal' in msg) or ('vision' in msg and 'not support' in msg)


def _stage3_reverse_prompt(args, video_hash: str, state: CopyState) -> CopyState:
    """S3：视觉模型逆向还原提示词，可选文本模型润色。

    - 按时间顺序把关键帧（data URL）写入 user message
    - 调用 -vm 模型；若返回 400 且提示 image_url 不支持，退出码 3
    - 可选润色：把原始提示词 + 生成参数交给 -m 模型
    """
    t0 = time.time()
    keyframes = state.keyframes
    duration = state.meta.duration

    # 视觉模型：-vm 优先，回退 -m
    vmodel = getattr(args, 'vmodel', None) or getattr(args, 'model', None)
    if not vmodel:
        logger.error('未配置视觉模型：请传入 -vm/--vmodel（或回退的 -m/--model）')
        raise SystemExit(EXIT_ERROR)

    # OpenAIClient 复用（S3 直接复用 openai.py 的客户端层）
    client = OpenAIClient.from_args(args)

    # 构建 user message：关键帧 data URL + 时间点文本
    keyframe_details = build_keyframe_details(keyframes, duration)
    user_content = render_prompt(REVERSE_USER_PMT,
                                 duration=duration,
                                 keyframe_count=len(keyframes),
                                 keyframe_details=keyframe_details)

    msgs = [
        {'role': 'system', 'content': REVERSE_SYSTEM_PMT},
        {'role': 'user', 'content': user_content},
    ]

    logger.info(f'[S3] 调用视觉模型 {vmodel} 逆向提示词，关键帧 {len(keyframes)} 帧')

    try:
        reversed_prompt = client.call_llm_retry(
            msgs, vmodel,
            retry=getattr(args, 'retry', 10),
            temp=getattr(args, 'temp', 1),
            max_tokens=getattr(args, 'max_tokens', None),
            extra_body=getattr(args, 'extra_body', None),
        )
    except Exception as ex:
        if _vision_supports_image_error(ex):
            logger.error(
                f'[S3] 视觉模型「{vmodel}」不支持图片输入，请更换支持多模态的模型（-vm）'
            )
            raise SystemExit(EXIT_VISION_NO_IMAGE)
        raise

    state.reversed_prompt = reversed_prompt.strip()
    logger.info(f'[S3] 逆向提示词已生成（{len(state.reversed_prompt)} 字），耗时 {time.time() - t0:.1f}s')
    logger.debug(f'逆向提示词：{state.reversed_prompt}')

    # ── 可选润色：文本模型 -m ────────────────────────────────────
    no_polish = getattr(args, 'no_polish', False)
    tmodel = getattr(args, 'model', None)

    if no_polish:
        state.polished_prompt = state.reversed_prompt
        logger.info('[S3] 已关闭润色（--no-polish），使用原始逆向提示词')
    elif tmodel:
        logger.info(f'[S3] 调用文本模型 {tmodel} 润色提示词')
        polish_user = render_prompt(POLISH_USER_PMT,
                                    reversed_prompt=state.reversed_prompt,
                                    duration=getattr(args, 'gen_duration', 8),
                                    resolution=getattr(args, 'gen_resolution', '1080p'),
                                    ratio=getattr(args, 'gen_ratio', '16:9'),
                                    audio=bool(getattr(args, 'gen_audio', True)),
                                    model=vmodel)
        polish_msgs = [
            {'role': 'system', 'content': POLISH_SYSTEM_PMT},
            {'role': 'user', 'content': polish_user},
        ]
        try:
            polished = client.call_llm_retry(
                polish_msgs, tmodel,
                retry=getattr(args, 'retry', 10),
                temp=getattr(args, 'temp', 1),
                max_tokens=getattr(args, 'max_tokens', None),
                extra_body=getattr(args, 'extra_body', None),
            )
            state.polished_prompt = polished.strip()
        except Exception as ex:
            logger.warning(f'[S3] 润色失败，使用原始提示词：{ex}')
            state.polished_prompt = state.reversed_prompt
    else:
        state.polished_prompt = state.reversed_prompt
        logger.info('[S3] 未配置文本模型（-m），跳过润色')

    # 写盘 reversed_prompt.txt
    safe_mkdir(_stage_dir(video_hash))
    write_text(_reversed_prompt_path(video_hash), state.polished_prompt or state.reversed_prompt)

    state.stage = 'done_s3'
    return state


# ── S4：Seedance 生成 ─────────────────────────────────────────────

def _stage4_seedance(args, video_hash: str, state: CopyState) -> CopyState:
    """S4：调用 Seedance 生成视频，记录任务 ID，续跑时先查已有任务。

    - 如 state 已有 seedance_task_id，先查状态；若已完成则直接取 URL
    - 否则创建新任务
    """
    t0 = time.time()

    prompt = state.polished_prompt or state.reversed_prompt
    if not prompt:
        raise RuntimeError('S3 未生成提示词，无法进行 S4')

    duration = int(getattr(args, 'gen_duration', 8))
    resolution = getattr(args, 'gen_resolution', '1080p')
    ratio = getattr(args, 'gen_ratio', '16:9')
    audio = bool(getattr(args, 'gen_audio', True))
    model = getattr(args, 'seedance_model', None) or 'seedance-2.0'

    client = create_client_from_args(args)

    task_id = state.seedance_task_id
    video_url = None

    # 续跑：若已有 task_id，先查状态
    if task_id:
        logger.info(f'[S4] 发现已有任务 ID {task_id}，先查状态（断点续跑）')
        try:
            task_data = client.get_task_status(task_id)
            status = getattr(task_data, 'status', None)
            status_val = status.value if hasattr(status, 'value') else str(status)
            logger.info(f'[S4] 任务 {task_id} 当前状态：{status_val}')
            if task_data.result and task_data.result.video_url:
                video_url = str(task_data.result.video_url)
                logger.info(f'[S4] 任务已完成，复用 video_url：{video_url}')
            elif status_val in ('failed', 'cancelled'):
                logger.warning(f'[S4] 任务 {task_id} 已结束（{status_val}），将重新创建')
                task_id = None
        except Exception as ex:
            logger.warning(f'[S4] 查询已有任务失败，将重新创建：{ex}')
            task_id = None

    # 创建新任务
    if not task_id:
        logger.info(f'[S4] 创建 Seedance 任务：duration={duration}s, resolution={resolution}, ratio={ratio}')
        task_id = client.create_task(
            prompt=prompt,
            duration=duration,
            resolution=resolution,
            ratio=ratio,
            audio=audio,
            model=model,
        )
        state.seedance_task_id = task_id

    # 轮询
    try:
        video_url = client.wait_task(task_id)
    except SeedanceTimeoutError as ex:
        logger.error(f'[S4] {ex}')
        state.video_url = None
        state.stage = 'done_s4'
        _save_state(state)
        raise SystemExit(EXIT_POLL_TIMEOUT)
    except SeedanceAuthError as ex:
        logger.error(f'[S4] {ex}')
        raise SystemExit(4)
    except SeedanceError as ex:
        logger.error(f'[S4] {ex}')
        raise SystemExit(EXIT_ERROR)

    state.video_url = video_url
    state.stage = 'done_s4'

    # 写盘 seedance_task.json
    task_record = SeedanceTask(
        task_id=task_id,
        status='succeeded',
        video_url=video_url,
        error=None,
        created_at=time.time() - t0,
        updated_at=time.time(),
        prompt=prompt,
        duration=duration,
        resolution=resolution,
        ratio=ratio,
        model=model,
        audio=audio,
    )
    safe_mkdir(_stage_dir(video_hash))
    write_text(_seedance_task_path(video_hash),
               json.dumps(task_record.model_dump(), ensure_ascii=False, indent=2))

    logger.info(f'[S4] 任务完成：task_id={task_id}, video_url={video_url}，耗时 {time.time() - t0:.1f}s')
    return state


# ── S5：下载 ──────────────────────────────────────────────────────

def _stage5_download(args, video_hash: str, state: CopyState) -> CopyState:
    """S5：下载视频到 <out>/copy_<hash>_<ts>.mp4。

    - 已存在且大小 > 0 则跳过
    """
    t0 = time.time()
    video_url = state.video_url
    if not video_url:
        raise RuntimeError('S4 未产生 video_url，无法进行 S5')

    out_dir = getattr(args, 'out', 'output/copy')
    safe_mkdir(out_dir)
    dest_path = _output_fname(out_dir, video_hash)

    # 已存在且非空则跳过
    if path.isfile(dest_path) and path.getsize(dest_path) > 0:
        logger.info(f'[S5] 输出文件已存在，跳过下载：{dest_path}')
        state.output_path = dest_path
        state.stage = 'done_s5'
        return state

    client = create_client_from_args(args)
    try:
        client.download_video(video_url, dest_path, max_retries=3)
    except SeedanceError as ex:
        logger.error(f'[S5] {ex}')
        raise SystemExit(EXIT_DOWNLOAD_FAIL)

    state.output_path = dest_path
    state.stage = 'done_s5'
    logger.info(f'[S5] 下载完成：{dest_path}，耗时 {time.time() - t0:.1f}s')
    return state


# ── 断点续跑：加载 / 保存 state ───────────────────────────────────

def _load_state(video_hash: str) -> Optional[CopyState]:
    sp = _state_path(video_hash)
    return read_yaml_model(sp, CopyState)


def _save_state(state: CopyState) -> None:
    safe_mkdir(_stage_dir(state.video_hash))
    write_yaml_model(_state_path(state.video_hash), state)


# ── 主流程 ────────────────────────────────────────────────────────

def copy(args) -> None:
    """copy 子命令入口：执行五阶段流水线。"""
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    t_all0 = time.time()
    video_path = args.VIDEO

    # 视频存在性检查（退出码 2）
    if not path.isfile(video_path):
        logger.error(f'视频文件不存在：{video_path}')
        raise SystemExit(EXIT_ERROR)

    # 视觉模型配置检查（退出码 2）
    vmodel = getattr(args, 'vmodel', None) or getattr(args, 'model', None)
    if not vmodel:
        logger.error('未配置视觉模型：请传入 -vm/--vmodel（或回退的 -m/--model）')
        raise SystemExit(EXIT_ERROR)

    video_hash = _video_hash(video_path)
    logger.info(f'视频：{video_path}，hash={video_hash}')

    # 加载或新建 state（断点续跑）
    state = _load_state(video_hash)
    force = getattr(args, 'force', False)
    if state and not force:
        logger.info(f'[缓存命中] 已有 state，stage={state.stage}，将跳过已完成阶段')
    else:
        state = CopyState(video_hash=video_hash, stage='init', meta=VideoMeta(
            path=video_path, fps=0.0, total_frames=0, duration=0.0, width=0, height=0
        ))
        logger.info('[缓存未命中] 从头开始')

    dry_run = getattr(args, 'dry_run', False)

    # S1 抽帧
    if force or state.stage not in ('done_s1', 'done_s2', 'done_s3', 'done_s4', 'done_s5'):
        state = _stage1_extract_frames(args, video_hash, state)
        _save_state(state)
        logger.info('[S1] 已保存 state')
    else:
        logger.info('[S1] 缓存命中，跳过抽帧')

    # S2 关键帧
    if force or state.stage not in ('done_s2', 'done_s3', 'done_s4', 'done_s5'):
        state = _stage2_keyframes(args, video_hash, state)
        _save_state(state)
        logger.info('[S2] 已保存 state')
    else:
        logger.info('[S2] 缓存命中，跳过关键帧提取')

    # S3 逆向提示词
    if force or state.stage not in ('done_s3', 'done_s4', 'done_s5'):
        state = _stage3_reverse_prompt(args, video_hash, state)
        _save_state(state)
        logger.info('[S3] 已保存 state')
    else:
        logger.info('[S3] 缓存命中，跳过逆向提示词')

    # 干跑：只到 S3，输出提示词后停止
    if dry_run:
        out_dir = getattr(args, 'out', 'output/copy')
        safe_mkdir(out_dir)
        prompt_out = _prompt_out_path(out_dir, video_hash)
        write_text(prompt_out, state.polished_prompt or state.reversed_prompt)
        logger.info(f'[dry-run] 已停止于 S3，逆向提示词：{prompt_out}')
        logger.info(f'[dry-run] 提示词内容：\n{state.polished_prompt or state.reversed_prompt}')
        elapsed = time.time() - t_all0
        logger.info(f'[*] dry-run 完成，总耗时 {elapsed:.1f}s')
        return

    # S4 Seedance 生成
    if force or state.stage not in ('done_s4', 'done_s5'):
        state = _stage4_seedance(args, video_hash, state)
        _save_state(state)
        logger.info('[S4] 已保存 state')
    else:
        logger.info('[S4] 缓存命中，跳过 Seedance 生成')

    # S5 下载
    if force or state.stage != 'done_s5':
        state = _stage5_download(args, video_hash, state)
        _save_state(state)
        logger.info('[S5] 已保存 state')
    else:
        logger.info('[S5] 缓存命中，跳过下载')

    elapsed = time.time() - t_all0
    logger.info(
        f'[*] 复刻完成：{state.output_path}，总耗时 {elapsed:.1f}s'
    )


def reg_subparser(subparsers):
    """注册 copy 子命令。"""
    parser = subparsers.add_parser(
        'copy',
        help='视频复刻：抽帧 → 关键帧 → 逆向 Seedance 提示词 → 生成 → 下载',
    )
    parser.add_argument('VIDEO', help='参考视频路径')
    parser.add_argument('-o', '--out', default='output/copy', help='输出目录（默认 output/copy）')
    parser.add_argument('--sample-fps', type=float, default=1.0, help='初筛抽帧频率（帧/秒，默认 1.0）')
    parser.add_argument('--phash-threshold', type=int, default=10, help='pHash 汉明距离阈值（默认 10）')
    parser.add_argument('--max-keyframes', type=int, default=10, help='送视觉模型的关键帧上限（默认 10）')
    parser.add_argument('--hash-size', type=int, default=8, help='pHash 尺寸（默认 8）')
    parser.add_argument('--vision-detail', choices=['low', 'high'], default='low',
                        help='OpenAI 图片 detail（低/高，默认 low）')
    parser.add_argument('--gen-duration', type=int, default=8, choices=[4, 8, 12],
                        help='Seedance 生成时长（秒，默认 8）')
    parser.add_argument('--gen-resolution', default='1080p', choices=['480p', '720p', '1080p'],
                        help='生成分辨率（默认 1080p）')
    parser.add_argument('--gen-ratio', default='16:9', choices=['16:9', '9:16', '1:1', '4:3', '3:4'],
                        help='画幅比例（默认 16:9）')
    parser.add_argument('--gen-audio', action='store_true', default=True, help='是否生成音频（默认开启）')
    parser.add_argument('--no-gen-audio', action='store_false', dest='gen_audio', help='关闭音频生成')
    parser.add_argument('--seedance-key', default=None, help='Seedance API Key（默认环境变量 SEEDANCE_API_KEY）')
    parser.add_argument('--seedance-host', default=None, help='Seedance Base URL（默认环境变量 SEEDANCE_BASE_URL）')
    parser.add_argument('--seedance-model', default='seedance-2.0', help='生成模型名（默认 seedance-2.0）')
    parser.add_argument('--poll-timeout', type=int, default=600, help='轮询超时秒（默认 600）')
    parser.add_argument('--dry-run', action='store_true', help='只执行 S1–S3，输出逆向提示词后停止，不调用 Seedance')
    parser.add_argument('--force', action='store_true', help='忽略缓存强制重跑')
    parser.add_argument('--keep-frames', action='store_true', help='保留全部采样帧（日志详情）')
    parser.add_argument('--no-polish', action='store_true', help='关闭文本模型润色（S3 步骤）')
    parser.set_defaults(func=copy)
