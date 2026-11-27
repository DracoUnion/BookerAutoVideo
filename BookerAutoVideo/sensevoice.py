"""SenseVoice 语音转文字客户端（S5）。

对应任务书第 7.5 节。支持两种模式：
- API 模式：优先使用环境变量/CLI 指定的 OpenAI 兼容 /v1/audio/transcriptions 接口；
- 本地模式：回退到项目已有的 sencevoice（FunASR）封装。

统一接口 transcribe(audio_path, ...) -> Transcript，主流程只依赖该抽象方法。
"""
import os
import logging
import subprocess as subp
import tempfile
import uuid
from os import path
from typing import Optional

from .copy_models import Transcript
from .util import safe_remove, safe_mkdir

logger = logging.getLogger(__name__)


class SenseVoiceError(Exception):
    """SenseVoice / ASR 错误基类。"""
    pass


def extract_audio(video_path: str, out_wav: Optional[str] = None) -> str:
    """从视频提取音频为 16kHz 单声道 PCM wav。

    ffmpeg -i video -vn -acodec pcm_s16le -ar 16000 -ac 1 audio.wav
    """
    if out_wav is None:
        out_wav = path.join(tempfile.gettempdir(), uuid.uuid4().hex + '.wav')
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        out_wav, '-y',
    ]
    logger.info(f'提取音频：{video_path} -> {out_wav}')
    r = subp.run(cmd, stdout=subp.PIPE, stderr=subp.STDOUT,
                 encoding='utf-8', errors='ignore', text=True)
    if r.returncode != 0 or not path.isfile(out_wav) or path.getsize(out_wav) == 0:
        raise SenseVoiceError(
            f'音频提取失败（请确认已安装 ffmpeg，且视频含音频流）：\n{(r.stdout or "")[-1000:]}'
        )
    return out_wav


def _transcribe_api(audio_path: str, language: str, model: str,
                    key: str, host: str) -> Transcript:
    """调用 OpenAI 兼容 /v1/audio/transcriptions 接口转写。"""
    import requests

    url = host.rstrip('/') + '/audio/transcriptions'
    headers = {'Authorization': f'Bearer {key}'}
    with open(audio_path, 'rb') as f:
        files = {'file': (path.basename(audio_path), f, 'audio/wav')}
        data = {'model': model}
        if language:
            data['language'] = language
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=300)

    if resp.status_code in (401, 403):
        raise SenseVoiceError(f'SenseVoice API 鉴权失败（{resp.status_code}）：请检查 key/host')
    resp.raise_for_status()

    j = resp.json()
    text = (j.get('text') or '').strip()
    if not text:
        raise SenseVoiceError(f'SenseVoice API 返回空文案：{j}')
    return Transcript(
        text=text,
        language=j.get('language') or language,
        segments=j.get('segments'),
    )


def _transcribe_local(audio_path: str, language: str, model: str) -> Transcript:
    """回退到本地 sencevoice（FunASR）转写。model 为模型目录。"""
    try:
        from .sencevoice import sencevoice as _local_asr
    except Exception as e:
        raise SenseVoiceError(
            f'未配置 SenseVoice API，且本地 ASR 不可用：{e}。'
            '请设置 --sensevoice-key/--sensevoice-host，或提供本地模型目录 --sensevoice-model'
        )

    if not model or not path.isdir(model):
        raise SenseVoiceError(
            f'本地 SenseVoice 模型目录无效：{model!r}。'
            '请通过 --sensevoice-model 指定本地模型目录，或改用 API 模式'
        )

    class _Args:
        fname = audio_path
        whisper = model

    try:
        results = _local_asr(_Args())
    except Exception as e:
        raise SenseVoiceError(f'本地 SenseVoice 转写失败：{e}') from e

    text = ' '.join(r.get('text', '') for r in results).strip()
    if not text:
        raise SenseVoiceError('本地 SenseVoice 返回空文案（可能无音频/静音）')
    return Transcript(text=text, language=language, segments=results)


def transcribe(
    audio_path: str,
    language: str = 'zh',
    model: str = 'sensevoice-small',
    key: Optional[str] = None,
    host: Optional[str] = None,
) -> Transcript:
    """统一转写接口：优先 API 模式，回退本地模式。

    Args:
        audio_path: 音频文件路径（wav/mp3 等）
        language: 语言（zh/en/...）
        model: API 模式为模型名；本地模式为模型目录
        key: API Key（None 则读环境变量）
        host: API Base URL（None 则读环境变量）

    Returns:
        Transcript
    """
    key = key or os.environ.get('SENSEVOICE_API_KEY')
    host = host or os.environ.get('SENSEVOICE_BASE_URL')
    model = model or os.environ.get('SENSEVOICE_MODEL', 'sensevoice-small')

    if key and host:
        logger.info(f'[S5] SenseVoice API 模式：model={model}, language={language}')
        return _transcribe_api(audio_path, language, model, key, host)

    logger.info(f'[S5] SenseVoice 本地模式：model={model}, language={language}')
    return _transcribe_local(audio_path, language, model)


def transcribe_video(
    video_path: str,
    language: str = 'zh',
    model: str = 'sensevoice-small',
    key: Optional[str] = None,
    host: Optional[str] = None,
) -> Transcript:
    """从视频提取音频并转写（S5 便捷入口）。"""
    wav = extract_audio(video_path)
    try:
        return transcribe(wav, language=language, model=model, key=key, host=host)
    finally:
        safe_remove(wav)