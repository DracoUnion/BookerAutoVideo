"""sv2api 子命令：用 Flask 将 SenseVoice 封装为 OpenAI /v1/audio/transcriptions 兼容 API。

用法：
  python -m BookerAutoVideo sv2api --model-path D:\\models\\SenseVoiceSmall

OpenAI 客户端示例：
  openai.OpenAI(base_url="http://localhost:8000/v1", api_key="sv2api")
  openai.audio.transcriptions.create(model="sensevoice", file=open("a.mp3", "rb"))

返回 OpenAI 转录 JSON：
  {"text": "...", "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "..."}], ...}
"""
import os
import json
import tempfile
import uuid
import threading
import subprocess as subp
from os import path

from flask import Flask, request, jsonify
from .sencevoice import sencevoice
from .util import safe_mkdir, safe_remove

app = Flask('sv2api')

# 模型目录与全局锁（FunASR 推理非线程安全，串行处理）
_model_dir = ''
_lock = threading.Lock()
_warmup_done = False


def _asr_results(aud_fname: str):
    """调用 sencevoice，返回 [{'start','end','text','time'}, ...]（秒）。"""
    # sencevoice(args) 依赖 args.fname 与 args.whisper
    class _Args:
        fname = aud_fname
        whisper = _model_dir
    return sencevoice(_Args())


def _convert_to_mp3(raw: bytes) -> str:
    """把上传的音频字节（任意格式）转为临时 mp3 供 SenseVoice 处理，返回临时文件路径。"""
    mp3 = path.join(tempfile.gettempdir(), uuid.uuid4().hex + '.mp3')
    src = path.join(tempfile.gettempdir(), uuid.uuid4().hex + '.in')
    open(src, 'wb').write(raw)
    subp.Popen([
        'ffmpeg', '-i', src,
        '-vn', '-acodec', 'libmp3lame',
        mp3, '-y',
    ], shell=True, stdin=subp.PIPE).communicate()
    safe_remove(src)
    if not path.isfile(mp3) or not path.getsize(mp3):
        raise RuntimeError(f'音频转换失败（请确认已安装 ffmpeg）')
    return mp3


def _segments(results) -> list:
    """把 sencevoice 结果（start/end/text，单位秒）转 OpenAI 分段（毫秒）。"""
    segs = []
    for i, r in enumerate(results):
        start = float(r.get('start', r.get('time', 0.0)))
        end = float(r.get('end', start))
        segs.append({
            'id': i,
            'start': round(start * 1000, 3),
            'end': round(end * 1000, 3),
            'text': r.get('text', ''),
        })
    return segs


@app.route('/v1/audio/transcriptions', methods=['POST'])
def transcriptions():
    global _warmup_done
    with _lock:
        # OpenAI 兼容：file 字段为音频（audio/*），model 字段可忽略
        if 'file' not in request.files:
            return jsonify({'error': {'message': '缺少 file 字段', 'type': 'invalid_request_error'}}), 400
        f = request.files['file']
        raw = f.read()
        if not raw:
            return jsonify({'error': {'message': 'file 内容为空', 'type': 'invalid_request_error'}}), 400

        mp3 = _convert_to_mp3(raw)
        try:
            results = _asr_results(mp3)
        except Exception as ex:
            return jsonify({'error': {'message': str(ex), 'type': 'server_error'}}), 500
        finally:
            safe_remove(mp3)

        text = ' '.join(r.get('text', '') for r in results).strip()
        _warmup_done = True
        return jsonify({
            'model': 'sensevoice',
            'text': text,
            'segments': _segments(results),
        })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'ok': True, 'model': _model_dir, 'ready': _warmup_done})


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'sv2api',
        'usage': 'POST /v1/audio/transcriptions (multipart: file=<audio>)',
        'openai_client': 'OpenAI(base_url="http://host:port/v1").audio.transcriptions.create(model="sensevoice", file=open("a.mp3","rb"))',
    })


def sv2api(args):
    """入口：加载模型并启动 Flask 服务。"""
    global _model_dir
    _model_dir = args.model_path
    if not path.isdir(_model_dir):
        raise FileNotFoundError(f'SenseVoice 模型目录不存在：{_model_dir}')

    # 预加载模型（首次推理较慢，提前 warmup）
    print(f'加载 SenseVoice 模型：{_model_dir}')
    _Args_warm = type('A', (), {'fname': '', 'whisper': _model_dir})
    with _lock:
        # 用一个空音频触发模型加载（不实际转写）
        try:
            from .util import gen_blank_audio
            wav = gen_blank_audio(0.1, fmt='mp3')
            warm_fname = path.join(tempfile.gettempdir(), uuid.uuid4().hex + '.mp3')
            open(warm_fname, 'wb').write(wav)
            sencevoice(type('W', (), {'fname': warm_fname, 'whisper': _model_dir}))
            safe_remove(warm_fname)
        except Exception as ex:
            print(f'模型预热跳过：{ex}')

    print(f'OpenAI 兼容转录 API：http://{args.host}:{args.port}/v1/audio/transcriptions')
    print(f'  curl -F file=@a.mp3 http://{args.host}:{args.port}/v1/audio/transcriptions')
    app.run(host=args.host, port=args.port, threaded=False)


def reg_subparser(subparsers):
    parser = subparsers.add_parser('sv2api', help='将 SenseVoice 封装为 OpenAI 兼容 /v1/audio/transcriptions API')
    parser.add_argument('-m', '--model-path', default=os.environ.get('SENSEVOICE_MODEL_PATH', ''),
                        help='SenseVoice 模型目录（含 fsmn-vad 子目录）')
    parser.add_argument('--host', default='0.0.0.0', help='监听地址')
    parser.add_argument('--port', type=int, default=8000, help='监听端口')
    parser.add_argument('-t', '--threads', type=int, default=1, help='保留参数（当前串行）')
    parser.set_defaults(func=sv2api)
