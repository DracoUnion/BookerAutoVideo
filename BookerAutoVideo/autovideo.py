import yaml
from os import path
import sys
import cv2
import math
import numpy as np
import librosa
import random
import hashlib
from .autovideo_config import config
from .util import *
from EpubCrawler.util import request_retry
import base64
from concurrent.futures import ThreadPoolExecutor

def tti(text):
    hash_ = hashlib.md5(text.encode('utf8')).hexdigest()
    cache = load_tti(hash_)
    if cache: return cache
    img = call_dalle_retry(
        text, config['ttiModel'], 
        config['ttiSize'], config['ttiQuality'],
        config['ttiRetry']
    )
    if img is None: img = open(config['defaultImg'], 'rb').read()
    save_tti(hash_, img)
    return img

def get_rand_asset_kw(dir, kw, func_filter=is_pic):
    tree = list(os.walk(dir))
    fnames = [path.join(d, n) for d, _, fnames in tree for n in fnames]
    pics = [n for n in fnames if func_filter(n)]
    cand = [n for n in pics if kw in n]
    return random.choice(cand) if len(cand) else  random.choice(pics)

def tts(text):
    hash_ = hashlib.md5(text.encode('utf8')).hexdigest()
    voice = config['ttsVoice']
    vol = config['ttsVolume']
    rate = config['ttsRate']
    cache = load_tts(hash_, voice, vol, rate)
    if cache: return cache
    data = edgetts_cli(text, voice=voice, volume=vol, rate=rate)
    save_tts(hash_, voice, vol, rate, data)
    return data

def srt_time_fmt(num):
    sec = int(num) % 60
    min_ = int(num) // 60 % 60
    hr = int(num) // 3600
    msec = int(num * 1000) % 1000
    return f'{hr:02d}:{min_:02d}:{sec:02d},{msec:03d}'

# 生成字幕
def gen_srt(frames):
    # 提取 audios 数组中的字幕
    subs = [
        {
            'text': a['subtitle'],
            'len': a['len'],
        }
        for a in frames
    ]
    # 将每个字幕按指定长度分割
    for s in subs:
        text = s['text']
        if not text: continue
        parts = split_sentence(text, config['subtitleMaxLen'])
        s['parts'] = [
            {
                'text': p,
                'len': len(p) / len(text) * s['len'],
            }
            for p in parts
        ]
    # 将分割后的字幕替换原字幕
    subs = sum([
        s.get('parts', s) for s in subs
    ], [])
    # 计算起始时间
    offset = 0
    for s in subs:
        s['start'] = offset
        offset += s['len']
    # 组装 SRT 文件
    srts = []
    for i, s in enumerate(subs):
        if not s['text']: continue
        st, ed = srt_time_fmt(s['start']), srt_time_fmt(s['start'] + s['len'])
        text = s['text']
        srts.append(f'{i+1}\n{st} --> {ed}\n{text}\n')
    srt = '\n'.join(srts)
    return srt

# 组装视频
def make_video(frames):
    # 合并视频
    video = ffmpeg_cat([f['video'] for f in frames])
    # 合并片头片尾
    if config['header']:
        header = open(config['header'], 'rb').read()
        video = ffmpeg_cat([header, video])
    if config['footer']:
        footer = open(config['footer'], 'rb').read()
        video = ffmpeg_cat([video, footer])
    return video

def tr_tts_tti(frame):
    text = frame['subtitle']
    frame['audio'] = tts(text)
    frame['image'] = tti(text)

def tr_asm_audio_video(frame):
    w, h = config['size']
    mode = config['resizeMode']
    # 缩放图像尺寸
    frame['image'] = resize_img(frame['image'], w, h, mode)
    # 静态图片转视频
    frame['len'] = audio_len(frame['audio'])
    frame['video_noaud'] = img_nsec_2video(frame['image'], frame['len'], config['fps'])
    # 组装音频和视频
    frame['video'] = ffmpeg_merge_video_audio(frame['video_noaud'], frame['audio'], audio_fmt='mp3')

def update_config(args):
    config['onePic'] = args.one_pic
    config['ttiRetry'] = args.retry
    config['ttiModel'] = args.model

def autovideo(args):
    set_openai_props(args.key, args.proxy, args.host)
    update_config(args)
    ext = extname(args.fname)
    if ext not in ['md', 'txt']:
        raise ValueError('文件扩展名必须是 TXT 或 MD')
    video_fname = args.fname[:-len(ext)-1] + '.' + config['format']
    if path.isfile(video_fname):
        raise ValueError('该文件已生成视频')
    # 将文件内容切分成行
    cont = open(args.fname, encoding='utf8').read()
    lines = md2lines(cont)
    print(lines)
    # 使用 TTS 和 TTI 工具生成语音和图像
    frames = [
        { 'subtitle': text }
        for text in lines
    ]
    pool = ThreadPoolExecutor(args.threads)
    hdls = []
    for f in frames:
        h = pool.submit(tr_tts_tti, f)
        hdls.append(h)
    for h in hdls:
        h.result()
    hdls.clear()
    for f in frames:
        h = pool.submit(tr_asm_audio_video, f)
        hdls.append(h)
    for h in hdls:
        h.result()
    # 组装视频
    video = make_video(frames)
    # 添加字幕
    srt = gen_srt(frames)
    video = ffmpeg_add_srt(video, srt)
    # 写文件
    print(video_fname)
    open(video_fname, 'wb').write(video)

    