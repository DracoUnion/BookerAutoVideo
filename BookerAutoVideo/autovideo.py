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

def tti(text):
    raise NotImplementedError()

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


'''
def update_config(user_cfg, cfg_dir):
    global tts
    global tti
    
    config.update(user_cfg)
    if not config['contents']:
        raise AttributeError('内容为空，无法生成')
        
    for cont in config['contents']:
        if cont['type'].endswith(':file'):
            cont['value'] = path.join(cfg_dir, norm_path_slash(cont['value']))
    if config['header']:
        config['header'] = path.join(cfg_dir, norm_path_slash(config['header']))
    if config['footer']:
        config['footer'] = path.join(cfg_dir, norm_path_slash(config['footer']))
        
    if config['external']:
        mod_fname = path.join(cfg_dir, norm_path_slash(config['external']))
        exmod = load_module(mod_fname)
        if hasattr(exmod, 'tts'): tts = exmod.tts
        if hasattr(exmod, 'tti'): tti = exmod.tti
'''

def autovideo(args):
    ext = extname(args.fname)
    if ext not in ['md', 'txt']:
        raise ValueError('文件扩展名必须是 TXT 或 MD')
    # 将文件内容切分成行
    cont = open(args.fname, encoding='utf8').read()
    lines = md2lines(cont)
    # 使用 TTS 和 TTI 工具生成语音和图像
    frames = [
        { 
            'subtitle': text,
            'audio': tts(text),
            'image': tti(text),
        }
        for text in lines
    ]
    w, h = config['size']
    mode = config['resizeMode']
    for f in frames:
        # 缩放图像尺寸
        f['image'] = resize_img(f['image'], w, h, mode)
        # 静态图片转视频
        f['len'] = audio_len(f['audio'])
        f['video_noaud'] = img_nsec_2video(f['image'], f['len'], config['fps'])
        # 组装音频和视频
        f['video'] = ffmpeg_merge_video_audio(f['video_noaud'], f['audio'], audio_fmt='mp3')
    # 组装视频
    video = make_video(frames)
    # 添加字幕
    srt = gen_srt(frames)
    video = ffmpeg_add_srt(video, srt)
    # 写文件
    video_fname = args.fname[:-len(ext)-1] + '.' + config['format']
    print(video_fname)
    open(video_fname, 'wb').write(video)

    