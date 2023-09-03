import yaml
from os import path
import sys
import cv2
import math
import numpy as np
import librosa
import random
from io import BytesIO
import hashlib
from .autovideo_config import config
from .util import *
from scipy.io import wavfile
from EpubCrawler.util import request_retry

DIR = path.abspath(path.dirname(__file__))
RE_MD_IMG = r'!\[.*?\]\((.*?)\)'
RE_MD_TITLE = r'^#+ (.+?)$'
RE_MD_PRE = r'```[\s\S]+?```'
RE_MD_TR = r'^\|.+?\|$'
RE_MD_PREFIX = r'^\s*(\+|\-|\*|\d+\.|\>|\#+)'
RE_SENT_DELIM = r'\n|。|\?|？|;|；|:|：|!|！'

def gen_blank_audio(nsec, sr=22050, fmt='wav'):
    audio = np.zeros(int(nsec * sr), dtype=np.uint8)
    bio = BytesIO()
    wavfile.write(bio, sr, audio)
    audio = bio.getvalue()
    if fmt != 'wav':
        audio = ffmpeg_conv_fmt(audio, 'wav', fmt)
    return audio
    

def tti(text):
    raise NotImplementedError()

def gen_mono_color(w, h, bgr):
    assert len(bgr) == 3
    img = np.zeros([h, w, 3])
    img[:, :] = bgr
    img = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])[1]
    return bytes(img)

def audio_len(data):
    return ffmpeg_get_info(data)['duration']

def md2playbook(args):
    fname = args.fname
    if not fname.endswith('.md'):
        print('请提供 Markdown 文件')
        return
    cont = open(fname, encoding='utf8').read()
    m = re.search(RE_MD_TITLE, cont, flags=re.M)
    if not m:
        print('未找到标题，无法转换')
        return
    title = m.group(1)
    # 去掉代码块
    cont = re.sub(RE_MD_PRE, '', cont)
    # 去掉表格
    cont = re.sub(RE_MD_TR, '', cont, flags=re.M)
    # 去掉各种格式
    cont = re.sub(RE_MD_PREFIX, '', cont, flags=re.M)
    # 切分
    lines = re.split(RE_SENT_DELIM, cont)
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if l]
    
    contents = []
    for l in lines:
        # 提取图片
        m = re.search(RE_MD_IMG, l)
        if m: 
            url = m.group(1)
            type = (
                'image:url'
                if url.startswith('http://') or 
                   url.startswith('https://')
                else 'image:file'
            )
            value = (
                url if type == 'image:url'
                else path.join(path.dirname(fname), url)
            )
            contents.append({
                'type': type,
                'value': value,
            })
        l = re.sub(RE_MD_IMG, '', l).strip()
        if l: contents.append({
            'type': 'audio:tts',
            'value': l,
        })
        
    playbook = {
        'name': title,
        'contents': contents,
    }
    ofname = fname_escape(title) + '.yml'
    open(ofname, 'w', encoding='utf8').write(yaml.safe_save(ofname))
    print(ofname)
        
        
def get_rand_asset_kw(dir, kw, func_filter=is_pic):
    tree = list(os.walk(dir))
    fnames = [path.join(d, n) for d, _, fnames in tree for n in fnames]
    pics = [n for n in fnames if func_filter(n)]
    cand = [n for n in pics if kw in n]
    return random.choice(cand) if len(cand) else  random.choice(pics)

# 素材预处理
def preproc_asset(config):
    # 如果第一张不是图片，插入纯黑图片
    c0type = config['contents'][0]['type']
    if not c0type.startswith('image:') and \
       not c0type.startswith('video:'):
        config['contents'].insert(0, {
            'type': 'image:color',
            'value': '#000000',
        })

    # 加载或生成内容
    for cont in config['contents']:
        if cont['type'].endswith(':file'):
            cont['asset'] = open(cont['value'], 'rb').read()
        elif cont['type'] == 'image:dir':
            assert config['assetDir']
            cont['asset'] = get_rand_asset_kw(config['assetDir'], cont['value'], is_pic)
        elif cont['type'] == 'video:dir':
            assert config['assetDir']
            cont['asset'] = get_rand_asset_kw(config['assetDir'], cont['value'], is_video)
        elif cont['type'].endswith(':url'):
            url = cont['value']
            print(f'下载：{url}')
            cont['asset'] = request_retry('GET', url).content
        elif cont['type'] == 'audio:tts':
            text = cont['value']
            print(f'TTS：{text}')
            cont['asset'] = tts(text)
        elif cont['type'] == 'image:color':
            bgr = cont['value']
            if isinstance(bgr, str):
                assert re.search(r'^#[0-9a-fA-F]{6}$', bgr)
                r, g, b = int(bgr[1:3], 16), int(bgr[3:5], 16), int(bgr[5:7], 16)
                bgr = [b, g, r]
            cont['asset'] = gen_mono_color(config['size'][0], config['size'][1], bgr)
        elif cont['type'] == 'image:tti':
            text = cont['value']
            print(f'TTI：{text}')
            cont['asset'] = tti(text)
        elif cont['type'] == 'audio:blank':
            cont['asset'] = gen_blank_audio(cont['value'])
            
    config['contents'] = [
        c for c in config['contents']
        if 'asset' in c
    ]
    
    # 剪裁图片
    w, h = config['size']
    mode = config['resizeMode']
    for c in config['contents']:
        if c['type'].startswith('image:'):
            c['asset'] = resize_img(c['asset'], w, h, mode)
        if c['type'].startswith('video:'):
            c['asset'] = resize_video_noaud(c['asset'], w, h, mode=mode)

def tts(text):
    hash_ = hashlib.md5(text.encode('utf8')).hexdigest()
    cache = load_tts(hash_, 'none')
    if cache: return cache
    data = edgetts_cli(text)
    save_tts(hash_, 'none', data)
    return data

def split_text_even(text, maxlen):
    textlen = len(text)
    num = math.ceil(textlen / maxlen)
    reallen = math.ceil(textlen / num)
    res = [text[i:i+reallen] for i in range(0, textlen, reallen)]
    return res
        

def srt_time_fmt(num):
    sec = int(num) % 60
    min_ = int(num) // 60 % 60
    hr = int(num) // 3600
    msec = int(num * 1000) % 1000
    return f'{hr:02d}:{min_:02d}:{sec:02d},{msec:03d}'

# 生成字幕
def gen_srt(audios):
    # 提取 audios 数组中的字幕
    subs = [
        {
            'text': a['subtitle'],
            'len': a['len'],
        }
        for a in audios
    ]
    # 将每个字幕按指定长度分割
    for s in subs:
        text = s['text']
        if not text: continue
        parts = split_text_even(text, config['subtitleMaxLen'])
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
    

# 内容成帧
def contents2frame(contents):
    frames = []
    for c in contents:
        if c['type'].startswith('image:'):
            frames.append({
                'image': c['asset'],
                'audios': [],
            })
        elif c['type'].startswith('video:'):
            frames.append({
                'video_noaud': c['asset'],
                'audios': [],
            })
        elif c['type'].startswith('audio:'):
            if len(frames) == 0: continue
            frames[-1]['audios'].append({
                'audio': c['asset'],
                'len': audio_len(c['asset']),
                'subtitle': c['value'] if c['type'] == 'audio:tts' else '',
            })
    for f in frames:
        f['len'] = sum([a['len'] for a in f['audios']])
        if 'video_noaud' in f:
            f['video_noaud'] = repeat_video_nsec(f['video_noaud'], f['len'])
        else:
            f['video_noaud'] = img_nsec_2video(f['image'], f['len'], config['fps'])
        f['audio'] = (
            f['audios'][0]['audio'] 
            if len(f['audios']) == 1 
            else ffmpeg_cat([a['audio'] for a in f['audios']], 'mp3')
        )
        f['video'] = ffmpeg_merge_video_audio(f['video_noaud'], f['audio'], audio_fmt='mp3')
        f['srt'] = gen_srt(f['audios'])
        f['video'] = ffmpeg_add_srt(f['video'], f['srt'])
    return frames

def repeat_video_nsec(video, total):
    nsec = ffmpeg_get_info(video)['duration']
    if total == nsec:
        return video
    elif total < nsec:
        return slice_video_noaud(video, total)
    nrepeat = int(total // nsec)
    new_len = total / nrepeat
    multi = nsec / new_len
    one_video = speedup_video_noaud(video, multi)
    return ffmpeg_cat([one_video] * nrepeat)

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

def update_config(user_cfg, cfg_dir):
    global tts
    global tti
    
    config.update(user_cfg)
    if not config['contents']:
        raise AttributeError('内容为空，无法生成')
        
    for cont in config['contents']:
        if cont['type'].endswith(':file'):
            cont['value'] = path.join(cfg_dir, cont['value'])
    if config['header']:
        config['header'] = path.join(cfg_dir, config['header'])
    if config['footer']:
        config['footer'] = path.join(cfg_dir, config['footer'])
        
    if config['external']:
        mod_fname = path.join(cfg_dir, config['external'])
        exmod = load_module(mod_fname)
        if hasattr(exmod, 'tts'): tts = exmod.tts
        if hasattr(exmod, 'tti'): tti = exmod.tti

def autovideo(args):
    cfg_fname = args.config
    if not cfg_fname.endswith('.yml'):
        print('请提供 YAML 文件')
        return
    cfg_dir = path.dirname(cfg_fname)
    user_cfg = yaml.safe_load(open(cfg_fname, encoding='utf8').read())
    update_config(user_cfg, cfg_dir)
        
    # 素材预处理
    preproc_asset(config)
    # 转换成帧的形式
    frames = contents2frame(config['contents'])
    # 组装视频
    video = make_video(frames)
    if config['format'] != 'mp4':
        video = ffmpeg_conv_fmt(video, 'mp4',  config['format'])
    # 写文件
    video_fname = fname_escape(config['name']) + '.' + config['format']
    print(video_fname)
    open(video_fname, 'wb').write(video)
    