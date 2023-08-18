import yaml
from os import path
import sys
import cv2
import math
import numpy as np
import librosa
from io import BytesIO
from moviepy.editor import *
from .autovideo_config import config
from .util import *
from scipy.io import wavfile
from EpubCrawler.util import request_retry
from moviepy.video.io.VideoFileClip import VideoFileClip, AudioFileClip

DIR = path.abspath(path.dirname(__file__))
RE_MD_IMG = r'!\[.*?\]\((.*?)\)'
RE_MD_TITLE = r'^#+ (.+?)$'
RE_MD_PRE = r'```[\s\S]+?```'
RE_MD_TR = r'^\|.+?\|$'
RE_MD_PREFIX = r'^\s*(\+|\-|\*|\d+\.|\>|\#+)'
RE_SENT_DELIM = r'\n|。|\?|？|;|；|:|：|!|！'

exmod = None

def audio_len(data):
    y, sr = librosa.load(BytesIO(data))
    return librosa.get_duration(y=y, sr=sr)

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
        

# 素材预处理
def preproc_asset(config):
    # 加载或生成内容
    for cont in config['contents']:
        if cont['type'].endswith(':file'):
            cont['asset'] = open(cont['value'], 'rb').read()
        elif cont['type'].endswith(':url'):
            url = cont['value']
            print(f'下载：{url}')
            cont['asset'] = request_retry('GET', url).content
        elif cont['type'] == 'audio:tts':
            text = cont['value']
            print(f'TTS：{text}')
            cont['asset'] = tts(text)
        elif cont['type'] == 'image:external':
            text = cont['value']
            print(f'Ex：{text}')
            cont['asset'] = exmod.txt2img(text)
            
    config['contents'] = [
        c for c in config['contents']
        if 'asset' in c
    ]
    
    # 剪裁图片
    for c in config['contents']:
        if c['type'].startswith('image:'):
            c['asset'] = trim_img(c['asset'])
    
    # 如果第一张不是图片，则提升第一个图片
    idx = -1
    for i, c in enumerate(config['contents']):
        if c['type'].startswith('image:'):
            idx = i
            break
    if idx == -1:
        print('内容中无图片，无法生成视频')
        sys.exit()
    if idx != 0:
        c = config['contents'][idx]
        del config['contents'][idx]
        config['contents'].insert(0, c)

def tts(text):
    hash_ = hashlib.md5(text.encode('utf8')).hexdigest()
    cache = load_tts(hash_, 'none')
    if cache return cache
    data = edgetts_cli(text)
    save_tts(hash_, 'none', data)
    return data

# 剪裁图片
def trim_img(img):
    if config['resizeMode'] == 'wrap':
        return resize_img_wrap(img)
    else:
        return resize_img_cut(img)

def resize_img_cut(img):
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    h, w, *_ = img.shape
    # 计算宽高的缩放比例，使用较大值等比例缩放
    x_scale = config['size'][0] / w
    y_scale = config['size'][1] / h
    scale = max(x_scale, y_scale)
    nh, nw = int(h * scale), int(w * scale)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    # 剪裁成预定大小
    cut_w = nw - config['size'][0]
    cut_h = nh - config['size'][1]
    img = img[
        cut_h // 2 : cut_h // 2 + config['size'][1],
        cut_w // 2 : cut_w // 2 + config['size'][0],
    ]
    img = bytes(cv2.imencode('.png', img)[1])
    return img


# 剪裁图片
def resize_img_wrap(img):
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    h, w, *_ = img.shape
    # 计算宽高的缩放比例，使用较小值等比例缩放
    x_scale = config['size'][0] / w
    y_scale = config['size'][1] / h
    scale = min(x_scale, y_scale)
    nh, nw = int(h * scale), int(w * scale)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    # 填充到预定大小
    pad_w = config['size'][0] - nw
    pad_h = config['size'][1] - nh
    img = cv2.copyMakeBorder(
        img, pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2, 
        cv2.BORDER_CONSTANT, None, (0,0,0)
    ) 
    img = bytes(cv2.imencode('.png', img)[1])
    return img

# 内容成帧
def contents2frame(contents):
    frames = []
    for c in contents:
        if c['type'].startswith('image:'):
            frames.append({
                'image': c['asset'],
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
    return frames

def pics2video(frames):
    ofname = path.join(tempfile.gettempdir(), uuid.uuid4().hex + '.mp4')
    fmt = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    vid = cv2.VideoWriter(ofname, fmt, config['fps'], config['size'])
    for f in frames:
        img = cv2.imdecode(np.frombuffer(f['image'], np.uint8), cv2.IMREAD_COLOR)
        ntimes = math.ceil(config['fps'] * f['len'])
        for _ in range(ntimes): vid.write(img)
    vid.release()
    res = open(ofname, 'rb').read()
    safe_remove(ofname)
    return res

# 组装视频
def make_video(frames):
    clips = []
    # 图像部分
    video = pics2video(frames)
    # 音频部分
    audios = [a['audio'] for f in frames for a in f['audios']]
    audio = ffmpeg_cat(audios, 'mp3')
    # 合并音视频
    video = ffmpeg_merge_video_audio(video, audio, audio_fmt='mp3')
    # 添加字幕 
    # TODO
    # 合并片头片尾
    if config['header']:
        header = open(config['header'], 'rb').read()
        video = ffmpeg_cat([header, video])
    if config['footer']:
        footer = open(config['footer'], 'rb').read()
        video = ffmpeg_cat([video, footer])
    return video

def update_config(user_cfg, cfg_dir):
    global exmod
    
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
    