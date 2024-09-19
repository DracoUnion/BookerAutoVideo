import os
import sys
from os import path
import re
import shutil
import tempfile
import uuid
import imgyaso
import cv2
import math
import numpy  as np
import subprocess  as subp
from io import BytesIO
from scipy.io import wavfile
import json
import openai
import httpx
import base64

DATA_DIR = path.join(tempfile.gettempdir(), 'autovideo')

IMWRITE_PNG_FLAG = [cv2.IMWRITE_PNG_COMPRESSION, 9]

DIR = path.abspath(path.dirname(__file__))
RE_MD_IMG = r'!\[.*?\]\((.*?)\)'
RE_MD_TITLE = r'^#+ (.+?)$'
RE_MD_PRE = r'```\w*[\s\S]+?```'
RE_MD_TR = r'^\|.+?\|$'
RE_MD_PREFIX = r'^\s*(\+|\-|\*|\d+\.|\>|\#+)'
RE_SENT_DELIM = r'\n|。|\?|？|;|；|:|：|!|！'
RE_MD_PIC = r'!\[[^\]]*\]\([^\)]*\)'
RE_MD_LINK = r'(?<!!)\[([^\]]*)\]\([^\)]*\)'
RE_MD_BI = r'(?<!\\)\*+'

def ensure_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) \
           if img.ndim == 3 else img

# 英文标点转成中文
def punc_en2zh(text):
    return  (
        text.replace(',', '，')
            .replace('.', '。')
            .replace('?', '？')
            .replace('!', '！')
            .replace(';', '；')
            .replace(':', '：')
    )

# 
def add_end_punc(text):
    pass

def stylish_text(text):
    # 英文标点转成中文，并补齐末尾句号
    text = punc_en2zh(text) + '。'
    # 连续多个标点只取一个
    text = re.sub(r'[。，！？：；]{2,}', lambda m: m.group()[0], text)
    # 50~100 个字为一段
    text = re.sub(r'(.{50,100}[。，！？：；])', r'\1\n\n', text)
    # 段尾逗号变句号
    text = re.sub(r'，$', '。', text, flags=re.M)
    return text

def opti_img(img, mode, colors):
    if mode == 'quant':
        return imgyaso.pngquant_bts(img, colors)
    elif mode == 'grid':
        return imgyaso.grid_bts(ensure_grayscale(img))
    elif mode == 'trunc':
        return imgyaso.trunc_bts(ensure_grayscale(img), colors)
    elif mode == 'thres':
        return imgyaso.adathres_bts(ensure_grayscale(img))
    else:
        return img

def is_pic(fname):
    ext = [
        'jpg', 'jpeg', 'jfif', 'png', 
        'gif', 'tiff', 'webp'
    ]
    m = re.search(r'\.(\w+)$', fname.lower())
    return bool(m and m.group(1) in ext)

def is_video(fname):
    ext = [
        'mp4', 'm4v', '3gp', 'mpg', 'flv', 'f4v', 
        'swf', 'avi', 'gif', 'wmv', 'rmvb', 'mov', 
        'mts', 'm2t', 'webm', 'ogg', 'mkv', 
    ]
    m = re.search(r'\.(\w+)$', fname.lower())
    return bool(m and m.group(1) in ext)

def is_audio(fname):
    ext = [
        'mp3', 'aac', 'ape', 'flac', 'wav', 'wma', 'amr', 'mid', 'm4a',
    ]
    m = re.search(r'\.(\w+)$', fname.lower())
    return bool(m and m.group(1) in ext)

def is_video_or_audio(fname):
    return is_video(fname) or is_audio(fname)

def safe_mkdir(dir):
    try: os.mkdir(dir)
    except: pass

def safe_remove(name):
    try: os.remove(name)
    except: pass

def safe_rmdir(name):
    try: shutil.rmtree(name)
    except: pass

def load_module(fname):
    if not path.isfile(fname) or \
        not fname.endswith('.py'):
        raise FileNotFoundError('外部模块应是 *.py 文件')
    tmpdir = path.join(tempfile.gettempdir(), 'load_module')
    safe_mkdir(tmpdir)
    if tmpdir not in sys.path:
        sys.path.insert(0, tmpdir)
    mod_name = 'x' + uuid.uuid4().hex
    nfname = path.join(tmpdir, mod_name + '.py')
    shutil.copy(fname, nfname)
    mod = __import__(mod_name)
    safe_remove(nfname)
    return mod
    
def find_cmd_path(name):
    delim = ';' if sys.platform == 'win32' else ':'
    suff = (
        ['.exe', '.cmd', '.ps1']
        if sys.platform == 'win32'
        else ['', '.sh']
    ) 
    for p in os.environ.get('PATH', '').split(delim):
        if any(path.isfile(path.join(p, name + s)) for s in suff):
            return p
    return ''


def edgetts_cli(
    text, voice='zh-CN-XiaoyiNeural', 
    rate='-10%', volume='+0%', fmt='mp3',
):
    fname = path.join(tempfile.gettempdir(), uuid.uuid4().hex + '.' + fmt)
    cmd = [
        'edge-tts', '-t', text, f'--rate={rate}',
        f'--volume={volume}',
        '-v', voice, '--write-media', fname,
    ]
    print(f'cmd: {cmd}')
    subp.Popen(cmd, shell=True).communicate()
    res = open(fname, 'rb').read()
    safe_remove(fname)
    return res

def ffmpeg_conv_fmt(video, from_, to):
    prefix = uuid.uuid4().hex
    from_fname = path.join(tempfile.gettempdir(), f'{prefix}.{from_}')
    to_fname = path.join(tempfile.gettempdir(), f'{prefix}.{to}')
    open(from_fname, 'wb').write(video)
    cmd = ['ffmpeg', '-i', from_fname, '-c', 'copy', to_fname, '-y']
    print(f'cmd: {cmd}')
    subp.Popen(cmd, shell=True, stdin=subp.PIPE).communicate()
    res = open(to_fname, 'rb').read()
    safe_remove(from_fname)
    safe_remove(to_fname)
    return res

def ffmpeg_cat(videos, fmt='mp4'):
    tmpdir = path.join(tempfile.gettempdir(), uuid.uuid4().hex)
    safe_mkdir(tmpdir)
    for i, video in enumerate(videos):
        fname = path.join(tmpdir, f'{i}.{fmt}')
        open(fname, 'wb').write(video)
    video_fnames = [
        'file ' + path.join(tmpdir, f'{i}.{fmt}').replace('\\', '\\\\')
        for i in range(len(videos))
    ]
    video_li_fname = path.join(tmpdir, f'list.txt') 
    open(video_li_fname, 'w', encoding='utf8').write('\n'.join(video_fnames))
    ofname = path.join(tmpdir, f'res.{fmt}')
    cmd = [
        'ffmpeg', '-f', 'concat', '-safe', '0',
        '-i', video_li_fname, '-c', 'copy', ofname, '-y',
    ]
    print(f'cmd: {cmd}')
    subp.Popen(cmd, shell=True, stdin=subp.PIPE).communicate()
    res = open(ofname, 'rb').read()
    safe_rmdir(tmpdir)
    return res

def ffmpeg_add_srt(video, srt, fontname='黑体', video_fmt='mp4'):
    tmpdir = path.join(tempfile.gettempdir(), uuid.uuid4().hex)
    safe_mkdir(tmpdir)
    vfname = path.join(tmpdir, f'video.{video_fmt}')
    open(vfname, 'wb').write(video)
    sfname = path.join(tmpdir, f'subtitle.srt')
    open(sfname, 'w', encoding='utf8').write(srt)
    res_fname = path.join(tmpdir, f'merged.{video_fmt}')
    cmd = [
        'ffmpeg', '-i', f'video.{video_fmt}', 
        '-vf', f"subtitles=subtitle.srt:force_style='FontName={fontname}'", 
        res_fname, '-y',
    ]
    '''
    cmd = [
        'ffmpeg', '-i', vfname, '-i', sfname, 
        '-c', 'copy', res_fname, '-y',
    ]
    if video_fmt == 'mp4': cmd += ['-c:s', 'mov_text']
    '''
    print(f'cmd: {cmd}')
    subp.Popen(cmd, shell=True, cwd=tmpdir, stdin=subp.PIPE).communicate()
    res = open(res_fname, 'rb').read()
    safe_rmdir(tmpdir)
    return res

def ffmpeg_merge_video_audio(video, audio, video_fmt='mp4', audio_fmt='mp4'):
    tmpdir = path.join(tempfile.gettempdir(), uuid.uuid4().hex)
    safe_mkdir(tmpdir)
    vfname = path.join(tmpdir, f'video.{video_fmt}')
    v0fname = path.join(tmpdir, f'video0.{video_fmt}')
    open(vfname, 'wb').write(video)
    afname = path.join(tmpdir, f'audio.{audio_fmt}')
    a0fname = path.join(tmpdir, f'audio0.{audio_fmt}')
    open(afname, 'wb').write(audio)
    res_fname = path.join(tmpdir, f'merged.{video_fmt}')
    cmds = [
        ['ffmpeg', '-i', vfname, '-vcodec', 'copy', '-an', v0fname, '-y'],
        ['ffmpeg', '-i', afname, '-acodec', 'copy', '-vn', a0fname, '-y'],
        ['ffmpeg', '-i', a0fname, '-i', v0fname, '-c', 'copy', res_fname, '-y'],
    ]
    for cmd in cmds:
        print(f'cmd: {cmd}')
        subp.Popen(cmd, shell=True, stdin=subp.PIPE).communicate()
    res = open(res_fname, 'rb').read()
    safe_rmdir(tmpdir)
    return res

def fname_escape(name):
    return name.replace('\\', '＼') \
               .replace('/', '／') \
               .replace(':', '：') \
               .replace('*', '＊') \
               .replace('?', '？') \
               .replace('"', '＂') \
               .replace('<', '＜') \
               .replace('>', '＞') \
               .replace('|', '｜')

def load_asr(hash_):
    fname = path.join(DATA_DIR, f'asr-{hash_}.json')
    if not path.isfile(fname):
        return None
    try:
        j = json.loads(open(fname, encoding='utf8').read())
        return j
    except:
        return None

def save_asr(hash_, j):
    safe_mkdir(DATA_DIR)
    fname = path.join(DATA_DIR, f'asr-{hash_}.json')
    open(fname, 'w', encoding='utf8').write(json.dumps(j))

def load_tti(hash_):
    fname = path.join(DATA_DIR, f'tti-{hash_}.png')
    if path.isfile(fname):
        return open(fname, 'rb').read()
    else:
        return None
        
def save_tti(hash_, data):
    safe_mkdir(DATA_DIR)
    fname = path.join(DATA_DIR, f'tti-{hash_}.png')
    open(fname, 'wb').write(data)

def load_tts(hash_, voice, volume, rate):
    fname = path.join(DATA_DIR, f'{hash_}-{voice}-{volume}-{rate}')
    if path.isfile(fname):
        return open(fname, 'rb').read()
    else:
        return None
        
def save_tts(hash_, voice, volume, rate, data):
    safe_mkdir(DATA_DIR)
    fname = path.join(DATA_DIR, f'{hash_}-{voice}-{volume}-{rate}')
    open(fname, 'wb').write(data)

def ffmpeg_get_info(video, fmt='mp4'):
    if isinstance(video, bytes):
        fname = path.join(tempfile.gettempdir(), uuid.uuid4().hex + '.' + fmt)
        open(fname, 'wb').write(video)
    else:
        fname = video
    cmd = ['ffmpeg', '-i', fname]
    print(f'cmd: {cmd}')
    r = subp.Popen(
            cmd, stdin=subp.PIPE,
            stdout=subp.PIPE, 
            stderr=subp.PIPE, shell=True
    ).communicate()
    text = r[1].decode('utf8')
    res = {}
    m = re.search(r'Duration:\x20(\d+):(\d+):(\d+)(.\d+)', text)
    if m:
        hr = int(m.group(1))
        min_ = int(m.group(2))
        sec = int(m.group(3))
        ms = float(m.group(4))
        res['duration'] = hr * 3600 + min_ * 60 + sec + ms
    m = re.search(r'(\d+)\x20fps', text)
    if m:
        res['fps'] = int(m.group(1))
    m = re.search(r'(\d+)\x20Hz', text)
    if m:
        res['sr'] = int(m.group(1))
    if isinstance(video, bytes):
        safe_remove(fname)
    return res
    
def resize_img_blur(img, nw, nh, *args, **kw):
    fmt_bytes = isinstance(img, bytes)
    if fmt_bytes:
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    h, w, *_ = img.shape
    # 生成模糊背景
    x_scale = nw / w
    y_scale = nh / h
    bg_scale = max(x_scale, y_scale)
    rh, rw = int(h * bg_scale), int(w * bg_scale)
    bg = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_CUBIC)
    # 剪裁成预定大小
    cut_w = rw - nw
    cut_h = rh - nh
    bg = bg[
        cut_h // 2 : cut_h // 2 + nh,
        cut_w // 2 : cut_w // 2 + nw,
    ]
    # 模糊
    bg = cv2.blur(bg, (199, 199))
    # 生成清晰前景
    fg_scale = min(x_scale, y_scale)
    rh, rw = int(h * fg_scale), int(w * fg_scale)
    fg = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_CUBIC)
    # 将背景覆盖到前景上
    pad_w = nw - rw
    pad_h = nh - rh
    bg[
        pad_h // 2 : pad_h // 2 + rh,
        pad_w // 2 : pad_w // 2 + rw 
    ] = fg
    if fmt_bytes:
        bg = bytes(cv2.imencode('.png', bg, IMWRITE_PNG_FLAG)[1])
    return bg
    
resize_img = resize_img_blur
    
def get_video_imgs(video, fps=0):
    if isinstance(video, bytes):
        fname = path.join(tempfile.gettempdir(), uuid.uuid4().hex + '.mp4')
        open(fname, 'wb').write(video)
    else:
        fname = video
    
    info = ffmpeg_get_info(fname)
    fps = fps or info['fps']
    
    cap = cv2.VideoCapture(fname) 
    if not cap.isOpened():
        raise Exception(f'无法打开文件 {fname}')
    tm = 0
    total = info['duration']
    imgs = []
    while tm < total:
        cap.set(cv2.CAP_PROP_POS_MSEC, tm * 1000)
        succ, img = cap.read()
        if not succ: break
        imgs.append(img)
        tm += 1 / fps
    
    cap.release()
    if isinstance(video, bytes):
        safe_remove(fname)
    return imgs, fps
    
def resize_video_noaud(video, nw, nh, fps=0, mode='wrap'):
    assert mode in ['wrap', 'fill']
    imgs, fps = get_video_imgs(video, fps)
    imgs = [resize_img_blur(img, nw, nh) for img in imgs]
    video = imgs2video(imgs, fps)
    return video
    
    
def imgs2video(imgs, fps=30):
    ofname = path.join(tempfile.gettempdir(), uuid.uuid4().hex + '.mp4')
    fmt = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    w, h = get_img_size(imgs[0])
    vid = cv2.VideoWriter(ofname, fmt, fps, [w, h])
    for img in imgs:
        if isinstance(img, bytes):
            img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        vid.write(img)
    vid.release()
    res = open(ofname, 'rb').read()
    safe_remove(ofname)
    return res
    
def get_img_size(img):
    if isinstance(img, bytes):
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    assert isinstance(img, np.ndarray) and img.ndim in [2, 3]
    return img.shape[1], img.shape[0]
    
def imgs_nsecs_2video(imgs, nsecs, fps=30):
    if isinstance(nsecs, int):
        nsecs = [nsecs] * len(imgs)
    assert len(imgs) == len(nsecs)
    counts = [math.ceil(fps * nsec) for nsec in nsecs]
    imgs = sum([[img] * count for img, count in zip(imgs, counts)])
    return imgs2video(imgs, fps)
 
def img_nsec_2video(img, nsec, fps=30):
    count = math.ceil(fps * nsec)
    imgs = [img] * count
    return imgs2video(imgs, fps)
    
def speedup_video_noaud(video, multi, fps=0):
    imgs, fps = get_video_imgs(video, fps)
    imgs = [imgs[int(i)] for i in np.arange(0, len(imgs), multi)]
    video = imgs2video(imgs, fps)
    return video
    
def slice_video_noaud(video, nsec, fps=0):
    imgs, fps = get_video_imgs(video, fps)
    count = nsec * fps
    video = imgs2video(imgs[:count], fps)

def split_text_even(text, maxlen):
    textlen = len(text)
    num = math.ceil(textlen / maxlen)
    reallen = math.ceil(textlen / num)
    res = [text[i:i+reallen] for i in range(0, textlen, reallen)]
    return res
    
def split_sentence(text, limit,  delims='。，、！？；：'):
    # 按照标点分割
    re_punc = ''.join([f'\\u{ord(ch):04x}' for ch in delims])
    sentences = re.split(f'(?<=[{re_punc}])', text)
    # 将后引号与前面的标点放到一起
    for i in range(1, len(sentences)):
        if sentences[i].startswith('”'):
            sentences[i] = sentences[i][:-1]
            sentences[i-1] += '”'
    # 如果单个句子长度超限，继续分割
    sentences = sum([
        ([s] if len(s) <= limit 
        else split_text_even(s, limit))
        for s in sentences
    ], [])
    # 组装不大于长度限制的文本
    res = ['']
    for s in sentences:
        if len(res[-1]) + len(s) <= limit:
            res[-1] += s
        else:
            res.append(s)
            
    return res

def word_ngram_diff(text1, text2, n=3):
    words1 = re.findall(r'\w+|[\u4e00-\u9fff]', text1)
    words2 = re.findall(r'\w+|[\u4e00-\u9fff]', text2)
    set1 = {tuple(words1[i:i+n]) for i in range(0, len(words1) - n + 1)}
    set2 = {tuple(words2[i:i+n]) for i in range(0, len(words2) - n + 1)}
    return 1 - len(set1 & set2) / (len(set1 | set2) + 1e-8)



def gen_mono_color(w, h, bgr):
    assert len(bgr) == 3
    img = np.zeros([h, w, 3])
    img[:, :] = bgr
    img = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])[1]
    return bytes(img)

def audio_len(data):
    return ffmpeg_get_info(data)['duration']


def gen_blank_audio(nsec, sr=22050, fmt='wav'):
    audio = np.zeros(int(nsec * sr), dtype=np.uint8)
    bio = BytesIO()
    wavfile.write(bio, sr, audio)
    audio = bio.getvalue()
    if fmt != 'wav':
        audio = ffmpeg_conv_fmt(audio, 'wav', fmt)
    return audio
    
def md2lines(cont):
    # 去掉代码块
    cont = re.sub(RE_MD_PRE, '', cont)
    # 去掉表格
    cont = re.sub(RE_MD_TR, '', cont, flags=re.M)
    # 去掉各种格式
    cont = re.sub(RE_MD_PREFIX, '', cont, flags=re.M)
    cont = re.sub(RE_MD_BI, '', cont, flags=re.M)
    # 去掉图片和链接
    cont = re.sub(RE_MD_PIC, '', cont, flags=re.M)
    cont = re.sub(RE_MD_LINK, r'\1', cont, flags=re.M)
    # 英文标点转中文
    # cont = punc_en2zh(cont)
    # 切分
    # lines = re.split(RE_SENT_DELIM, cont)
    lines = cont.split('\n')
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if l]

    # 补充末尾标点
    for i, l in enumerate(lines):
        if not re.search(r'[。，：；！？]$', l):
            lines[i] =  l + '。'
    # 合并较短的段落
    for i in range(1, len(lines)):
        if len(lines[i - 1]) < 100:
            lines[i] = lines[i - 1] + lines[i]
            lines[i - 1] = ''
    lines = [l for l in lines if l]

    return lines


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


def extname(fname):
    m = re.search(r'\.(\w+)$', fname.lower())
    return m.group(1) if m else ''

def call_dalle_retry(text, model_name, size, quality, retry=10, nothrow=True):
    for i in range(retry):
        try:
            print(f'tti: {json.dumps(text, ensure_ascii=False)}')
            client = openai.OpenAI(
                base_url=openai.base_url,
                api_key=openai.api_key,
                http_client=httpx.Client(
                    proxies=openai.proxy,
                    transport=httpx.HTTPTransport(local_address="0.0.0.0"),
                )
            )
            img = client.images.generate(
                model=model_name, 
                size='1024x1024',
                prompt=text,
                n=1,
                response_format='b64_json',
            ).data[0].b64_json
            # print(f'ans: {json.dumps(ans, ensure_ascii=False)}')
            return base64.b64decode(img)
        except Exception as ex:
            print(f'OpenAI retry {i+1}: {str(ex)}')
            if i == retry - 1 and not nothrow: raise ex

def set_openai_props(key=None, proxy=None, host=None):
    openai.api_key = key
    openai.proxy = proxy
    openai.base_url = host


RE_SRT = r'''
    ^(\d+)\n
    ^(\d+:\d+:\d+.\d+)\x20+\-+>\x20+(\d+:\d+:\d+.\d+)\n
    ^(.+)\n
'''

def parse_srt(srt):
    ms = re.findall(RE_SRT, srt, flags=re.M | re.VERBOSE)
    frames = []
    for m in ms:
        f = {
            'idx': int(m[0]),
            'start': hhmmss2time(m[1]),
            'end': hhmmss2time(m[2]),
            'text': m[3],
        }
        f['time'] = f['start']
        frames.append(f)
    return frames

def hhmmss2time(text):
    hhmmmss, ms = text.split('.')
    h, m, s = hhmmmss.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 10000