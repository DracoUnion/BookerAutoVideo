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

DATA_DIR = path.join(tempfile.gettempdir(), 'autovideo')

IMWRITE_PNG_FLAG = [cv2.IMWRITE_PNG_COMPRESSION, 9]

def ensure_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) \
           if img.ndim == 3 else img

def stylish_text(text):
    # 英文标点转成中文，并补齐末尾句号
    text = (
        text.replace(',', '，')
            .replace('.', '。')
            .replace('?', '？')
            .replace('!', '！')
    ) + '。'
    # 连续多个标点只取一个
    text = re.sub(r'[。，！？]{2,}', lambda m: m.group()[0], text)
    # 50~100 个字为一段
    text = re.sub(r'(.{50,100}[。，！？])', r'\1\n\n', text)
    # 段尾逗号变句号
    text = re.sub(r'，$', '。', text, flags=re.M)
    return text

def opti_img(img, mode, colors):
    if mode == 'quant':
        return imgyaso.pngquant_bts(img, colors)
    elif mode == 'grid':
        return imgyaso.grid_bts(img)
    elif mode == 'trunc':
        return imgyaso.trunc_bts(img, colors)
    elif mode == 'thres':
        return imgyaso.adathres_bts(img)
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


def edgetts_cli(text, voice='zh-CN-XiaoyiNeural', fmt='mp3'):
    fname = path.join(tempfile.gettempdir(), uuid.uuid4().hex + '.' + fmt)
    cmd = [
        'edge-tts', '-t', text, '-v', voice, '--write-media', fname,
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
    subp.Popen(cmd, shell=True).communicate()
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
    subp.Popen(cmd, shell=True).communicate()
    res = open(ofname, 'rb').read()
    safe_rmdir(tmpdir)
    return res

def ffmpeg_add_srt(video, srt, video_fmt='mp4'):
    tmpdir = path.join(tempfile.gettempdir(), uuid.uuid4().hex)
    safe_mkdir(tmpdir)
    vfname = path.join(tmpdir, f'video.{video_fmt}')
    open(vfname, 'wb').write(video)
    sfname = path.join(tmpdir, f'subtitle.srt')
    open(sfname, 'w', encoding='utf8').write(srt)
    res_fname = path.join(tmpdir, f'merged.{video_fmt}')
    cmd = ['ffmpeg', '-i', f'video.{video_fmt}', '-vf', f'subtitles=subtitle.srt', res_fname, '-y']
    '''
    cmd = [
        'ffmpeg', '-i', vfname, '-i', sfname, 
        '-c', 'copy', res_fname, '-y',
    ]
    if video_fmt == 'mp4': cmd += ['-c:s', 'mov_text']
    '''
    print(f'cmd: {cmd}')
    subp.Popen(cmd, shell=True, cwd=tmpdir).communicate()
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
        subp.Popen(cmd, shell=True).communicate()
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



def load_tts(hash_, voice):
    fname = path.join(DATA_DIR, f'{hash_}-{voice}')
    if path.isfile(fname):
        return open(fname, 'rb').read()
    else:
        return None
        
def save_tts(hash_, voice, data):
    safe_mkdir(DATA_DIR)
    fname = path.join(DATA_DIR, f'{hash_}-{voice}')
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
            cmd, stdout=subp.PIPE, stderr=subp.PIPE, shell=True
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
    
# 缩放到最小填充尺寸并剪裁
def resize_img_fill(img, nw, nh):
    fmt_bytes = isinstance(img, bytes)
    if fmt_bytes:
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    h, w, *_ = img.shape
    # 计算宽高的缩放比例，使用较大值等比例缩放
    x_scale = nw / w
    y_scale = nh / h
    scale = max(x_scale, y_scale)
    rh, rw = int(h * scale), int(w * scale)
    img = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_CUBIC)
    # 剪裁成预定大小
    cut_w = rw - nw
    cut_h = rh - nh
    img = img[
        cut_h // 2 : cut_h // 2 + nh,
        cut_w // 2 : cut_w // 2 + nw,
    ]
    if fmt_bytes:
        img = bytes(cv2.imencode('.png', img, IMWRITE_PNG_FLAG)[1])
    return img

# 缩放到最大包围并填充
def resize_img_wrap(img, nw, nh):
    fmt_bytes = isinstance(img, bytes)
    if fmt_bytes:
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    h, w, *_ = img.shape
    # 计算宽高的缩放比例，使用较小值等比例缩放
    x_scale = nw / w
    y_scale = nh / h
    scale = min(x_scale, y_scale)
    rh, rw = int(h * scale), int(w * scale)
    img = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_CUBIC)
    # 填充到预定大小
    pad_w = nw - rw
    pad_h = nh - rh
    img = cv2.copyMakeBorder(
        img, pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2, 
        cv2.BORDER_CONSTANT, None, (0,0,0)
    ) 
    if fmt_bytes:
        img = bytes(cv2.imencode('.png', img, IMWRITE_PNG_FLAG)[1])
    return img
    
def resize_img(img, nw, nh, mode='wrap'):
    assert mode in ['wrap', 'fill']
    func_resize_img = resize_img_wrap if mode == 'wrap' else resize_img_fill
    return func_resize_img(img, nw, nh)
    
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
    func_resize_img = resize_img_wrap if mode == 'wrap' else resize_img_fill
    imgs, fps = get_video_imgs(video, fps)
    imgs = [func_resize_img(img, nw, nh) for img in imgs]
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
    imgs = [imgs[int(i)] for i in np.arange(0, len(imgs), nulti)]
    video = imgs2video(imgs, fps)
    