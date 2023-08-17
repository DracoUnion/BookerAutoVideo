import os
import sys
from os import path
import re
import shutil
import tempfile
import uuid
import imgyaso
import cv2
import numpy  as np
import subprocess  as subp

def stylish_text(text):
    text = (
        text.replace(',', '，')
            .replace('.', '。')
            .replace('?', '？')
            .replace('!', '！')
    )
    text = re.sub(r'(.{50,100}(?:，|。|！|？))', r'\1\n\n', text)
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
        path.join(tmpdir, f'{i}.{fmt}') 
        for i in range(len(videos))
    ]
    video_li_fname = path.join(tmpdir, f'list.txt') 
    open(video_li_fname, 'w', encoding='utf8').write('\n'.join(video_fnames))
    ofname = path.join(tmpdir, f'res.{fmt}')
    cmd = [
        'ffmpeg', '-f', 'concat',
        '-i', video_li_fname,
        '-c', 'copy', ofname, '-y',
    ]
    print(f'cmd: {cmd}')
    subp.Popen(cmd, shell=True).communicate()
    res = open(ofname, 'rb').read()
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



