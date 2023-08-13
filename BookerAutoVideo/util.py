import os
import sys
from os import path
import re
import shutil
import tempfile
import uuid
import imgyaso
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


def edgetts_cli(text, voice='zh-CN-XiaoyiNeural'):
    fname = path.join(tempfile.gettempdir(), uuid.uuid4().hex + '.mp3')
    cmd = [
        'edge-tts', '-t', text, '-v', voice, '--write-media', fname,
    ]
    print(f'cmd: {cmd}')
    subp.Popen(cmd, shell=True).communicate()
    res = open(fname, 'rb').read()
    safe_remove(fname)
    return res
    

def ffmpeg_pic2video(img, sec):
    prefix = uuid.uuid4().hex
    img_fname = path.join(tempfile.gettempdir(), prefix + '.png')
    open(img_fname, 'wb').write(img)
    vid_fname = path.join(tempfile.gettempdir(), prefix + '.mp4')
    cmd = [
        'ffmpeg', '-r', str(1 / sec), '-f', 'image2', '-i', img_fname, vid_fname,
    ]
    print(f'cmd: {cmd}')
    subp.Popen(cmd, shell=True).communicate()
    res = open(vid_fname, 'rb').read()
    safe_remove(img_fname)
    safe_remove(vid_fname)
    return res

def ffmpeg_cat_audio(audios):
    prefix = uuid.uuid4().hex
    for i, audio in enumerate(audios):
        fname = path.join(tempfile.gettempdir(), f'{prefix}-{i}.mp3')
        open(fname, 'wb').write(audio)
    audio_fnames = [
        path.join(tempfile.gettempdir(), f'{prefix}-{i}.mp3') 
        for i in range(len(audios))
    ]
    ofname = path.join(tempfile.gettempdir(), f'prefix.mp3')
    cmd = [
        'ffmpeg', '-i', 
        'concat:' + '|'.join(audio_fnames), 
        '-c:a', 'copy', ofname,
    ]
    print(f'cmd: {cmd}')
    subp.Popen(cmd, shell=True).communicate()
    res = open(ofname, 'rb').read()
    safe_remove(ofname)
    for f in audio_fnames: safe_remove(f)
    return res

def ffmpeg_cat_videos(videos):
    prefix = uuid.uuid4().hex
    for i, video in enumerate(videos):
        fname = path.join(tempfile.gettempdir(), f'{prefix}-{i}.mp4')
        open(fname, 'wb').write(video)
    video_fnames = [
        path.join(tempfile.gettempdir(), f'{prefix}-{i}.mp4') 
        for i in range(len(videos))
    ]
    ofname = path.join(tempfile.gettempdir(), f'prefix.mp4')
    cmd = [
        'ffmpeg', '-i', 
        'concat:' + '|'.join(video_fnames), 
        '-c:a', 'copy', '-v:a', 'copy', ofname,
    ]
    print(f'cmd: {cmd}')
    subp.Popen(cmd, shell=True).communicate()
    res = open(ofname, 'rb').read()
    safe_remove(ofname)
    for f in video_fnames: safe_remove(f)
    return res

def ffmpeg_merge_video_audio(video, audio):
    tmpdir = path.join(tempfile.gettempdir(), uuid.uuid4().hex)
    safe_mkdir(tmpdir)
    vfname = path.join(tmpdir, 'video.mp4')
    open(vfname, 'wb').write(video)
    afname = path.join(tmpdir, 'audio.mp4')
    open(afname, 'wb').write(audio)
    res_fname = path.join(tmpdir, 'merged.mp4')
    cmds = [
        ['ffmpeg', '-i', vfname, '-vcodec', 'copy', '-an', vfname, '-y'],
        ['ffmpeg', '-i', afname, '-acodec', 'copy', '-vn', afname, '-y'],
        ['ffmpeg', '-i', afname, '-i', vfname, '-vcodec', 'copy', '-acodec', 'copy', res_fname, '-y'],
    ]
    for cmd in cmds:
        print(f'cmd: {cmd}')
        subp.Popen(cmd, shell=True).communicate()
    res = open(res_fname, 'rb').read()
    safe_rmdir(tmpdir)
    return res



