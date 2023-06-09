import os
import sys
from os import path
import re
import shutil
import tempfile
import uuid
import imgyaso

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
