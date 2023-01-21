import os
import sys
from os import path
import shutil
import tempfile
import uuid

def is_video(fname):
    ext = [
        'mp4', 'm4v', '3gp', 'mpg', 'flv', 'f4v', 
        'swf', 'avi', 'gif', 'wmv', 'rmvb', 'mov', 
        'mts', 'm2t', 'webm', 'ogg', 'mkv', 
    ]
    m = re.search(r'\.(\w+)$', fname)
    return bool(m and m.group(1) in ext)

def is_audio(fname):
    ext = [
        'mp3', 'aac', 'ape', 'flac', 'wav', 'wma', 'amr', 'mid',
    ]
    m = re.search(r'\.(\w+)$', fname)
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