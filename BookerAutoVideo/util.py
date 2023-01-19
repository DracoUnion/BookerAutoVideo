import os
import sys
from os import path
import shutil
import tempfile
import uuid

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