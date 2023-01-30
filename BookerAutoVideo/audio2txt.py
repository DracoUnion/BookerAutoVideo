# import paddle
import whisper
import re
import traceback
import copy
import os
import re
from os import path
from multiprocessing import Pool
from .util import *
# from paddlespeech.cli.text.infer import TextExecutor 

def merge_words(words, maxl=500):
    res = []
    st = 0
    l = 0
    for i, w in enumerate(words):
        if l >= maxl:
            res.append(''.join(words[st:i]))
            st = i
            l = 0
        l += len(w)
    res.append(''.join(words[st:]))
    return res

def audio2txt_handle(args):
    if path.isdir(args.fname):
        audio2txt_dir(args)
    else: 
        audio2txt_file(args)

def audio2txt_dir(args):
    dir = args.fname
    fnames = os.listdir(dir)
    pool = Pool(args.threads)
    for fname in fnames:
        args = copy.deepcopy(args)
        args.fname = path.join(dir, fname)
        pool.apply_async(audio2txt_file_safe, [args])
    pool.close()
    pool.join()
    
def audio2txt_file_safe(args):
    try: audio2txt_file(args)
    except: traceback.print_exc()

def audio2txt_file(args):
    fname = args.fname
    if not (path.isfile(fname) and is_video_or_audio(fname)):
        print('请提供音频或视频文件')
        return
    print(fname)
    # 语音识别
    model = whisper.load_model(args.model)
    r = model.transcribe(fname, fp16=False, language='Chinese')
    words = [s['text'] for s in r['segments']]
    print(f'words: {words}')
    # 标点修正
    '''
    words = merge_words(words)
    text_executor = TextExecutor()
    text = ''.join([
        text_executor(
            text=w,
            task='punc',
            model='ernie_linear_p3_wudao',
            device=paddle.get_device(),
        ) for w in words
    ])
    '''
    # 排版
    text = stylish_text('，'.join(words))
    title = path.basename(fname)
    title = re.sub(r'\.\w+$', '', title)
    text = f'# {title}\n\n{text}'
    nfname = re.sub(r'\.\w+$', '', fname) + '.md'
    open(nfname , 'w', encoding='utf8').write(text)
    print(nfname + '.md')
