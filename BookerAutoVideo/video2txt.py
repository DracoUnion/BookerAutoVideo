# import paddle
import whisper
import re
import traceback
import copy
import os
import re
import hashlib
from os import path
from multiprocessing import Pool
from .util import *
from .keyframe import *
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

def video2txt_handle(args):
    if path.isdir(args.fname):
        video2txt_dir(args)
    else: 
        video2txt_file(args)

def video2txt_dir(args):
    dir = args.fname
    fnames = os.listdir(dir)
    pool = Pool(args.threads)
    for fname in fnames:
        args = copy.deepcopy(args)
        args.fname = path.join(dir, fname)
        pool.apply_async(video2txt_file_safe, [args])
    pool.close()
    pool.join()
    
def video2txt_file_safe(args):
    try: video2txt_file(args)
    except: traceback.print_exc()

def video2txt_file(args):
    fname = args.fname
    if not (path.isfile(fname) and is_video_or_audio(fname)):
        print('请提供音频或视频文件')
        return
    print(fname)
    if args.device == 'privateuseone':
        import torch_directml
    # 语音识别
    model = whisper.load_model(args.asr_model, device=args.device)
    r = model.transcribe(fname, fp16=False, language=args.language)
    words = [
        {'time': s['start'], 'text': s['text']}
        for s in r['segments']
    ]
    print(words)
    # 获取关键帧
    if not args.no_image and is_video(fname):
        frames = extract_keyframe(args)
        words += frames
        words.sort(key=lambda x: x['time'])
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
    title = path.basename(fname)
    title = re.sub(r'\.\w+$', '', title)
    title_hash = hashlib.md5(title.encode('utf8')).hexdigest()
    for i in range(len(words) - 1, 0, -1):
        if 'text' in words[i] and 'text' in words[i - 1]:
            words[i - 1]['text'] += '，' + words[i]['text']
            del words[i]
    imgs = {}
    for i, w in enumerate(words):
        if 'img' in w: 
           imgname = f'{title_hash}_{i}.png'
           imgs[imgname] = w['img']
           w['text'] = f'![](img/{imgname})'
        elif 'text' in w:
            w['text'] = stylish_text(w['text']) + '。'
    text = '\n\n'.join([w['text'] for w in words])
    text = f'# {title}\n\n{text}'
    print(text)
    nfname = re.sub(r'\.\w+$', '', fname) + '.md'
    open(nfname , 'w', encoding='utf8').write(text)
    print(nfname + '.md')
    imgdir = path.join(path.dirname(fname), 'img')
    safe_mkdir(imgdir)
    for imgname, img in imgs.items():
        img_fname = path.join(imgdir, imgname)
        print(img_fname)
        open(img_fname, 'wb').write(img)
