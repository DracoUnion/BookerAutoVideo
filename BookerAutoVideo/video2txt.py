# import paddle
import re
import traceback
import copy
import os
import json
import hashlib
from os import path
from multiprocessing import Pool
import subprocess as subp
import json
from .util import *
from .keyframe import *
# from paddlespeech.cli.text.infer import TextExecutor 
from .sencevoice import *

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
    # pool = Pool(args.threads)
    for fname in fnames:
        # args = copy.deepcopy(args)
        args.fname = path.join(dir, fname)
        # pool.apply_async(video2txt_file_safe, [args])
        video2txt_file_safe(args)
    # pool.close()
    # pool.join()
    
def video2txt_file_safe(args):
    try: video2txt_file(args)
    except: traceback.print_exc()

def whisper_cpp(args):
    fname = args.fname
    wav_fname = path.join(tempfile.gettempdir(), uuid.uuid4().hex + '.wav')
    subp.Popen(
        ['ffmpeg', '-i', fname, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '2', wav_fname, '-y'],
        shell=True,
    ).communicate()
    if not path.isfile(wav_fname):
        raise FileNotFoundError(f'{fname} 转换失败')
    subp.Popen(
        ['whisper', '-f', wav_fname, '-m', args.whisper, '-t', str(args.threads), '-l', args.lang, '-oj'],
        shell=True, 
    ).communicate()
    json_fname = wav_fname + '.json'
    if not path.isfile(wav_fname):
        safe_remove(wav_fname)
        raise FileNotFoundError(f'{fname} 识别失败')
    res = json.loads(open(
        json_fname, 
        encoding='utf8',
        errors='ignore',
    ).read())
    safe_remove(wav_fname)
    safe_remove(json_fname)
    return [
        {
            'time': s['offsets']['from'] / 1000,
            'text': s['text'],
        } 
        for s in res['transcription']
    ]

def asr(args):
    # 加载缓存
    fname = args.fname
    hash_ = hashlib.md5(open(fname, 'rb').read()).hexdigest()
    words = load_asr(hash_)
    if not words:
        srt_fname = re.sub(r'\.\w+$', '', fname) + '.srt'
        if path.isfile(srt_fname):
            srt = open(srt_fname, encoding='utf8').read()
            print(srt)
            words = parse_srt(srt)
        else:
            # 语音识别
            words = sencevoice(args)
        save_asr(hash_, words)
    return words


def video2txt_file(args):
    fname = args.fname
    if not (path.isfile(fname) and is_video_or_audio(fname)):
        print('请提供音频或视频文件')
        return
    print(fname)
    nfname = re.sub(r'\.\w+$', '', fname) + '.md'
    if path.isfile(nfname):
        print(f'{nfname} 已存在')
        return
    # 启动 ASR
    words = asr(args)
    # 获取关键帧
    if not args.no_image and is_video(fname):
        frames = extract_keyframe(args)
        words += frames
        words.sort(key=lambda x: x['time'])
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
            w['text'] = stylish_text(w['text'])
    text = '\n\n'.join([w['text'] for w in words])
    text = f'# {title}\n\n{text}'
    print(text)
    open(nfname , 'w', encoding='utf8').write(text)
    print(nfname + '.md')
    imgdir = path.join(path.dirname(fname), 'img')
    safe_mkdir(imgdir)
    for imgname, img in imgs.items():
        img_fname = path.join(imgdir, imgname)
        print(img_fname)
        if isinstance(img, np.ndarray):
            img = bytes(cv2.imencode(
                '.png', img, 
                [cv2.IMWRITE_PNG_COMPRESSION, 9]
            )[1])
        open(img_fname, 'wb').write(img)
