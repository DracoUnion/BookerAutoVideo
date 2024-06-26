import cv2
import numpy as np
import argparse
import os
import re
import math
import uuid
import tempfile
import subprocess as subp
from os import path
from imgyaso import adathres, adathres_bts
from .util import *
from .imgsim import *
import easyocr
import PIL

DIR_F = 'forward'
DIR_B = 'backward'
DIR_T = 'twoway'

ocr_reader = None
PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

def img2text(img):
    global ocr_reader
    if not ocr_reader:
        ocr_reader = easyocr.Reader(['ch_sim', 'en'])
        
    img = cv2.imencode(
        '.png', img, 
        [cv2.IMWRITE_PNG_COMPRESSION , 9]
    )[1]
    img = adathres_bts(bytes(img))
    res = ocr_reader.readtext(img)
    text = '\n'.join([line[1] for line in res])
    return text

def nsec2hms(nsec):
    nsec = int(nsec)
    h = nsec // 3600
    m = nsec // 60 % 60
    s = nsec % 60
    return f'{h}h{m:02d}m{s:02d}s'

def calc_diffs(frames, args, diff_func, src_prop='img', diff_prop='diff'):
    direction = args.direction
    frames[0][diff_prop] = 1
    for prev, curr in zip(frames[:-1], frames[1:]):
        curr[diff_prop] = diff_func(prev[src_prop], curr[src_prop])
    if direction == DIR_B:
        for curr, next in zip(frames[:-1], frames[1:]):
            curr[diff_prop] = next[diff_prop]
        frames[-1][diff_prop] = 1
    elif direction == DIR_T:
        for curr, next in zip(frames[:-1], frames[1:]):
            curr[diff_prop] = (curr[diff_prop] + next[diff_prop]) / 2
        frames[-1][diff_prop] = (frames[-1][diff_prop] + 1) / 2

def calc_img_diffs(frames, args):
    calc_diffs(frames, args, lambda x, y: 1 - pixel_l1_sim(x, y), 'img', 'diff')

def calc_text_diffs(frames, args):
    calc_diffs(frames, args, lambda x, y: word_ngram_diff(x, y), 'text', 'textDiff')

def extract_keyframe(args):
    print(args)
    fname = args.fname
    # 从视频中读取帧
    imgs, _ = get_video_imgs(fname, args.rate)
    # 第一个过滤：根据锐度和丰度过滤出幻灯片
    frames = [
        {
            'idx': i,
            'time': i / args.rate,
            'img': img,
        } 
        for i, img in enumerate(imgs)
        if sharpness(img) <= args.sharpness and
           colorfulness(img) <= args.colorfulness
    ]
    # 第二个过滤：根据帧间差过滤重复幻灯片
    while True:
        nframe = len(frames)
        # 计算差分
        calc_img_diffs(frames, args)
        for f in frames:
            print(f"time {nsec2hms(f['time'])} diff: {f['diff']:.16f}")
        print('=' * 30)
        # 计算关键帧
        frames = [
            f for f in frames
            if f['diff'] >= args.thres
        ]
        if len(frames) == nframe: break
    # 第三个过滤：OCR 之后根据文字过滤语义相似幻灯片
    if args.ocr > 0:
        for f in frames:
            f['text'] = img2text(f['img'])
        frames = [f for f in frames if f['text']]
        for f in frames:
            print(f"time {nsec2hms(f['time'])} text: {json.dumps(f['text'], ensure_ascii=False)}")
        print('=' * 30)
        while True:
            nframe = len(frames)
            # 计算文字差异
            calc_text_diffs(frames, args)
            for f in frames:
                print(f"time {nsec2hms(f['time'])} textDiff: {f['textDiff']:.16f}")
            print('=' * 30)
            # 计算关键帧
            frames = [
                f for f in frames
                if f['textDiff'] >= args.ocr
            ]
            if nframe == len(frames): break
    # 优化图像
    for f in frames:
        img = cv2.imencode(
            '.png', f['img'], 
            [cv2.IMWRITE_PNG_COMPRESSION, 9]
        )[1]
        f['img'] = bytes(img)
        f['img'] = opti_img(img, args.opti_mode, 8)
    return frames

def extract_keyframe_file(args):
    fname = args.fname
    opti_mode = args.opti_mode
    if not is_video(fname):
        print('请提供视频')
        return
    print(fname)
    frames = extract_keyframe(args)
    # 保存所有关键帧
    opath = re.sub(r'\.\w+$', '', fname) + '_keyframe'
    if not path.isdir(opath): os.mkdir(opath)
    for f in frames:
        ofname = path.join(opath, f'keyframe_{nsec2hms(f["time"])}.png')
        print(ofname)
        open(ofname, 'wb').write(f['img'])

    