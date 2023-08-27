import cv2
import numpy as np
import argparse
import os
import re
import math
import uuid
import tempfile
import subprocess as subp
from moviepy.video.io.VideoFileClip import VideoFileClip
from os import path
from imgyaso import adathres, adathres_bts
from .util import *
import easyocr
import PIL

DIR_F = 'forward'
DIR_B = 'backward'
DIR_T = 'twoway'

ocr_reader = None
PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

def load_ocr():
    global ocr_reader
    ocr_reader = easyocr.Reader(['ch_sim', 'en'])

def text_ngram_diff(text1, text2, n=3):
    set1 = {text1[i:i+n] for i in range(0, len(text1) - n + 1)}
    set2 = {text2[i:i+n] for i in range(0, len(text2) - n + 1)}
    return len(set1 & set2) / len(set1 | set2)
    
def img2text(img):
    img = cv2.imencode(
        '.png', img, 
        [cv2.IMWRITE_PNG_COMPRESSION , 9]
    )[1]
    img = adathres_bts(bytes(img))
    res = ocr_reader.readtext(img)
    text = '\n'.join([line[1] for line in res])

def nsec2hms(nsec):
    nsec = int(nsec)
    h = nsec // 3600
    m = nsec // 60 % 60
    s = nsec % 60
    return f'{h}h{m:02d}m{s:02d}s'

def load_frames(fname, rate):
    cap = cv2.VideoCapture(fname) 
    if not cap.isOpened():
        raise Exception(f'无法打开文件 {fname}')
    frames = []
    tm = 0
    total = VideoFileClip(fname).duration
    while(tm < total):
        cap.set(cv2.CAP_PROP_POS_MSEC, tm * 1000)
        succ, img = cap.read()
        if not succ: break
        print(f'time {nsec2hms(tm)} loaded')
        frames.append({
            'idx': len(frames), 
            'time': tm,
            'img': img,
            'text': img2text(img),
        })
        tm += 1 / rate
    cap.release()
    return frames

def calc_frame_diffs(frames, args):
    direction = args.direction
    frames[0]['diff'] = 1
    for prev, curr in zip(frames[:-1], frames[1:]):
        curr['diff'] = text_ngram_diff(prev['text'], curr['text'])
    if direction == DIR_B:
        for curr, next in zip(frames[:-1], frames[1:]):
            curr['diff'] = next['diff']
        frames[-1]['diff'] = 1
    elif direction == DIR_T:
        for curr, next in zip(frames[:-1], frames[1:]):
            curr['diff'] = (curr['diff'] + next['diff']) / 2
        frames[-1]['diff'] = (frames[-1]['diff'] + 1) / 2
    
def extract_keyframe(args):
    fname = args.fname
    opti_mode = args.opti_mode
    # 从视频中读取帧
    frames = load_frames(fname, args.rate)
    # 计算差分
    calc_frame_diffs(frames, args)
    for f in frames:
        print(f"time {nsec2hms(f['time'])} diff: {f['diff']:.16f}")
    # 计算关键帧
    frames = [
        f for f in frames
        if f['diff'] >= args.thres
    ]
    # 优化图像
    for f in frames:
        img = cv2.imencode(
            '.png', f['img'], 
            [cv2.IMWRITE_PNG_COMPRESSION, 9]
        )[1]
        img = bytes(img)
        # 保证最小 1080p，不够就放大
        h, w = f['img'].shape[:2]
        scale = 1080 / min(h, w)
        f['img'] = opti_img(img, args.opti_mode, 8)
    return frames

def extract_keyframe_file(args):
    fname = args.fname
    opti_mode = args.opti_mode
    if not is_video(fname):
        print('请提供视频')
        return
    print(fname)
    load_ocr()
    frames = extract_keyframe(args)
    # 保存所有关键帧
    opath = re.sub(r'\.\w+$', '', fname) + '_keyframe'
    if not path.isdir(opath): os.mkdir(opath)
    for f in frames:
        ofname = path.join(opath, f'keyframe_{nsec2hms(f["time"])}.png')
        print(ofname)
        open(ofname, 'wb').write(f['img'])

    