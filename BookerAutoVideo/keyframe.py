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
from imgsim import *
import PIL

DIR_F = 'forward'
DIR_B = 'backward'
DIR_T = 'twoway'

PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

    

def nsec2hms(nsec):
    nsec = int(nsec)
    h = nsec // 3600
    m = nsec // 60 % 60
    s = nsec % 60
    return f'{h}h{m:02d}m{s:02d}s'

def calc_frame_diffs(frames, args):
    direction = args.direction
    frames[0]['diff'] = 1
    for prev, curr in zip(frames[:-1], frames[1:]):
        curr['diff'] = 1 - pixel_l1_sim(prev['text'], curr['text'])
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
    imgs, _ = get_video_imgs(fname, args.rate)
    frames = [
        {
            'idx': i,
            'time': i / args.rate,
            'img': img,
        } 
        for i, img in enumerate(imgs)
        if sharpness(img) >= args.sharpness and
           colorfulness(img) <= args.colorfulness
    ]
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

    