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
from scipy import signal
from imgyaso import adathres
from .imgsim import img_sim
from .util import *

DIR_F = 'forward'
DIR_B = 'backward'
DIR_T = 'twoway'

ext_modes = ['normthres', 'relthres', 'adathres', 'thres']

def frame_diff(prev, next, mode):
    if mode in img_sim:
        return 1 - img_sim[mode](prev, next)
    else:
        raise ValueError('差分模式未定义！')

'''
def smooth(frames, win_sz=20):
    frames = np.asarray(frames)
    assert frames.ndim == 3
    if win_sz % 2 == 0: win_sz += 1
    _, h, w = frames.shape
    kern = np.ones(win_sz)
    res = np.zeros_like(frames)
    for r in range(h):
        for c in range(w):
            pix = frames[:, r, c]
            sum = np.convolve(pix, kern, mode='same')
            count = np.convolve(np.ones_like(pix), kern, mode='same')
            res[:, r, c] = sum // count
    return res
'''

def nsec2hms(nsec):
    nsec = int(nsec)
    h = nsec // 3600
    m = nsec // 60 % 60
    s = nsec % 60
    return f'{h}h{m:02d}m{s:02d}s'

def load_frames(fname, rate, bw):
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
        })
        if bw: frames[-1]['grey'] = adathres(frames[-1]['grey'])
        tm += 1 / rate
    cap.release()
    return frames

def calc_frame_diffs(frames, args):
    direction, diff_mode = args.direction, args.diff_mode  
    frames[0]['diff'] = 1
    for prev, curr in zip(frames[:-1], frames[1:]):
        curr['diff'] = frame_diff(
            prev['img'], curr['img'], diff_mode,
        )
    if direction == DIR_B:
        for curr, next in zip(frames[:-1], frames[1:]):
            curr['diff'] = next['diff']
        frames[-1]['diff'] = 1
    elif direction == DIR_T:
        for curr, next in zip(frames[:-1], frames[1:]):
            curr['diff'] = (curr['diff'] + next['diff']) / 2
        frames[-1]['diff'] = (frames[-1]['diff'] + 1) / 2
    
def postproc_frame_diffs(frames, args):
    if args.extract_mode == 'normthres':
        max_diff = max([f['diff'] for f in frames if f['diff'] != 1])
        for f in frames: f['diff'] /= max_diff
    elif args.extract_mode == 'relthres':
        for f in frames: f['oriDiff'] = f['diff']
        frames[0]['diff'] = 1
        for prev, curr in zip(frames[:-1], frames[1:]):
            curr['diff'] = (curr['oriDiff'] - prev['oriDiff']) / curr['oriDiff']
    elif args.extract_mode == 'adathres': 
        diffs = [f['diff'] for f in frames]
        kern = np.ones([args.win_size])
        sum = np.convolve(diffs, kern, mode='same')
        cnt = np.convolve(np.ones_like(diffs), kern, mode='same')
        mean = sum / cnt
        for f, m in zip(frames, mean):
            f['diff'] = (f['diff'] - m) / f['diff']
    
def extract_keyframe(args):
    config_scene(args)
    config_thres(args)
    fname = args.fname
    ext_mode = args.extract_mode
    opti_mode = args.opti_mode
    # 从视频中读取帧
    frames = load_frames(fname, args.rate, args.bw)
    # 计算差分
    calc_frame_diffs(frames, args)
    postproc_frame_diffs(frames, args)
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
        # img = anime4k_scale(img, scale, args.threads)
        f['img'] = opti_img(img, args.opti_mode, 8)
    return frames

def extract_keyframe_file(args):
    fname = args.fname
    ext_mode = args.extract_mode
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
        
def config_scene(args):
    if args.scene == 'ppt':
        args.extract_mode = 'thres'
        args.diff_mode = 'pixel_l1'
        args.opti_mode = 'quant'
        args.rate = 0.2
        args.direction = DIR_B
        args.bw = False
        args.thres = 0.1
    elif args.scene == 'ppt2':
        args.extract_mode = 'thres'
        args.diff_mode = 'fullness'
        args.opti_mode = 'thres'
        args.rate = 0.1
        args.direction = DIR_B
        args.bw = False
        args.thres = 0.2
    elif args.scene == 'movie':
        args.extract_mode = 'thres'
        args.diff_mode = 'phash'
        args.opti_mode = 'quant'
        args.rate = 0.2
        args.direction = DIR_F
        args.bw = False
        args.thres = 0.9

        
def config_thres(args):
    if not math.isnan(args.thres):
        return
    if args.extract_mode == 'relthres':
        args.thres = 0.6
    elif args.extract_mode == 'thres':
        args.thres = 0.1
    elif args.extract_mode == 'normthres':
        args.thres = 0.1
    elif args.extract_mode == 'adathres':
        args.thres = 0.1
    