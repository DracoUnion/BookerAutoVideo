import cv2
import numpy as np
import argparse
import os
import re
import math
from os import path
from scipy import signal
from imgyaso import adathres
from .imgsim import img_sim
from .util import *

DIR_F = 'forward'
DIR_B = 'backward'
DIR_T = 'twoway'

ext_modes = ['topn', 'normthres', 'relthres', 'adathres', 'relmax']

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
    idx = 0
    tm = 0
    while(True):
        cap.set(cv2.CAP_PROP_POS_MSEC, tm * 1000)
        succ, img = cap.read()
        if not succ: break
        print(f'time {nsec2hms(tm)} loaded')
        frames.append({
            'idx': idx, 
            'time': tm,
            'img': img,
            'grey': cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        })
        if bw: frames[-1]['grey'] = adathres(frames[-1]['grey'])
        idx += 1
        tm += 1 / rate
    cap.release()
    return frames

'''
def dedup(frames, rate):
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=int(rate * 15), 
        varThreshold=16,
        detectShadows=False,
    )
    captured = False
    res = []
    for  f in frames:
        mask = fgbg.apply(f['img']) # apply the background subtractor
        # apply a series of erosions and dilations to eliminate noise
        # eroded_mask = cv2.erode(mask, None, iterations=2)
        # mask = cv2.dilate(mask, None, iterations=2)
        # if the width and height are empty, grab the spatial dimensions
        h, w = mask.shape[:2]
        # compute the percentage of the mask that is "foreground"
        p_diff = cv2.countNonZero(mask) / float(w * h) * 100
        # if p_diff less than N% then motion has stopped, thus capture the frame
        if p_diff < 0.1 and not captured:
            captured = True
            res.append(f)
        # otherwise, either the scene is changing or we're still in warmup
        # mode so let's wait until the scene has settled or we're finished
        # building the background model
        elif captured and p_diff >= 3:
            captured = False
    return res
'''

def calc_frame_diffs(frames, args):
    direction, diff_mode = args.direction, args.diff_mode  
    frames[0]['diff'] = 1
    for prev, curr in zip(frames[:-1], frames[1:]):
        curr['diff'] = frame_diff(
            prev['grey'], curr['grey'], diff_mode,
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
    if ext_mode == 'topn':
        frames.sort(key=lambda f: f['diff'], reverse=True)
        frames = frames[:args.top_num]
    elif ext_mode in ['normthres', 'relthres', 'adathres']:
        frames = [
            f for f in frames
            if f['diff'] >= args.thres
        ]
    elif ext_mode == 'relmax':
        if args.win_size % 2 == 0:
            args.win_size += 1
        odr = (args.win_size - 1) // 2
        diffs = np.array([f['diff'] for f in frames])
        idcs = np.asarray(signal.argrelmax(diffs, order=odr))[0]
        frames = [frames[i] for i in idcs]
    else:
        raise valueError('提取模式未定义！')
    # 优化图像
    for f in frames:
        img = cv2.imencode(
            '.png', f['img'], 
            [cv2.IMWRITE_PNG_COMPRESSION, 9]
        )[1]
        f['img'] = opti_img(bytes(img), args.opti_mode, 8)
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
        args.extract_mode = 'normthres'
        args.diff_mode = 'pixel_l1'
        args.opti_mode = 'quant'
        args.rate = 0.2
        args.direction = 'backward'
        args.bw = False
        args.thres = 0.1
        
def config_thres(args):
    if not math.isnan(args.thres):
        return
    if args.extract_mode == 'relthres':
        args.thres = 0.6
    elif args.extract_mode == 'normthres':
        args.thres = 0.1
    elif args.extract_mode == 'adathres':
        args.thres = 0.1
    