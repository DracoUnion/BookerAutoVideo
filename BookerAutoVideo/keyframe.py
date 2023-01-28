import cv2
import numpy as np
import argparse
import os
import re
from os import path
from scipy import signal
from imgyaso import adathres
from .imgsim import img_sim
from .util import is_video

DIR_F = 'forward'
DIR_B = 'backward'
DIR_T = 'twoway'

def frame_diff(prev, next, mode):
    if mode in img_sim:
        return 1 - img_sim[mode](prev, next)
    else:
        raise ValueError('mode not found')

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
    
def calc_frame_diffs(frames, direction, diff_mode):
    frames[0]['diff'] = 0
    for i in range(1, len(frames)):
        frames[i]['diff'] = frame_diff(
            frames[i - 1]['grey'],
            frames[i]['grey'],
            diff_mode,
        )
    if direction == DIR_B:
        for i in range(0, len(frames) - 1):
            frames[i]['diff'] = frames[i + 1]['diff']
        frames[-1]['diff'] = 0
    elif direction == DIR_T:
        frames[0]['diff'] = frames[1]['diff']
        for i in range(1, len(frames) - 1):
            frames[i]['diff'] += frames[i + 1]['diff']
            frames[i]['diff'] /= 2
    
def extract_keyframe(args):
    fname = args.fname
    mode = args.mode
    opti_mode = args.opti_mode
    if not is_video(fname):
        print('请提供视频')
        return
    print(fname)
    # 从视频中读取帧
    frames = load_frames(fname, args.rate, args.bw)
    nframes = max([f['idx'] for f in frames]) + 1
    # 平滑
    if args.smooth:
        greies = [f['grey'] for f in frames]
        greies = smooth(greies, args.smooth_win_size)
        for f, g in zip(frames, greies): f['grey'] = g
    # 计算差分
    calc_frame_diffs(frames, args.direction, args.diff_mode)
    for f in frames:
        print(f"time {nsec2hms(f['time'])} diff: {f['diff']}")
    # 计算关键帧
    if mode == 'topn':
        frames.sort(key=lambda f: f['diff'], reverse=True)
        frames = frames[:args.top_num]
    elif mode == 'thres':
        max_diff = max([f['diff'] for f in frames])
        frames = [
            f for f in frames
            if f['diff'] / max_diff >= args.thres
        ]
    elif mode == 'relmax':
        if args.relmax_win_size % 2 == 0:
            args.relmax_win_size += 1
        odr = (args.relmax_win_size - 1) // 2
        diffs = np.array([f['diff'] for f in frames])
        idcs = np.asarray(signal.argrelmax(diffs, order=odr))[0]
        frames = [frames[i] for i in idcs]
    # 去重复
    # frames = dedup(frames, args.rate)
    # 保存所有关键帧
    l = len(str(nframes))
    opath = re.sub(r'\.\w+$', '', fname) + '_keyframe'
    if not path.isdir(opath): os.mkdir(opath)
    for f in frames:
        ofname = path.join(opath, f'keyframe_{nsec2hms(f["time"])}.png')
        print(ofname)
        data = cv2.imencode(
            '.png', f['img'], 
            [cv2.IMWRITE_PNG_COMPRESSION, 9]
        )[1]
        open(ofname, 'wb').write(bytes(data))
        
