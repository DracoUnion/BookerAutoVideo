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
from .clip import *
import torchvision as tv
from concurrent.futures import ThreadPoolExecutor

DIR_F = 'forward'
DIR_B = 'backward'
DIR_T = 'twoway'

ocr_reader = None
PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

def preproc_imgs(imgs):
    # bytes -> ndarray
    if isinstance(imgs[0], bytes):
        imgs = [
            cv2.imdecode(np.frombuffer(i, np.uint8), cv2.IMREAD_COLOR)
            for i in imgs
        ]
    # resize -> 224x224
    imgs = [
        cv2.resize(i, [224, 224], interpolation=cv2.INTER_CUBIC) 
        for i in imgs
    ]
    imgs = (
        np.asarray(imgs) 
            # HWC -> CHW
            .transpose([0, 3, 1, 2])
            # BGR -> RGB
            [:, ::-1]
            # norm
            .__truediv__(255)
    )
    return imgs

def load_ppt_ext_model(model_path=None, freeze_nonlast=True):
    model = tv.models.resnet18(num_classes=1)
    if model_path:
        stdc = torch.load(model_path)
        if 'fc.weight' in stdc and stdc['fc.weight'].shape != torch.Size([1, 512]):
            del stdc['fc.weight']
        if 'fc.bias' in stdc and stdc['fc.bias'].shape != torch.Size([1]):
            del stdc['fc.bias']
        model.load_state_dict(stdc, False)
    model = model.half()
    if torch.cuda.is_available():
        model = model.cuda()
    if freeze_nonlast:
        for name, param in model.named_parameters():
            if not name.startswith('fc.'):
                param.requires_grad = False
            
    return model

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

def filter_repeat(frames, args):
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
            if f['diff'] >= args.diff_thres
        ]
        if len(frames) == nframe: break
    return frames

def filter_by_ppt_model(frames, args):
    model = load_ppt_ext_model(args.model_path).eval()
    imgs = preproc_imgs([f['img'] for f in frames])

    probs = []
    for i in range(0, len(imgs), args.batch_size):
        imgs_batch = imgs[i:i+args.batch_size]
        imgs_batch = torch.tensor(imgs_batch).half()
        if torch.cuda.is_available():
            imgs_batch = imgs_batch.cuda()
        probs_batch = torch.sigmoid(model.forward(imgs_batch)).flatten()
        probs += probs_batch.tolist()

    is_ppt = np.greater_equal(probs, args.ppt_thres)
    for f, p, l in zip(frames, probs, is_ppt):
        print(f"{f['time']}: {p}, {l}")
    return [f for f, l in zip(frames, is_ppt) if l]

def filter_by_ocr(frames, args):
    if args.ocr <= 0: return frames
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
    return frames
   
def check_cut_args(args):
    assert (
        args.left >= 0 and
        args.right >= 0 and
        args.top >= 0 and
        args.bottom >= 0
    )

    assert (
        args.left + args.right <= 1 and
        args.bottom + args.top <= 1
    )

def cut_img(img, args):
    h, w, *_ = img.shape
    left = int(args.left * w)
    right = -int(args.right * w)
    if right == 0:
        right = None
    top = int(args.top * h)
    bottom = -int(args.bottom * h)
    if bottom == 0:
        bottom = None
    return img[top:bottom, left:right]

def tr_calc_met(f):
    img = f['img']
    f.update({
     'color': colorfulness(img),
      'hog': hog_entro(img),
    })

def extract_keyframe(args):
    print(args)
    check_cut_args(args)
    fname = args.fname
    # 从视频中读取帧
    imgs, _ = get_video_imgs(fname, args.rate)
    frames = [
        {
            'idx': i,
            'time': i / args.rate,
            'img': cut_img(img, args),
            # 'color': colorfulness(img),
            # 'hog': hog_entro(img),
        } 
        for i, img in enumerate(imgs)
    ]
    pool = ThreadPoolExecutor(args.threads)
    hdls = []
    for f in frames:
        h = pool.submit(tr_calc_met, f)
        hdls.append(h)
    for h in hdls: h.result()
    for f in frames:
        tm = f['time']
        img = f['img']
        color = f['color']
        hog = f['hog']
        print(f'time: {tm}, color: {color}, hog: {hog}')
    frames = [
        f for f in frames 
        if f['color'] < 0.4 and f['hog'] < 0.5
    ]
    # 第一个过滤：根据帧间差过滤重复幻灯片
    frames = filter_repeat(frames, args)
    # 第二个过滤：过滤人像和景物
    if args.model_path:
        frames = filter_by_ppt_model(frames, args)
        if frames:
            frames = filter_repeat(frames, args)
    # 第三个过滤：OCR 之后根据文字过滤语义相似幻灯片
    # frames = filter_by_ocr(frames, args)
    # 优化图像
    for f in frames:
        if 'text' in f: del f['text']
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

    