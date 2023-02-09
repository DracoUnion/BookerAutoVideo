import cv2
import numpy as np

# 直方图余弦
def hist_cos_sim(prev, next):
    assert prev.ndim == 2 and next.ndim == 2
    prev_hist = np.histogram(prev, bins=list(range(257)))[0]
    next_hist = np.histogram(next, bins=list(range(257)))[0]
    prev_hist = prev_hist / prev_hist.sum()
    next_hist = next_hist / next_hist.sum()
    prev_hist_mod = np.sum(prev_hist ** 2) ** 0.5
    next_hist_mod = np.sum(next_hist ** 2) ** 0.5
    return np.sum(prev_hist * next_hist) / prev_hist_mod / next_hist_mod

# 直方图 L1
def hist_l1_sim(prev, next):
    assert prev.ndim == 2 and next.ndim == 2
    prev_hist = np.histogram(prev, bins=list(range(257)))[0]
    next_hist = np.histogram(next, bins=list(range(257)))[0]
    max = np.fmax(prev_hist, next_hist)
    diff = np.mean(np.where(max == 0, 0, np.abs(prev_hist - next_hist) / max))
    return 1 - diff

# 均值哈希
def ahash(img, size=12):
    assert img.ndim == 2
    resize = cv2.resize(img, (size, size))
    mean = resize.mean()
    # 灰度大于平均值为 1 相反为 0
    hash = (resize >= mean).astype(int).ravel()
    return hash
    
# 差值哈希
def dhash(img, size=12):
    assert img.ndim == 2
    resize = cv2.resize(img, (size, size))
    before = resize[:, :-1]
    after = resize[:, 1:]
    # 每行前一个像素大于后一个像素为 1，相反为 0
    hash = (before > after).astype(int).ravel()
    return hash

# 感知哈希
def phash(img, size=32):
    assert img.ndim == 2
    resize = cv2.resize(img, (size, size))
    dct = cv2.dct(np.float32(resize))
    roi = dct[0:8, 0:8]
    mean = roi.mean()
    hash = (roi > mean).astype(int).ravel()
    return hash

def ahash_sim(prev, next, size=12):
    return np.mean(ahash(prev) == ahash(next))

def dhash_sim(prev, next, size=12):
    return np.mean(dhash(prev) == dhash(next))

def phash_sim(prev, next, size=32):
    return np.mean(phash(prev) == phash(next))

def pixel_cos_sim(prev, next):
    prev_norm = prev / prev.mean()
    next_norm = next / next.mean()
    prev_norm_mod = np.sum(prev_norm ** 2) ** 0.5
    next_norm_mod = np.sum(next_norm ** 2) ** 0.5
    return np.sum(prev_norm * next_norm) / prev_norm_mod / next_norm_mod

def pixel_l1_sim(prev, next):
    max = np.fmax(next, prev)
    diff = np.mean(np.where(max != 0, cv2.absdiff(next, prev) / max, 0))
    return 1 - diff

img_sim = {
    'pixel_l1':  pixel_l1_sim,
    'pixel_cos': pixel_cos_sim,
    'ahash': ahash_sim,
    'dhash': dhash_sim,
    'phash': phash_sim,
    'hist_l1':   hist_l1_sim,
    'hist_cos':  hist_cos_sim,
}

def img_sim_handle(args):
    img1_fname, img2_fname = args.img1, args.img2
    mode = args.mode
    img1 = cv2.imdecode(np.fromfile(img1_fname, np.uint8), IMREAD_GRAYSCALE)
    img2 = cv2.imdecode(np.fromfile(img2_fname, np.uint8), IMREAD_GRAYSCALE)
    sim = img_sim[mode](img1, img2)
    print(f'mode: {mode}, sim: {sim}')
    
    