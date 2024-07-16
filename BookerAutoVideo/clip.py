from transformers.models.chinese_clip import ChineseCLIPModel, ChineseCLIPProcessor
import torch
from typing import List
from os import path
import os
from .util import *
from io import BytesIO
from PIL import Image

def load_clip(model_path, device):
    return (
        ChineseCLIPModel.from_pretrained(model_path).to(device=device),
        ChineseCLIPProcessor.from_pretrained(model_path),   
    )

# 填充批量使每个句子都等长
def pad_ids(ids_batch: List[List[int]], pad_id=0, max_len=None, pad_left=False):
    max_len = max_len or max([len(ids) for ids in ids_batch])
    for i, ids in enumerate(ids_batch):
        if len(ids) < max_len:
            pad_len = max_len - len(ids)
            ids_batch[i] = (
                [pad_id] * pad_len + ids
                if pad_left else
                ids + [pad_id] * pad_len
            )

def clip_test(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, proc = load_clip(args.model_path, device)
    cates = args.cates.split(',')
    if len(cates) == 0:
        raise ValueError('请提供类别')
    cids = proc(text=cates).input_ids
    pad_ids(cids, proc.tokenizer.pad_token_id)
    cids = torch.tensor(cids, dtype=int, device=model.device)
    img_fnames = (
        [args.img] 
        if path.isfile(args.img) 
        else [
            path.join(args.img, f)
            for f in os.listdir(args.img)
        ]
    )
    img_fnames = [f for f in img_fnames if is_pic(f)]
    if len(img_fnames) == 0:
        raise ValueError('请提供图片路径')
    imgs = [
        Image.open(BytesIO(open(f, 'rb').read()))
        for f in img_fnames
    ]
    imgs_norm = proc(images=imgs, return_tensors='np').pixel_values
    probs = []
    for i in range(0, len(imgs_norm), args.batch_size):
        imgs_batch = imgs_norm[i:i+args.batch_size]
        imgs_batch = torch.tensor(imgs_batch, device=model.device)
        logits_batch = model.forward(input_ids=cids, pixel_values=imgs_batch).logits_per_image
        probs_batch = torch.softmax(logits_batch, -1)
        probs += probs_batch.tolist()
    best_labels = np.argmax(probs, -1)
    best_labels = [cates[l] for l in best_labels]
    for f, ps, l in zip(img_fnames, probs, best_labels):
        print(f'{f}: {l}')
        prob_msg = ', '.join([f'{c}: {p:.3f}' for c, p in zip(cates, ps)])
        print(prob_msg)
        
    

