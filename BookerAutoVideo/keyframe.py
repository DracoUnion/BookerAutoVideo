import av
import av.datasets
import os
import shutil
import math
import sys

def ext_keyframe(args):
    fname = args.file
    videos_save_path = args.save_path
    if not os.path.exists(videos_save_path):
        os.makedirs(videos_save_path)
    print(fname)
    
    img_path = os.path.join(videos_save_path, os.path.basename(fname) + '_img')
    print(img_path)
    if os.path.isdir(img_path):
        shutil.rmtree(img_path)
    os.makedirs(img_path)
    container = av.open(fname)

    stream = container.streams.video[0]
    stream.codec_context.skip_frame = 'NONKEY'

    for frame in container.decode(stream):
        frame.to_image().save(
            os.path.join(img_path, f'{frame.pts:04d}.jpg'), 
            quality=80,
        )

