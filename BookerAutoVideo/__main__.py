import argparse
import sys
import math
import warnings
from . import __version__
from .autovideo import *
from .video2txt import *
from .keyframe import *
from .imgsim import *

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(prog="BookerAutoVideo", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version=f"PYBP version: {__version__}")
    parser.set_defaults(func=lambda x: parser.print_help())
    subparsers = parser.add_subparsers()

    autovid_parser = subparsers.add_parser("auto-video", help="generate video")
    autovid_parser.add_argument("config", help="config file name")
    autovid_parser.set_defaults(func=autovideo)

    video2txt_parser = subparsers.add_parser("video2txt", help="convert audio to text")
    video2txt_parser.add_argument("fname", help="file name")
    video2txt_parser.add_argument("-t", "--threads", type=int, default=8, help="num of threads")
    video2txt_parser.add_argument("-I", "--no-image", action='store_true', help="whether to not catch screenshots")
    video2txt_parser.add_argument("-m", "--model", default='ggml-large-v2-q5_1', help="model name or path")
    video2txt_parser.add_argument("-l", "--lang", default='zh',  help="language")
    video2txt_parser.add_argument("-s", "--scene", choices=['auto', 'ppt', 'ppt2', 'movie'], default='auto', help="scene")
    video2txt_parser.set_defaults(
        opti_mode='none',
        rate=0.2,
        direction=DIR_T,
        thres=0.6,
        func=video2txt_handle,
    )

    kf_parser = subparsers.add_parser("ext-kf", help="extract keyframes")
    kf_parser.add_argument("fname", help="file name")
    kf_parser.add_argument("-o", "--opti-mode", default="none", help="img opti mode, default 'none'")
    kf_parser.add_argument("-r", "--rate", type=float, default=0.2, help="how many frames to extract in 1s")
    kf_parser.add_argument("-d", "--direction", choices=[DIR_F, DIR_B, DIR_T], default=DIR_T, help="the direction used to calc frame diff")
    kf_parser.add_argument("-t", "--thres", type=float, default=0.6, help="thres in thres mode")
    kf_parser.add_argument("-T", "--threads", type=int, default=8, help="thread count")
    kf_parser.set_defaults(func=extract_keyframe_file)

    sim_parser = subparsers.add_parser("img-sim", help="calc sim of 2 imgs")
    sim_parser.add_argument("img1", help="img1 fname")
    sim_parser.add_argument("img2", help="img1 fname")
    sim_parser.set_defaults(func=img_sim_handle)
    
    sim_dir_parser = subparsers.add_parser("img-dir-sim", help="calc sim of imgs in dir")
    sim_dir_parser.add_argument("dir", help="dir name")
    sim_dir_parser.set_defaults(func=img_sim_dir_handle)

    args = parser.parse_args()
    args.func(args)
    
if __name__ == '__main__': main()