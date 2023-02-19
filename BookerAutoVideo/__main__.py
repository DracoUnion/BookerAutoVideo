import argparse
import sys
import warnings
from . import __version__
from .video2txt import *
from .keyframe import *
from .imgsim import *

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(prog="BookerAutoVideo", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version=f"PYBP version: {__version__}")
    parser.set_defaults(func=lambda x: parser.print_help())
    subparsers = parser.add_subparsers()
    
    video2txt_parser = subparsers.add_parser("video2txt", help="convert audio to text")
    video2txt_parser.add_argument("fname", help="file name")
    video2txt_parser.add_argument("-T", "--threads", type=int, default=8, help="num of threads")
    video2txt_parser.add_argument("-a", "--asr-model", default='base', choices=['tiny', 'base', 'small', 'medium', 'large'], help="model name")
    video2txt_parser.add_argument("-I", "--no-image", action='store_true', help="whether to not extract image")
    video2txt_parser.add_argument("-l", "--language", default='Chinese',  help="language")
    video2txt_parser.add_argument("-e", "--extract-mode", default="relmax", help="extract mode")
    video2txt_parser.add_argument("-d", "--diff-mode", choices=list(img_sim.keys()),default="pixel_l1", help="frame diff mode")
    video2txt_parser.add_argument("-o", "--opti-mode", default="none", help="img opti mode, default 'none'")
    video2txt_parser.add_argument("-r", "--rate", type=float, default=1, help="how many frames to extract in 1s")
    video2txt_parser.add_argument("-S", "--smooth", action='store_true', help="whether to smooth frames")
    video2txt_parser.add_argument("-D", "--direction", choices=[DIR_F, DIR_B, DIR_T], default=DIR_F, help="the direction used to calc frame diff")
    video2txt_parser.add_argument("--bw", action='store_true', help="convert img into bw instead of greyscale when calculating diff")
    video2txt_parser.add_argument("--smooth-win-size", type=int, default=20, help="window size for smooth")
    video2txt_parser.add_argument("-n", "--top-num", type=int, default=20, help="num in top mode")
    video2txt_parser.add_argument("-t", "--thres", type=float, default=0.6, help="thres in thres mode")
    video2txt_parser.add_argument("--relmax-win-size", type=int, default=3, help="window size for relmax")
    video2txt_parser.add_argument("-s", "--scene", choices=['auto', 'ppt'], default='auto', help="scene")
    video2txt_parser.set_defaults(func=video2txt_handle)

    kf_parser = subparsers.add_parser("ext-kf", help="extract keyframes")
    kf_parser.add_argument("fname", help="file name")
    kf_parser.add_argument("-e", "--extract-mode", default="relmax", help="extract mode")
    kf_parser.add_argument("-d", "--diff-mode", choices=list(img_sim.keys()),default="pixel_l1", help="frame diff mode")
    kf_parser.add_argument("-o", "--opti-mode", default="none", help="img opti mode, default 'none'")
    kf_parser.add_argument("-r", "--rate", type=float, default=1, help="how many frames to extract in 1s")
    kf_parser.add_argument("-N", "--norm", action='store_true', help="whether to apply normalization to frame diff")
    kf_parser.add_argument("-D", "--direction", choices=[DIR_F, DIR_B, DIR_T], default=DIR_F, help="the direction used to calc frame diff")
    kf_parser.add_argument("--bw", action='store_true', help="convert img into bw instead of greyscale when calculating diff")
    kf_parser.add_argument("-n", "--top-num", type=int, default=20, help="num in top mode")
    kf_parser.add_argument("-t", "--thres", type=float, default=0.6, help="thres in thres mode")
    kf_parser.add_argument("--relmax-win-size", type=int, default=3, help="window size for relmax")
    kf_parser.add_argument("-s", "--scene", choices=['auto', 'ppt'], default='auto', help="scene")
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