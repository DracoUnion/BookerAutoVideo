import argparse
import sys
import warnings
from . import __version__
from .audio2txt import *
from .keyframe import *

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(prog="BookerAutoVideo", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version=f"PYBP version: {__version__}")
    parser.set_defaults(func=lambda x: parser.print_help())
    subparsers = parser.add_subparsers()
    
    audio2txt_parser = subparsers.add_parser("audio2txt", help="convert audio to text")
    audio2txt_parser.add_argument("fname", help="file name")
    audio2txt_parser.add_argument("-t", "--threads", type=int, default=8, help="num of threads")
    audio2txt_parser.add_argument("-m", "--model", default='base', choices=['tiny', 'base', 'small', 'medium', 'large'], help="model name")
    audio2txt_parser.set_defaults(func=audio2txt_handle)

    kf_parser = subparsers.add_parser("ext-kf", help="extract keyframes")
    kf_parser.add_argument("fname", help="file name")
    kf_parser.add_argument("-e", "--extract-mode", default="relmax", help="extract mode")
    kf_parser.add_argument("-d", "--diff-mode", choices=list(img_sim.keys()),default="pixel_l1", help="frame diff mode")
    kf_parser.add_argument("-o", "--opti-mode", default="none", help="img opti mode")
    kf_parser.add_argument("-r", "--rate", type=float, default=1, help="how many frames to extract in 1s")
    kf_parser.add_argument("-s", "--smooth", action='store_true', help="whether to smooth frames")
    kf_parser.add_argument("-D", "--direction", choices=[DIR_F, DIR_B, DIR_T], default=DIR_F, help="the direction used to calc frame diff")
    kf_parser.add_argument("--bw", action='store_true', help="convert img into bw instead of greyscale when calculating diff")
    kf_parser.add_argument("--smooth-win-size", type=int, default=20, help="window size for smooth")
    kf_parser.add_argument("-n", "--top-num", type=int, default=20, help="num in top mode")
    kf_parser.add_argument("-t", "--thres", type=float, default=0.6, help="thres in thres mode")
    kf_parser.add_argument("--relmax-win-size", type=int, default=3, help="window size for relmax")
    kf_parser.set_defaults(func=extract_keyframe)

    args = parser.parse_args()
    args.func(args)
    
if __name__ == '__main__': main()