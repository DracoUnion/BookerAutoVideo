import argparse
import sys
import warnings
from . import __version__
from . import *

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
    kf_parser.add_argument("file", help="file")
    kf_parser.add_argument("-o", "--save-path", default='out', help="path to save")
    kf_parser.set_defaults(func=ext_keyframe)
        
    args = parser.parse_args()
    args.func(args)
    
if __name__ == '__main__': main()