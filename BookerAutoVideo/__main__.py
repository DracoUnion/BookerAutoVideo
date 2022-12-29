import argparse
import sys
from . import __version__
from . import *

def main():
    parser = argparse.ArgumentParser(prog="BookerAutoVideo", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version=f"PYBP version: {__version__}")
    parser.set_defaults(func=lambda x: parser.print_help())
    subparsers = parser.add_subparsers()
    
    gh_book_parser = subparsers.add_parser("audio2txt", help="convert audio to text")
    gh_book_parser.add_argument("fname", help="file name")
    gh_book_parser.add_argument("-t", "--threads", type=int, default=8, help="num of threads")
    gh_book_parser.add_argument("-m", "--model", default='base', choices=['tiny', 'base', 'small', 'medium', 'large'], help="model name")
    gh_book_parser.set_defaults(func=audio2txt_handle)
        
    args = parser.parse_args()
    args.func(args)
    
if __name__ == '__main__': main()