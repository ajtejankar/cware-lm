import os
import sys
import time
import random
import argparse
import builtins

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from tools import get_logger


def parse_option():
    parser = argparse.ArgumentParser('Train Char-LM')

    parser.add_argument('--checkpoint_path', default='output/', type=str,
                        help='where to save checkpoints logs')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')

    args = parser.parse_args()

    return args


def main():

    args = parse_option()
    os.makedirs(args.checkpoint_path, exist_ok=True)

    if not args.debug:
        os.environ['PYTHONBREAKPOINT'] = '0'
        logger = get_logger(
            logpath=os.path.join(args.checkpoint_path, 'logs'),
            filepath=os.path.abspath(__file__)
        )
        def print_pass(*args):
            logger.info(*args)
        builtins.print = print_pass

    print(args)


if __name__ == '__main__':
    main()

