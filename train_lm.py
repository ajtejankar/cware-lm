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

    # arch options
    parser.add_argument('--arch-large', action='store_true',
                        help='use the large model architecture')

    # optimization options
    parser.add_argument('--lr', type=float, default=1.0,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of training epochs')

    # general experiment options
    parser.add_argument('--checkpoint_path', default='output/', type=str,
                        help='where to save checkpoints logs')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')

    args = parser.parse_args()

    return args


class CharCNN(nn.Module):
    def __init__(self, d, w, h):
        super(CharCNN, self).__init__()
        self.conv = nn.ModuleList([
            nn.Conv1d(d, h_i, w_i) for w_i, h_i in zip(w, h)
        ])
        self.tanh = nn.Tanh()

    def forward(self, inp):
        feats = []
        for conv in self.conv:
            feats.append(self.tanh(conv(inp)))
        return torch.cat(feats, dim=-1)


class HighwayLayer(nn.Module):
    def __init__(self, d):
        super(HighwayLayer, self).__init__()
        self.linear_t = nn.Linear(d, d)
        self.linear_h = nn.Linear(d, d)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # todo. initialize bT to a negative value

    def forward(self, inp):
        # t = sigmoid(Wt * inp + bt)
        t = self.sigmoid(self.linear_t(inp))
        # t .O relu(Wh * inp + bh) + (1 - t) .O y
        out = t * self.relu(self.linear_h(inp)) + (1 - t) * inp
        return out


class Highway(nn.Module):
    def __init__(self, l, d):
        super(Highway, self).__init__()
        self.layers = nn.ModuleList([Highway(d) for _ in range(l)])

    def forward(self, inp):
        for layer in self.layers:
            inp = layer(inp)
        return inp


class CharLSTM(nn.Module):
    def __init__(self, args):
        super(CharLSTM, self).__init__()

        # architecture configurations
        self.char_d = 15
        self.lstm_l = 2
        if args.arch_large:
            self.char_w = list(range(7))
            self.char_h = [min(200, w * 50) for w in self.char_w]
            self.highway_l = 2
            self.lstm_m = 650
        else:
            self.char_w = list(range(6))
            self.char_h = [w * 25 for w in self.char_w]
            self.highway_l = 1
            self.lstm_m = 300

        # todo: add char-cnn model
        self.char_cnn = CharCNN(d, self.char_w, self.char_h)

        # todo: add highway network model
        self.highway = Highway(d, self.highway_l)

        inp_d = sum(self.char_h)
        self.lstm = nn.LSTM(inp_d, self.lstm_m, self.lstm_l)


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

    model = CharLSTM(args)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    print(args)


if __name__ == '__main__':
    main()

