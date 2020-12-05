import builtins
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import dataset
import model
from tools import get_logger

parser = argparse.ArgumentParser(description='Train character and word level LSTM language models')
parser.add_argument('--data', type=str, default='./data/ptb',
                    help='location of the data corpus')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=25,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')

parser.add_argument('--word', action='store_true',
                    help='use word level model')
parser.add_argument('--arch_large', action='store_true',
                    help='use large architecture')

parser.add_argument('--save', type=str, default='output/',
                    help='where to save checkpoints logs')
parser.add_argument('--debug', action='store_true',
                    help='whether in debug mode or not')

args = parser.parse_args()

os.makedirs(args.save, exist_ok=True)
if not args.debug:
    os.environ['PYTHONBREAKPOINT'] = '0'
    logger = get_logger(
        logpath=os.path.join(args.save, 'logs'),
        filepath=os.path.abspath(__file__)
    )
    def print_pass(*args):
        logger.info(*args)
    builtins.print = print_pass

print(args)

torch.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = dataset.Corpus(args.data, args.batch_size, args.bptt)
train_data = corpus.all_tensors['train']
val_data = corpus.all_tensors['valid']
test_data = corpus.all_tensors['test']

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.word_dict)
n_chars = len(corpus.char_dict)

if args.word:
    model = model.WordLSTM(args, ntokens).cuda()
    def get_batch(inp, i):
        word_targets = inp['targets']
        seq_len = min(args.bptt, len(word_targets) - 1 - i)
        data = word_targets[i:i+seq_len]
        target = word_targets[i+1:i+1+seq_len].view(-1)
        return data, target
else:
    model = model.CharLSTM(args, n_chars, ntokens).cuda()
    def get_batch(inp, i):
        word_targets = inp['targets']
        seq_len = min(args.bptt, len(word_targets) - 1 - i)
        data = inp['characters'][i:i+seq_len]
        target = word_targets[i+1:i+1+seq_len].view(-1)
        return data, target

print(model)

criterion = nn.NLLLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate(data_source):
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.word_dict)
    nseq = data_source['targets'].shape[0]
    hidden = model.init_hidden(args.batch_size)
    with torch.no_grad():
        for i in range(0, nseq - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (nseq - 1)


def train(data_source):
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.word_dict)
    nseq = data_source['targets'].shape[0]
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, nseq - 1, args.bptt)):
        data, targets = get_batch(data_source, i)
        model.zero_grad()
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, nseq // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# Loop over epochs.
lr = args.lr
best_val_loss = None
prev_ppl = float('+inf')

print('=' * 89)
params = sum(p.numel() for p in model.parameters())//1000000
print('| Begin training | parameters {}'.format(params))
print('=' * 89)

for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    train(train_data)
    val_loss = evaluate(val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                       val_loss, math.exp(val_loss)))
    print('-' * 89)
    # Save the model if the validation loss is the best we've seen so far.
    if not best_val_loss or val_loss < best_val_loss:
        with open(os.path.join(args.save, 'model.pth'), 'wb') as f:
            torch.save(model, f)
        best_val_loss = val_loss
    cur_ppl = math.exp(val_loss)
    if prev_ppl - cur_ppl < 1:
        lr /= 2
    prev_ppl = cur_ppl

# Load the best saved model.
with open(os.path.join(args.save, 'model.pth'), 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    model.lstm.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

