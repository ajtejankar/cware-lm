import math
import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.key2idx = {}
        self.idx2key = []

    def add_key(self, key):
        if key not in self.key2idx:
            self.idx2key.append(key)
            self.key2idx[key] = len(self.idx2key) - 1
        return self.key2idx[key]

    def __len__(self):
        return len(self.idx2key)


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = torch.split(data, nbatch, dim=0)
    data = torch.stack(data, dim=1)
    return data.cuda()


class Corpus(object):
    def __init__(self, data_dir, batch_size, bptt):
        self.word_dict = Dictionary()
        self.char_dict = Dictionary()
        self.batch_size = batch_size
        self.bptt = bptt
        self.file_paths = {
            'train': os.path.join(data_dir, 'train.txt'),
            'valid': os.path.join(data_dir, 'valid.txt'),
            'test': os.path.join(data_dir, 'test.txt'),
        }

        max_w_len = 0
        all_words = {'train': [], 'valid': [], 'test': []}
        self.char_dict.add_key(' ') # zero padding char
        self.char_dict.add_key('{') # start of word char
        self.char_dict.add_key('}') # end of word char
        self.char_dict.add_key('|') # unk word char
        self.char_dict.add_key('+') # eos word char

        for split, file_path in self.file_paths.items():
            assert os.path.exists(file_path)
            with open(file_path, 'r', encoding="utf8") as f:
                lines = f.readlines()
                lines = [line.strip().split(' ') + ['<eos>'] for line in lines]
                for line in lines:
                    for word in line:
                        #### word pre-processing ####
                        if word == '<unk>':
                            word = '|' # replace <unk> with special char
                        elif word == '<eos>':
                            word = '+' # replace <eos> with special char
                        self.word_dict.add_key(word)
                        all_words[split].append(word)
                        max_w_len = max(max_w_len, len(word))
                        for char in word:
                            self.char_dict.add_key(char)

        self.all_tensors = {}
        for split, words in all_words.items():
            w_inds = torch.tensor([self.word_dict.key2idx[w] for w in words])
            c_inds = []
            for i, word in enumerate(words):
                #### character pre-processing ####
                word = '{' + word + '}'
                word = word + ' '*(max_w_len+2-len(word))
                c_inds.append(torch.tensor([self.char_dict.key2idx[c] for c in word]))
            c_inds = torch.stack(c_inds, dim=0)

            targets = batchify(w_inds, batch_size).long()
            characters = batchify(c_inds, batch_size).long()

            self.all_tensors[split] = {
                'targets': targets,
                'characters': characters,
            }

