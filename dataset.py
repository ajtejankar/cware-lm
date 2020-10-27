import os
import glob


def parse_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    sentences = []
    cur_sentence = []
    for line in lines:
        line = line.strip()

        if line == '======================================':
            if len(cur_sentence):
                sentences.append(cur_sentence)
            cur_sentence = []
        elif len(line):
            for token in line.split(' '):
                if token == '[' or token == ']':
                    continue
                else:
                    cur_sentence.append(token.split('/')[0])
    if len(cur_sentence):
        sentences.append(cur_sentence)

    return sentences


def get_all_sentences(pat):
    all_sentences = []
    for file_path in glob.glob(pat):
        sentences = parse_file(file_path)
        all_sentences.extend(sentences)
    return all_sentences


def print_stats(all_sentences):
    tokens = list(w for s in all_sentences for w in s)
    chars = ''.join(tokens)
    chars = set(chars)
    types = set(tokens)

    print('chars: {}'.format(len(chars)))
    print('types: {}'.format(len(types)))
    print('tokens: {}'.format(len(tokens)))
