import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):
    def __init__(self, d, w, h):
        super(CharCNN, self).__init__()
        self.conv = nn.ModuleList([
            nn.Conv1d(d, h_i, w_i, 1, w_i-1) for w_i, h_i in zip(w, h)
        ])
        self.tanh = nn.Tanh()

    def forward(self, inp):
        feats = []
        for conv in self.conv:
            f = self.tanh(conv(inp))
            f = torch.max(f, dim=-1).values
            feats.append(f)
        return torch.cat(feats, dim=-1)


class HighwayLayer(nn.Module):
    def __init__(self, d):
        super(HighwayLayer, self).__init__()
        self.linear_t = nn.Linear(d, d)
        self.linear_h = nn.Linear(d, d)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        # nn.init.constant_(self.linear_t.bias, -2)

    def forward(self, inp):
        # t = sigmoid(Wt * inp + bt)
        t = self.sigmoid(self.linear_t(inp) - 2)
        # t .O relu(Wh * inp + bh) + (1 - t) .O y
        out = t * self.relu(self.linear_h(inp)) + (1 - t) * inp
        return out


class Highway(nn.Module):
    def __init__(self, d, l):
        super(Highway, self).__init__()
        self.layers = nn.ModuleList([HighwayLayer(d) for _ in range(l)])

    def forward(self, inp):
        out = inp
        for layer in self.layers:
            out = layer(out)
        return out


class WordLSTM(nn.Module):
    def __init__(self, args, n_words):
        super(WordLSTM, self).__init__()
        dropout = 0.5
        self.lstm_l = 2
        if args.arch_large:
            self.word_d = 650
            self.lstm_h = 650
        else:
            self.word_d = 200
            self.lstm_h = 200

        self.n_words = n_words
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(n_words, self.word_d)
        self.lstm = nn.LSTM(self.word_d, self.lstm_h, self.lstm_l, dropout=dropout)
        self.decoder = nn.Linear(self.lstm_h, n_words)

        self.init_weights()

    def init_weights(self):
        initrange = 0.05
        for p in self.parameters():
            nn.init.uniform_(p, -initrange, initrange)

    def forward(self, input, hidden):
        # emb = self.drop(self.encoder(input))
        emb = self.encoder(input)
        output, hidden = self.lstm(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.n_words)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        shape = (self.lstm_l, bsz, self.lstm_h)
        return (weight.new_zeros(shape), weight.new_zeros(shape))


class CharLSTM(nn.Module):
    def __init__(self, args, n_chars, n_words):
        super(CharLSTM, self).__init__()

        # architecture configurations
        self.n_words = n_words
        self.n_chars = n_chars
        self.char_d = 15
        self.lstm_l = 2
        if args.arch_large:
            self.char_w = [w+1 for w in range(7)]
            self.char_h = [min(200, w * 50) for w in self.char_w]
            self.highway_l = 2
            self.lstm_h = 650
        else:
            self.char_w = [w+1 for w in range(6)]
            self.char_h = [w * 25 for w in self.char_w]
            self.highway_l = 1
            self.lstm_h = 300
        self.emb_d = sum(self.char_h)
        dropout = 0.5

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(self.n_chars, self.char_d)
        self.char_cnn = CharCNN(self.char_d, self.char_w, self.char_h)
        self.highway = Highway(self.emb_d, self.highway_l)
        self.lstm = nn.LSTM(self.emb_d, self.lstm_h, self.lstm_l, dropout=dropout)
        self.decoder = nn.Linear(self.lstm_h, n_words)

        self.init_weights()

    def init_weights(self):
        initrange = 0.05
        for p in self.parameters():
            nn.init.uniform_(p, -initrange, initrange)

    def forward(self, input, hidden):
        # step x. get character embeddings [s, b, w_len]
        emb = self.encoder(input)
        # step x. swap time and channel dimensions [s, b, w_len, char_d]
        emb = emb.permute((0, 1, 3, 2))
        # step x. collapse batch and seq len dims [s, b, char_d, w_len]
        seq_len, bsz = emb.shape[:2]
        emb = emb.view((emb.shape[0]*emb.shape[1], *emb.shape[2:]))
        # step x. pass through char cnn [s*b, char_d, w_len]
        emb = self.char_cnn(emb)
        # step x. pass through highway network [s*b, emb_d]
        emb = self.highway(emb)
        # step x. uncollapse batch and seq len [s*b, emb_d]
        emb = emb.view((seq_len, bsz, self.emb_d))
        # step x. standard LSTM language modeling code
        # emb = self.drop(emb) # [s, b, emb_d]
        output, hidden = self.lstm(emb, hidden) # [s, b, lstm_h]
        output = self.drop(output) # [s, b, lstm_h]
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.n_words)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        shape = (self.lstm_l, bsz, self.lstm_h)
        return (weight.new_zeros(shape), weight.new_zeros(shape))

