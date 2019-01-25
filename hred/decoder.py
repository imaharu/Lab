from define import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

class WordDecoder(nn.Module):
    def __init__(self):
        super(WordDecoder, self).__init__()
        self.embed = nn.Embedding(target_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTMCell(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, target_size)

    def forward(self, summary_words, w_hx, w_cx):
        embed = self.embed(summary_words)
        w_hx, w_cx = self.lstm(embed, (w_hx, w_cx) )
        return w_hx, w_cx

class SentenceDecoder(nn.Module):
    def __init__(self, opts):
        super(SentenceDecoder, self).__init__()
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)

    def forward(self, w_hx, s_hx, s_cx):
        s_hx, s_cx = self.lstm(w_hx, (s_hx, s_cx) )
        return s_hx, s_cx
