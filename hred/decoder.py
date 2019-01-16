from define import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

class WordDecoder(nn.Module):
    def __init__(self):
        super(WordDecoder, self).__init__()
        self.embed = nn.Embedding(target_size, embed_size, padding_idx=0)
        self.drop = nn.Dropout(p=args.dropout)
        self.lstm = nn.LSTMCell(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, target_size)

    def forward(self, summary_words, w_hx, w_cx):
        embed = self.embed(summary_words)
        embed = self.drop(embed)
        '''
            0があるときは、where
        '''
        if torch.nonzero(summary_words.eq(0)).size(0):
            before_w_hx , before_w_cx = w_hx, w_cx
            mask = torch.cat( [ summary_words.unsqueeze(-1) ] * hidden_size, 1)
            w_hx, w_cx = self.lstm(embed, (w_hx, w_cx) )
            w_hx = torch.where(mask == 0, before_w_hx, w_hx)
            w_cx = torch.where(mask == 0, before_w_cx, w_cx)
        else:
            w_hx, w_cx = self.lstm(embed, (w_hx, w_cx) )
        return w_hx, w_cx

class SentenceDecoder(nn.Module):
    def __init__(self):
        super(SentenceDecoder, self).__init__()
        self.drop = nn.Dropout(p=args.dropout)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)

    def forward(self, w_hx, s_hx, s_cx):
        w_hx = self.drop(w_hx)
        s_hx, s_cx = self.lstm(w_hx, (s_hx, s_cx) )
        return s_hx ,s_cx
