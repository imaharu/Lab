from define import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

class Decoder(nn.Module):
    def __init__(self, target_size, opts):
        super(Decoder, self).__init__()
        self.opts = opts
        self.embed_target = nn.Embedding(target_size, embed_size, padding_idx=0)
        self.lstm_target = nn.LSTMCell(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, target_size)

    def forward(self, t_input, hx, cx):
        embed = self.embed_target(t_input)
        hx, cx = self.lstm_target(embed, (hx, cx) )
        return hx, cx
