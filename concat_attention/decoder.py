from define import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from model_util import *

class Decoder(nn.Module):
    def __init__(self, target_size, opts):
        super(Decoder, self).__init__()
        self.opts = opts
        self.attention = Attention(opts)
        self.embed = nn.Embedding(target_size, embed_size, padding_idx=0)
#        init_wt_normal(self.embed.weight)
        self.lstm = nn.LSTMCell(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, target_size)

    def forward(self, t_input, hx, cx, encoder_outputs, encoder_features, mask_tensor):
        embed = self.embed(t_input)
        hx, cx = self.lstm(embed, (hx, cx) )
        final_dist = self.attention(
                hx, encoder_outputs, encoder_features, mask_tensor)
        return final_dist, hx, cx

class Attention(nn.Module):
    def __init__(self, opts):
        super(Attention, self).__init__()
        self.opts = opts
        self.W_s = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, decoder_hx, encoder_outputs, encoder_features, mask_tensor):
        t_k, b, n = list(encoder_outputs.size())
        dec_feature = self.W_s(decoder_hx)
        dec_feature = dec_feature.unsqueeze(0).expand(t_k, b, n)
        att_features = encoder_features + dec_feature

        e = torch.tanh(att_features)
        scores = self.v(e)
        align_weight = torch.softmax(scores, dim=0) * mask_tensor

        content_vector = (align_weight * encoder_outputs).sum(0)
        concat = torch.cat((content_vector, decoder_hx), 1)
        final_dist = torch.tanh(self.linear(concat))
        return final_dist
