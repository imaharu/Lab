from define_sin import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *

class HierachicalEncoderDecoder(nn.Module):
    def __init__(self, source_size, target_size, hidden_size):
        super(HierachicalEncoderDecoder, self).__init__()
        self.encoder = Encoder(source_size, hidden_size)
        self.decoder = Decoder(target_size, hidden_size)

class Encoder(nn.Module):
    def __init__(self, source_size, hidden_size):
        super(Encoder, self).__init__()
        self.w_encoder = WordEncoder(source_size, hidden_size)
        self.s_encoder = SentenceEncoder(hidden_size)

class WordEncoder(nn.Module):
    def __init__(self, source_size, hidden_size):
        super(WordEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.source_size = source_size
        self.embed_source = nn.Embedding(source_size, hidden_size, padding_idx=0)
        self.drop_source = nn.Dropout(p=args.dropout)
        self.lstm = nn.ModuleList([ nn.LSTMCell(hidden_size, hidden_size) for i in range(args.num_layer)])

    def create_mask(self ,sentence_words):
        return torch.cat( [ sentence_words.unsqueeze(-1) ] * hidden_size, 1)

    def multi_layer(self, source_k, mask, hx, cx):
        for i, lstm in enumerate(self.lstm):
            b_hx , b_cx = hx[i], cx[i]
            if i == 0:
                hx[i], cx[i] = lstm(source_k, (hx[i], cx[i]) )
            else:
                hx[i], cx[i] = lstm(hx[i - 1], (hx[i], cx[i]) )
            hx[i] = torch.where(mask == 0, b_hx, hx[i])
            cx[i] = torch.where(mask == 0, b_cx, cx[i])
        return hx, cx

    def forward(self, words, w_hx, w_cx):
        source_k = self.embed_source(words)
        source_k = self.drop_source(source_k)
        mask = self.create_mask(words)
        w_hx, w_cx = self.multi_layer(source_k, mask ,w_hx, w_cx)
        return w_hx, w_cx

    def init(self):
        init = torch.zeros(batch_size, hidden_size).cuda()
        return init

class SentenceEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(SentenceEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.drop_source_s = nn.Dropout(p=args.dropout)
        self.lstm = nn.ModuleList([ nn.LSTMCell(hidden_size, hidden_size) for i in range(args.num_layer)])

    def multi_layer(self, w_hx, mask, hx, cx):
        for i, lstm in enumerate(self.lstm):
            b_hx , b_cx = hx[i], cx[i]
            if i == 0:
                hx[i], cx[i] = lstm(w_hx, (hx[i], cx[i]) )
            else:
                hx[i], cx[i] = lstm(hx[i - 1], (hx[i], cx[i]) )
            hx[i] = torch.where(mask == 0, b_hx, hx[i])
            cx[i] = torch.where(mask == 0, b_cx, cx[i])
        return hx, cx

    def forward(self, mask, w_hx, s_hx, s_cx):
        w_hx = self.drop_source_s(w_hx[args.num_layer - 1])
        s_hx, s_cx = self.multi_layer(w_hx, mask, s_hx, s_cx )
        return s_hx, s_cx

    def init(self):
        init = torch.zeros(batch_size, hidden_size).cuda()
        return init

class Decoder(nn.Module):
    def __init__(self, target_size, hidden_size):
        super(Decoder, self).__init__()
        self.w_decoder = WordDecoder(target_size, hidden_size)
        self.s_decoder = SentenceDecoder(hidden_size)

class WordDecoder(nn.Module):
    def __init__(self, target_size, hidden_size):
        super(WordDecoder, self).__init__()
        self.embed_target = nn.Embedding(target_size, hidden_size, padding_idx=0)
        self.drop_target = nn.Dropout(p=args.dropout)
        self.lstm = nn.ModuleList([ nn.LSTMCell(hidden_size, hidden_size) for i in range(args.num_layer)])
        self.linear = nn.Linear(hidden_size, target_size)

    def create_mask(self ,sentence_words):
        return torch.cat( [ sentence_words.unsqueeze(-1) ] * hidden_size, 1)

    def multi_layer(self, source_k, mask, hx, cx):
        for i, lstm in enumerate(self.lstm):
            b_hx , b_cx = hx[i], cx[i]
            if i == 0:
                hx[i], cx[i] = lstm(source_k, (hx[i], cx[i]) )
            else:
                hx[i], cx[i] = lstm(hx[i - 1], (hx[i], cx[i]) )
            hx[i] = torch.where(mask == 0, b_hx, hx[i])
            cx[i] = torch.where(mask == 0, b_cx, cx[i])
        return hx, cx

    def forward(self, words, w_hx, w_cx):
        target_k = self.embed_target(words)
        target_k = self.drop_target(target_k)
        mask = self.create_mask(words)
        w_hx, w_cx = self.multi_layer(target_k, mask ,w_hx, w_cx)
        return w_hx, w_cx

class SentenceDecoder(nn.Module):
    def __init__(self, hidden_size):
        super(SentenceDecoder, self).__init__()
        self.drop_target_doc = nn.Dropout(p=args.dropout)
        self.lstm = nn.ModuleList([ nn.LSTMCell(hidden_size, hidden_size) for i in range(args.num_layer)])
        self.attention_linear = nn.Linear(hidden_size * 2, hidden_size)

    def multi_layer(self, w_hx, mask, hx, cx):
        for i, lstm in enumerate(self.lstm):
            b_hx , b_cx = hx[i], cx[i]
            if i == 0:
                hx[i], cx[i] = lstm(w_hx, (hx[i], cx[i]) )
            else:
                hx[i], cx[i] = lstm(hx[i - 1], (hx[i], cx[i]) )
            hx[i] = torch.where(mask == 0, b_hx, hx[i])
            cx[i] = torch.where(mask == 0, b_cx, cx[i])
        return hx, cx

    def forward(self, mask, w_hx, s_hx, s_cx):
        w_hx = self.drop_target_doc(w_hx[args.num_layer - 1])
        s_hx, s_cx = self.multi_layer(w_hx, mask, s_hx, s_cx )
        return s_hx, s_cx

    def attention(self, s_hx, es_hx_list, es_mask, inf):
        dot = (s_hx * es_hx_list).sum(-1, keepdim=True)
        dot = torch.where(es_mask == 0, inf, dot)
        a_t = F.softmax(dot, 0)
        d = (a_t * es_hx_list).sum(0)
        concat = torch.cat((d, s_hx), 1)
        hx_attention = torch.tanh(self.attention_linear(concat))
        return hx_attention
