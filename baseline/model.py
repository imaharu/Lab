from define import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *

def create_mask(sentence_words, hidden_size):
    return torch.cat( [ sentence_words.unsqueeze(-1) ] * hidden_size, 1)

class HierachicalEncoderDecoder(nn.Module):
    def __init__(self, source_size, target_size, hidden_size):
        super(HierachicalEncoderDecoder, self).__init__()
        self.w_encoder = WordEncoder(source_size, hidden_size)
        self.w_decoder = WordDecoder(target_size, hidden_size)
        self.s_encoder = SentenceEncoder(hidden_size)
        self.s_decoder = SentenceDecoder(hidden_size)

    def forward(self, source, target):
        loss = 0
        es_hx_list = []
        es_mask = []
        print(source)
        exit()
        device = torch.cuda.current_device()
        ew_hx, ew_cx, es_hx, es_cx = self.set_zero(device)
        max_dsn = self.max_sentence_num(source)
        max_dtn = self.max_sentence_num(target)
        for i in range(max_dsn):
            ew_hx, ew_cx = es_hx, es_cx
            lines = torch.tensor([ x[i]  for x in source ]).t().cuda(device=device)
            for words in lines:
                ew_hx , ew_cx = self.w_encoder(words, ew_hx, ew_cx)
            s_mask = create_mask(lines[0], hidden_size)
            es_hx , es_cx = self.s_encoder(s_mask, ew_hx, es_hx, es_cx)

            es_hx_list.append(es_hx[args.num_layer - 1])
            es_mask.append( torch.cat([ lines[0].unsqueeze(-1) ] , 1).unsqueeze(0))

        ds_hx, ds_cx = es_hx, es_cx
        es_hx_list = torch.stack(es_hx_list, 0)
        es_mask = torch.cat(es_mask)
        inf = torch.full((max_dsn, batch_size), float("-inf")).cuda(device=device)
        inf = torch.unsqueeze(inf, -1)

        for i in range(max_dtn):
            if i == 0:
                dw_hx, dw_cx = ds_hx, ds_cx
            else:
                dw_hx, dw_cx = ds_hx, ds_cx
                dw_hx[0] = ds_new_hx
            lines = torch.tensor([ x[i]  for x in target ]).t().cuda(device=device)
            # t -> true, f -> false
            lines_t_last = lines[1:]
            lines_f_last = lines[:(len(lines) - 1)]

            for words_f, word_t in zip(lines_f_last, lines_t_last):
                dw_hx , dw_cx = self.w_decoder(words_f, dw_hx, dw_cx)

                loss += F.cross_entropy(
                    self.w_decoder.linear(
                        dw_hx[args.num_layer - 1]),
                        word_t , ignore_index=0)

            s_mask = create_mask(lines[0], hidden_size)
            ds_hx , ds_cx = self.s_decoder(s_mask, dw_hx, ds_hx, ds_cx)
            ds_new_hx = self.s_decoder.attention(
                        ds_hx[args.num_layer - 1],
                        es_hx_list, es_mask, inf)
        return loss

    def max_sentence_num(self, docs):
        max_sentence_num = max([*map(lambda x: len(x), docs )])
        return max_sentence_num

    def init(self, device):
        init = torch.zeros(batch_size, hidden_size).cuda(device=device)
        return init

    def set_zero(self, device):
        ew_hx, ew_cx, es_hx, es_cx = [], [], [], []
        for i in range(args.num_layer):
            ew_hx.append(self.init(device))
            ew_cx.append(self.init(device))
            es_hx.append(self.init(device))
            es_cx.append(self.init(device))
        return ew_hx, ew_cx, es_hx, es_cx

class WordEncoder(nn.Module):
    def __init__(self, source_size, hidden_size):
        super(WordEncoder, self).__init__()
        self.embed_source = nn.Embedding(source_size, hidden_size, padding_idx=0)
        self.drop_source = nn.Dropout(p=args.dropout)
        self.lstm = nn.ModuleList([ nn.LSTMCell(hidden_size, hidden_size) for i in range(args.num_layer)])

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
        mask = create_mask(words, hidden_size)
        w_hx, w_cx = self.multi_layer(source_k, mask ,w_hx, w_cx)
        return w_hx, w_cx

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

class WordDecoder(nn.Module):
    def __init__(self, target_size, hidden_size):
        super(WordDecoder, self).__init__()
        self.embed_target = nn.Embedding(target_size, hidden_size, padding_idx=0)
        self.drop_target = nn.Dropout(p=args.dropout)
        self.lstm = nn.ModuleList([ nn.LSTMCell(hidden_size, hidden_size) for i in range(args.num_layer)])
        self.linear = nn.Linear(hidden_size, target_size)

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
        mask = create_mask(words, hidden_size)
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
