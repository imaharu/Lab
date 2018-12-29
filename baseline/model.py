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

    def forward(self, source, target):
        loss = 0
        es_hx_list = []
        es_mask = []
#        device = torch.cuda.current_device()
        ew_hx, ew_cx = self.set_zero(device)

        for words in lines:
            ew_hx , ew_cx = self.w_encoder(words, ew_hx, ew_cx)
            s_mask = create_mask(lines[0], hidden_size)

            es_hx_list.append(es_hx[args.num_layer - 1])
            es_mask.append( torch.cat([ lines[0].unsqueeze(-1) ] , 1).unsqueeze(0))

        dw_hx, ds_cx = ew_hx, ew_cx
        es_hx_list = torch.stack(es_hx_list, 0)
        es_mask = torch.cat(es_mask)
        inf = torch.full((max_dsn, batch_size), float("-inf")).cuda(device=device)
        inf = torch.unsqueeze(inf, -1)

        print(source)
        lines_t_last = source[1:]
        lines_f_last = source[:(len(source) - 1)]
        print("lines f", lines_f_last)
        exit()
        for words_f, word_t in zip(lines_f_last, lines_t_last):
            dw_hx , dw_cx = self.w_decoder(words_f, dw_hx, dw_cx)
            loss += F.cross_entropy(
               self.w_decoder.linear(dw_hx),
                   word_t , ignore_index=0)

            ds_new_hx = self.s_decoder.attention(
                        ds_hx,es_hx_list, es_mask, inf)
        return loss

    def max_sentence_num(self, docs):
        max_sentence_num = max([*map(lambda x: len(x), docs )])
        return max_sentence_num

    def init(self, device):
        init = torch.zeros(batch_size, hidden_size).cuda(device=device)
        return init

    def set_zero(self, device):
        ew_hx, ew_cx = [], []
        for i in range(args.num_layer):
            ew_hx.append(self.init(device))
            ew_cx.append(self.init(device))
        return ew_hx, ew_cx

class WordEncoder(nn.Module):
    def __init__(self, source_size, hidden_size):
        super(WordEncoder, self).__init__()
        self.embed_source = nn.Embedding(source_size, hidden_size, padding_idx=0)
        self.drop_source = nn.Dropout(p=args.dropout)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)

    def forward(self, words, w_hx, w_cx):
        source_k = self.embed_source(words)
        source_k = self.drop_source(source_k)
        mask = create_mask(words, hidden_size)
        b_hx, b_cx = w_hx, w_cx
        w_hx, w_cx = self.lstm(source_k, w_hx, w_cx)
        w_hx = torch.where(mask == 0, b_hx, w_hx)
        w_cx = torch.where(mask == 0, b_cx, w_cx)
        return w_hx, w_cx

class WordDecoder(nn.Module):
    def __init__(self, target_size, hidden_size):
        super(WordDecoder, self).__init__()
        self.embed_target = nn.Embedding(target_size, hidden_size, padding_idx=0)
        self.drop_target = nn.Dropout(p=args.dropout)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.attention_linear = nn.Linear(hidden_size * 2, hidden_size)
        self.linear = nn.Linear(hidden_size, target_size)

    def forward(self, words, w_hx, w_cx):
        target_k = self.embed_target(words)
        target_k = self.drop_target(target_k)
        mask = create_mask(words, hidden_size)
        b_hx, b_cx = w_hx, w_cx
        w_hx, w_cx = self.lstm(target_k ,w_hx, w_cx)
        w_hx = torch.where(mask == 0, b_hx, w_hx)
        w_cx = torch.where(mask == 0, b_cx, w_cx)
        return w_hx, w_cx

    def attention(self, decoder_hx, es_hx_list, es_mask, inf):
        attention_weights = (s_hx * es_hx_list).sum(-1, keepdim=True)
        masked_score = torch.where(es_mask == 0, inf, attention_weights)
        align_weight = F.softmax(masked_score, 0)
        content_vector = (align_weight * es_hx_list).sum(0)
        concat = torch.cat((content_voctor, decoder_hx), 1)
        hx_attention = torch.tanh(self.attention_linear(concat))
        return hx_attention
