from define import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *

def create_mask(sentence_words, hidden_size):
    return torch.cat( [ sentence_words.unsqueeze(-1) ] * hidden_size, 1).cuda(device=sentence_words.device)

class EncoderDecoder(nn.Module):
    def __init__(self, source_size, target_size, hidden_size):
        super(EncoderDecoder, self).__init__()
        self.w_encoder = WordEncoder(source_size, hidden_size)
        self.w_decoder = WordDecoder(target_size, hidden_size)

    def forward(self, source, target):

        def init(source_len):
            hx= torch.zeros(source_len, hidden_size).cuda(device=source.device)
            cx= torch.zeros(source_len, hidden_size).cuda(device=source.device)
            return hx, cx

        source_len = len(source)
        ew_hx, ew_cx = init(source_len)

        source = source.t()
        target = target.t()
        loss = 0

        ew_hx_list = []
        ew_masks = []
        for words in source:
            ew_hx , ew_cx = self.w_encoder(words, ew_hx, ew_cx)
            ew_hx_list.append(ew_hx)
            masks = torch.cat( [ words.unsqueeze(-1) ] , 1)
            ew_masks.append( torch.unsqueeze(masks, 0) )

        dw_hx, dw_cx = ew_hx, ew_cx
        ew_hx_list = torch.stack(ew_hx_list, 0)
        ew_masks = torch.cat(ew_masks)

        inf = torch.full((len(source), source_len), float("-inf")).cuda(device=source.device)
        inf = torch.unsqueeze(inf, -1)

        lines_t_last = target[1:]
        lines_f_last = target[:(len(source) - 1)]

        for words_f, word_t in zip(lines_f_last, lines_t_last):
            dw_hx , dw_cx = self.w_decoder(words_f, dw_hx, dw_cx)
            dw_new_hx = self.w_decoder.attention(
                        dw_hx, ew_hx_list, ew_masks, inf)
            loss += F.cross_entropy(
               self.w_decoder.linear(dw_new_hx),
                   word_t , ignore_index=0)
        loss = torch.tensor(loss, requires_grad=True).unsqueeze(0).cuda(device=source.device)
        return loss

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
        w_hx, w_cx = self.lstm(source_k, (w_hx, w_cx))
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
        w_hx, w_cx = self.lstm(target_k , (w_hx, w_cx) )
        w_hx = torch.where(mask == 0, b_hx, w_hx)
        w_cx = torch.where(mask == 0, b_cx, w_cx)
        return w_hx, w_cx

    def attention(self, decoder_hx, ew_hx_list, ew_mask, inf):
        attention_weights = (decoder_hx * ew_hx_list).sum(-1, keepdim=True)
        masked_score = torch.where(ew_mask == 0, inf, attention_weights)
        align_weight = F.softmax(masked_score, 0)
        content_vector = (align_weight * ew_hx_list).sum(0)
        concat = torch.cat((content_vector, decoder_hx), 1)
        hx_attention = torch.tanh(self.attention_linear(concat))
        return hx_attention
