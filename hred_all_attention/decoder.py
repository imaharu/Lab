from define import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

class WordDecoder(nn.Module):
    def __init__(self):
        super(WordDecoder, self).__init__()
        self.embed = nn.Embedding(target_size, embed_size, padding_idx=0)
        self.gru = nn.GRUCell(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, target_size)
        self.w_attention = WordAttention()

    def forward(self, summary_words, w_hx, g_attn_hx_outputs, g_attn_hx_features, word_mask):
        embed = self.embed(summary_words)
        w_hx = self.gru(embed, w_hx)
        word_final_dist = self.w_attention(w_hx, g_attn_hx_outputs, g_attn_hx_features, word_mask)
        return word_final_dist

class WordAttention(nn.Module):
    def __init__(self):
        super(WordAttention, self).__init__()
        self.W_w = nn.Linear(hidden_size, hidden_size)
        self.v_w = nn.Linear(hidden_size, 1)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, w_hx, g_attn_hx_outputs, g_attn_hx_features, word_mask):
        t_k, b, n = list(g_attn_hx_outputs.size())
        dec_feature = self.W_w(w_hx)
        dec_feature = dec_feature.unsqueeze(0).expand(t_k, b, n)
        attn_features = g_attn_hx_features + dec_feature
        e = torch.tanh(attn_features)
        scores = self.v_w(e)
        align_weight = torch.softmax(scores, dim=0) * word_mask # sen_len x Batch x 1

        content_vector = (align_weight * g_attn_hx_outputs).sum(0)
        concat = torch.cat((content_vector, w_hx), 1)
        final_dist = torch.tanh(self.linear(concat))
        return final_dist

### Sentence ###

class SentenceDecoder(nn.Module):
    def __init__(self, opts):
        super(SentenceDecoder, self).__init__()
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.s_attention = SentenceAttention(opts)

    def forward(self, w_hx, s_hx, encoder_outputs, encoder_features, coverage_vector, sentence_mask):
        s_hx = self.gru(w_hx, s_hx)
        final_dist, align_weight, next_coverage_vector = self.s_attention(
                s_hx, encoder_outputs, encoder_features, coverage_vector, sentence_mask)
        return final_dist, s_hx, align_weight, next_coverage_vector

class SentenceAttention(nn.Module):
    def __init__(self, opts):
        super(SentenceAttention, self).__init__()
        self.opts = opts
        self.W_s = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.coverage = Coverage()

    def forward(self, s_hx, encoder_outputs ,encoder_feature, coverage_vector, sentence_mask):
        t_k, b, n = list(encoder_outputs.size())
        dec_feature = self.W_s(s_hx)
        dec_feature = dec_feature.unsqueeze(0).expand(t_k, b, n)
        attn_features = encoder_feature + dec_feature

        if self.opts["coverage_vector"]:
            att_features = self.coverage.getFeature(coverage_vector, attn_features)
        e = torch.tanh(attn_features)
        scores = self.v(e)
        align_weight = torch.softmax(scores, dim=0) * sentence_mask # sen_len x Batch x 1

        if self.opts["coverage_vector"]:
            next_coverage_vector = self.coverage.getNextCoverage(coverage_vector, align_weight)
        else:
            next_coverage_vector = coverage_vector

        content_vector = (align_weight * encoder_outputs).sum(0)
        concat = torch.cat((content_vector, s_hx), 1)
        final_dist = torch.tanh(self.linear(concat))
        return final_dist, align_weight, next_coverage_vector

class Coverage(nn.Module):
    def __init__(self):
        super(Coverage, self).__init__()
        self.W_c = nn.Linear(1, hidden_size)

    def getFeature(self, coverage_vector, attn_features):
        coverage_input = coverage_vector.view(-1, 1)
        coverage_features = self.W_c(coverage_input).unsqueeze(-1)
        coverage_features = coverage_features.view(-1, attn_features.size(1), hidden_size)
        attn_features += coverage_features
        return attn_features

    def getNextCoverage(self, coverage_vector, align_weight):
        next_coverage_vector = coverage_vector + align_weight
        return next_coverage_vector
