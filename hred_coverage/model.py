from define import *
from encoder import *
from decoder import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

class Hierachical(nn.Module):
    def __init__(self , opts):
        super(Hierachical, self).__init__()
        self.opts = opts
        self.w_encoder = WordEncoder(opts)
        self.s_encoder = SentenceEncoder(opts)
        self.w_decoder = WordDecoder()
        self.s_decoder = SentenceDecoder(opts)

    def forward(self, articles_sentences=None, summaries_sentences=None, train=False, generate=False):

        word_hx_outputs = []
        for sentences in articles_sentences:
            hx, cx = self.w_encoder(sentences.cuda())
            word_hx_outputs.append(hx)
        word_hx_outputs = torch.stack(word_hx_outputs, 0)

        sentence_outputs, sentence_features, s_hx, s_cx = self.s_encoder(word_hx_outputs)
        mask_tensor = [ torch.tensor([ [ words[0].item() ] for words in sentences ])
                for sentences in articles_sentences ]
        mask_tensor = torch.stack(mask_tensor, 0).gt(0).float().cuda()
        w_hx, w_cx = s_hx, s_cx
        coverage_vector = torch.zeros(mask_tensor.size()).cuda()
        if train:
            loss = 0
            for summaries_sentence in summaries_sentences:
                summaries_sentence = summaries_sentence.t().cuda()
                for words_before, words_after in zip(summaries_sentence[:-1], summaries_sentence[1:]):
                    w_hx, w_cx = self.w_decoder(words_before, w_hx, w_cx)
                    loss += F.cross_entropy(
                        self.w_decoder.linear(w_hx), words_after , ignore_index=0)
                final_dist, s_hx, s_cx, align_weight, next_coverage_vector = self.s_decoder(w_hx, s_hx, s_cx,
                    sentence_outputs, sentence_features, coverage_vector, mask_tensor)

                if self.opts["coverage_vector"]:
                    align_weight = align_weight.squeeze()
                    coverage_vector = coverage_vector.squeeze()
                    step_coverage_loss = torch.sum(torch.min(align_weight, coverage_vector), 0)
                    step_coverage_loss = torch.mean(step_coverage_loss)
                    cov_loss_wt = 1
                    loss += (cov_loss_wt * step_coverage_loss)
                    coverage_vector = next_coverage_vector
                s_hx = final_dist
                w_hx = final_dist
            return loss

        elif generate:
            loop_w = 0
            loop_s = 0
            doc = []
            while True:
                loop_w = 0
                word_id = torch.tensor( [ target_dict["[START]"] ] ).cuda()
                sentence = []
                while True:
                    w_hx, w_cx = self.w_decoder(word_id, w_hx, w_cx)
                    word_id = torch.tensor([ torch.argmax(
                        self.w_decoder.linear(w_hx), dim=1).data[0]]).cuda()
                    loop_w += 1
                    if loop_w >= 100 or int(word_id) == target_dict['[STOP]'] or int(word_id) == target_dict['[EOD]']:
                        break
                    sentence.append(word_id.item())
                if loop_s >= 10 or int(word_id) == target_dict['[EOD]']:
                    break
                final_dist, s_hx, s_cx, align_weight, next_coverage_vector = self.s_decoder(w_hx, s_hx, s_cx,
                    sentence_outputs, sentence_features, coverage_vector, mask_tensor)
                if self.opts["coverage_vector"]:
                    coverage_vector = next_coverage_vector
                s_hx = final_dist
                w_hx = final_dist
                doc.append(sentence)
                loop_s += 1
            return doc
