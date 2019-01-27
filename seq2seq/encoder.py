from define import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

class Encoder(nn.Module):
    def __init__(self, source_size, opts):
        super(Encoder, self).__init__()
        self.opts = opts
        self.embed = nn.Embedding(source_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=self.opts["bidirectional"])
        self.W_h = nn.Linear(hidden_size, hidden_size)

    def forward(self, sentences):
        except_flag = False
        b = sentences.size(0)

        input_lengths = sentences.ne(0).sum(-1)
        sorted_lengths, indices = torch.sort(input_lengths, descending=True)
        sentences = sentences[indices]
        masked_select = sorted_lengths.masked_select(sorted_lengths.ne(0))

        # if all 0 sentence is appeared
        if not torch.equal(sorted_lengths, masked_select):
            s_b = b
            b = masked_select.size(0)
            sentences = sentences.narrow(0, 0, b)
            sorted_lengths = masked_select
            except_flag = True

        embed = self.embed(sentences)
        sequence = rnn.pack_padded_sequence(embed, sorted_lengths, batch_first=True)
        _, (hx, cx) = self.lstm(sequence)

        if self.opts["bidirectional"]:
            hx = hx.view(-1, 2 , b, hidden_size).sum(1)
            cx = cx.view(-1, 2 , b, hidden_size).sum(1)
        hx = hx.view(b , -1)
        cx = cx.view(b , -1)

        if except_flag:
            zeros = torch.zeros((s_b - b, hidden_size)).cuda()
            hx = torch.cat((hx, zeros), 0)
            cx = torch.cat((cx, zeros), 0)

        inverse_indices = indices.sort()[1] # Inverse permutation
        hx = hx[inverse_indices]
        hx = cx[inverse_indices]
        return hx, cx
