from define import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        opts = { "bidirectional": True }
        self.encoder = Encoder(opts)
        self.decoder = Decoder()
        self.attention = Attention()

    def forward(self, source=None, target=None, train=False, generate=False):
        if train:
            target = target.t()
            loss = 0
            encoder_outputs , hx, cx = self.encoder(source)
            mask_tensor = source.t().eq(PADDING).unsqueeze(-1)

            for words_f, word_t in zip(target[:-1], target[1:]):
                hx , cx = self.decoder(words_f, hx, cx)
                new_hx = self.attention(hx, encoder_outputs, mask_tensor)
                loss += F.cross_entropy(
                   self.decoder.linear(new_hx),
                       word_t , ignore_index=0)
            return loss

        elif generate:
            encoder_outputs , hx, cx = self.encoder(source)
            mask_tensor = source.t().eq(PADDING).unsqueeze(-1)
            word_id = torch.tensor( [ target_dict["[START]"] ] ).cuda()
            doc = []
            loop = 0
            while True:
                hx , cx = self.decoder(word_id, hx, cx)
                hx_new = self.attention(hx, encoder_outputs, mask_tensor)
                word_id = torch.tensor([ torch.argmax(F.softmax(self.decoder.linear(hx_new), dim=1).data[0]) ]).cuda()
                loop += 1
                if loop >= 200 or int(word_id) == target_dict['[STOP]']:
                    break
                doc.append(word_id)
            return doc


class Encoder(nn.Module):
    def __init__(self, opts):
        super(Encoder, self).__init__()
        self.opts = opts
        self.embed = nn.Embedding(source_size, embed_size, padding_idx=0)
        self.drop = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=self.opts["bidirectional"])

    def forward(self, sentences):
        '''
            return
                encoder_ouput, hx, cx
            option
                bidirectional
        '''
        input_lengths = torch.tensor(
            [seq.size(-1) for seq in sentences])
        embed = self.embed(sentences)
        embed = self.drop(embed)
        sequence = rnn.pack_padded_sequence(embed, input_lengths, batch_first=True)

        packed_output, (hx, cx) = self.lstm(sequence)
        output, _ = rnn.pad_packed_sequence(
            packed_output
        )
        if self.opts["bidirectional"]:
            output = output[:, :, :hidden_size] + output[:, :, hidden_size:]
            hx = hx.view(-1, 2 , sentences.size(0), hidden_size).sum(1)
            cx = cx.view(-1, 2 , sentences.size(0), hidden_size).sum(1)
        hx = hx.view(sentences.size(0) , -1)
        cx = cx.view( sentences.size(0) , -1)
        return output, hx, cx

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(target_size, embed_size, padding_idx=0)
        self.drop = nn.Dropout(p=args.dropout)
        self.lstm = nn.LSTMCell(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, target_size)

    def forward(self, target_words, hx, cx):
        embed = self.embed(target_words)
        embed = self.drop(embed)
        hx, cx = self.lstm(embed, (hx, cx) )
        return hx, cx

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, decoder_hx, hx_list, mask_tensor):
        attention_weights = (decoder_hx * hx_list).sum(-1, keepdim=True)
        masked_score = attention_weights.masked_fill_(mask_tensor, float('-inf'))
        align_weight = F.softmax(masked_score, 0)
        content_vector = (align_weight * hx_list).sum(0)
        concat = torch.cat((content_vector, decoder_hx), 1)
        hx_attention = torch.tanh(self.linear(concat))
        return hx_attention
