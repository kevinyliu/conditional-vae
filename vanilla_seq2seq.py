import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder


# vanilla seq2seq decoder
class BasicDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dpt=0.2):
        super(BasicDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dpt)
        self.linear = nn.Linear(3*hidden_size, vocab_size)
#         self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=dpt)

    def forward(self, trg, encoded_src, hidden=None):
        trg_len = trg.size(0)
        batch_size = trg.size(1)
        h_src = encoded_src[-1,:,:].view(1, batch_size, -1)

        x = self.embedding(trg)
        output, hidden = self.lstm(x, hidden)
        output = self.dropout(output)
        output = torch.cat((output, h_src.repeat(trg_len,1,1)), dim=2)
        output = F.log_softmax(self.linear(output), dim=2)
        return output, hidden


# vanilla seq2seq model
class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_size, hidden_size, num_layers, dpt=0.2):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, hidden_size, num_layers, dpt)
        self.decoder = BasicDecoder(trg_vocab_size, embed_size, hidden_size, num_layers, dpt)

    def forward(self, src, trg):
        enc_output = self.encoder(src)
        output, hidden = self.decoder(trg, enc_output)
        return output, hidden
