import torch
import torch.nn as nn
import torch.nn.functional as F

# seq2seq decoder
class BasicDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, latent_size, num_layers, dpt=0.2):
        super(BasicDecoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + latent_size, hidden_size, num_layers, dropout=dpt)
#         self.linear = nn.Linear(hidden_size, vocab_size)
        self.linear = nn.Linear(3*hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=dpt)

    def forward(self, trg, z, encoded_src, hidden=None, word_dpt=0.0):
        trg_len = trg.size(0)
        batch_size = trg.size(1)
        h_src = encoded_src[-1,:,:].view(1, batch_size, -1)

        x = self.embedding(trg)

        # word dropout
        mask = torch.bernoulli((1 - word_dpt) * torch.ones(trg_len, batch_size)).unsqueeze(2).expand_as(x)
        x = x * mask

        x = torch.cat((x, z.unsqueeze(0).repeat(trg.size(0),1,1)), dim=2)
        output, hidden = self.lstm(x, hidden)
        output = self.dropout(output)
        output = torch.cat((output, h_src.repeat(trg_len,1,1)), dim=2) # to make it exactly like the vanilla seq2seq, otherwise comment out (and above too .linear)
        output = F.log_softmax(self.linear(output), dim=2)
        return output, hidden