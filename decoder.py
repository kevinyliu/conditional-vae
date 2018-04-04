import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# seq2seq decoder
class BasicDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, latent_size, num_layers, dpt=0.2):
        super(BasicDecoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + latent_size, hidden_size, num_layers, dropout=dpt)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=dpt)

    def forward(self, trg, z, encoded_src, hidden=None):
        x = self.embedding(trg)
        x = torch.cat((x, z.unsqueeze(0).repeat(trg.size(0),1,1)), dim=2)
        output, hidden = self.lstm(x, hidden)
        output = self.dropout(output)
        output = F.log_softmax(self.linear(output), dim=2)
        return output, hidden