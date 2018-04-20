import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dpt=0.3):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dpt, bidirectional=True)
        self.dropout = nn.Dropout(p=dpt)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x = self.dropout(x)
        output, hidden = self.lstm(x, hidden)
        return output
