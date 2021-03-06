import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dpt=0.3, embedding=None):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.embedding.weight.data.copy_((torch.rand(vocab_size, embed_size) - 0.5) * 2)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dpt, bidirectional=True)
        self.dropout = nn.Dropout(p=dpt)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x = self.dropout(x)
        output, hidden = self.lstm(x, hidden)
        return output


class SharedEncoder(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, dpt=0.3):
        super(SharedEncoder, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dpt, bidirectional=True)
        self.dropout = nn.Dropout(p=dpt)

    def forward(self, x, hidden=None):
        x = self.dropout(x)
        output, hidden = self.lstm(x, hidden)
        return output
