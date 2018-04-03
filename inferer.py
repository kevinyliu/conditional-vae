import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Prior(nn.Module):
    def __init__(self, hidden_size, latent_size):
        super(Prior, self).__init__()

        self.latent_size = latent_size
        self.hidden_size = hidden_size

        self.linear = nn.Linear(2*hidden_size, latent_size)
        self.linear_mu = nn.Linear(latent_size, latent_size)
        self.linear_var = nn.Linear(latent_size, latent_size)


    def forward(self, encoded_src):
        encoded_src = encoded_src.transpose(0,1).transpose(1,2)
        h_src = F.avg_pool1d(encoded_src, encoded_src.size(2)).view(encoded_src.size(0), -1)
        h_z = F.tanh(self.linear(h_src))
        mu = self.linear_mu(h_z)
        log_var = self.linear_var(h_z)

        return mu, log_var


class ApproximatePosterior(nn.Module):
    def __init__(self, hidden_size, latent_size):
        super(ApproximatePosterior, self).__init__()

        self.latent_size = latent_size
        self.hidden_size = hidden_size

        self.linear = nn.Linear(4*hidden_size, latent_size)
        self.linear_mu = nn.Linear(latent_size, latent_size)
        self.linear_var = nn.Linear(latent_size, latent_size)

    def forward(self, encoded_src, encoded_trg):
        encoded_src = encoded_src.transpose(0,1).transpose(1,2)
        encoded_trg = encoded_trg.transpose(0,1).transpose(1,2)

        h_src = F.avg_pool1d(encoded_src, encoded_src.size(2)).view(encoded_src.size(0), -1)
        h_trg = F.avg_pool1d(encoded_trg, encoded_trg.size(2)).view(encoded_trg.size(0), -1)
        h_z = F.tanh(self.linear(torch.cat((h_src, h_trg), dim=1)))
        mu = self.linear_mu(h_z)
        log_var = self.linear_var(h_z)

        return mu, log_var
