import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from encoder import Encoder
from decoder import BasicDecoder
from inferer import Prior, ApproximatePosterior

class CVAE(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_size, hidden_size, latent_size, num_layers):
        super(CVAE, self).__init__()
        self.src_encoder = Encoder(src_vocab_size, embed_size, hidden_size, num_layers)
        self.trg_encoder = Encoder(trg_vocab_size, embed_size, hidden_size, num_layers)
        self.decoder = BasicDecoder(trg_vocab_size, embed_size, hidden_size, latent_size, num_layers)
        self.p = Prior(hidden_size, latent_size)
        self.q = ApproximatePosterior(hidden_size, latent_size)

    def reparameterize(self, mu, log_var):
        eps = Variable(torch.randn(mu.size(0), mu.size(1)))
        if mu.is_cuda:
            eps = eps.cuda()
        return mu + eps * torch.exp(log_var/2)

    def step():
        pass

    def forward(self, src, trg):
        encoded_src = self.src_encoder(src)
        encoded_trg = self.trg_encoder(trg)

        mu_prior, log_var_prior = self.p(encoded_src)
        mu_posterior, log_var_posterior = self.q(encoded_src, encoded_trg)

        z = self.reparameterize(mu_posterior, log_var_posterior)

        ll, hidden = self.decoder(trg, z, encoded_src)

        return ll, hidden, mu_prior, log_var_prior, mu_posterior, log_var_posterior
