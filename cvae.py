import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

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

    def encode(self, src):
        encoded_src = self.src_encoder(src)
        mu_prior, log_var_prior = self.p(encoded_src)
        p_normal = Normal(loc=mu_prior, scale=log_var_prior.mul(0.5).exp())
        return encoded_src, p_normal

    def generate(self, trg, p_normal, encoded_src, hidden=None):
        z = p_normal.sample()
        ll, hidden = self.decoder(trg, z, encoded_src, hidden)
        return ll, hidden

    def forward(self, src, trg):
        encoded_src = self.src_encoder(src)
        encoded_trg = self.trg_encoder(trg)

        mu_prior, log_var_prior = self.p(encoded_src)
        mu_posterior, log_var_posterior = self.q(encoded_src, encoded_trg)

        p_normal = Normal(loc=mu_prior, scale=log_var_prior.mul(0.5).exp())
        q_normal = Normal(loc=mu_posterior, scale=log_var_posterior.mul(0.5).exp())
        kl = kl_divergence(q_normal, p_normal).sum()

        # z = self.reparameterize(mu_posterior, log_var_posterior)
        z = q_normal.rsample()

        ll, hidden = self.decoder(trg, z, encoded_src)

        return ll, kl, hidden
               # //mu_prior, log_var_prior, mu_posterior, log_var_posterior
