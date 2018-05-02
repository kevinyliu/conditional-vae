import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

import encoder
import decoder
import inferer

class CVAE(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_size, hidden_size, latent_size, num_layers, dpt=0.3, word_dpt=0.0):
        super(CVAE, self).__init__()
        self.if_zero = False
        
        self.src_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.trg_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.src_embedding.weight.data.copy_((torch.rand(src_vocab_size, embed_size) - 0.5) * 2)
        self.trg_embedding.weight.data.copy_((torch.rand(trg_vocab_size, embed_size) - 0.5) * 2)

        
        self.src_encoder_t = encoder.Encoder(src_vocab_size, embed_size, hidden_size, num_layers, dpt, self.src_embedding)
        self.src_encoder_i = encoder.Encoder(src_vocab_size, embed_size, hidden_size, num_layers, dpt, self.src_embedding)
        self.trg_encoder = encoder.Encoder(trg_vocab_size, embed_size, hidden_size, num_layers, dpt, self.trg_embedding)
        
        #self.decoder = decoder.BasicDecoder(trg_vocab_size, embed_size, hidden_size, latent_size, num_layers, dpt, word_dpt, self.trg_embedding)
        self.decoder = decoder.BasicAttentionDecoder(trg_vocab_size, embed_size, 2 * hidden_size, latent_size, num_layers, dpt, word_dpt, self.trg_embedding)
        #self.decoder = decoder.DummyDecoder(trg_vocab_size, embed_size, hidden_size, latent_size, num_layers, dpt, self.trg_embedding)
        #self.decoder = decoder.BahdanauAttnDecoder(trg_vocab_size, embed_size, hidden_size, latent_size, num_layers, dpt, self.trg_embedding)
        
        self.p = inferer.Prior(hidden_size, latent_size, dpt)
        #self.q = inferer.ApproximatePosterior(hidden_size, latent_size, dpt)
        #self.q = inferer.AttentionApproximatePosterior(src_vocab_size, trg_vocab_size, embed_size, hidden_size, latent_size, dpt, self.src_embedding, self.trg_embedding)
        #self.q = inferer.AttentionApproximatePosterior(src_vocab_size, trg_vocab_size, embed_size, hidden_size, latent_size, dpt)
        self.q = inferer.LSTMAttentionApproximatePosterior(hidden_size, latent_size, dpt)
        
    def encode_source_translate(self, src):
        return self.src_encoder_t(src) 

    def encode_source_infer(self, src):
        return self.src_encoder_i(src) 
    
    def generate(self, trg, src, encoded_src_t, encoded_src_i, hidden=None):
        z, _ = self.p(src, encoded_src_i) # at eval time, we don't sample, we just use the mean
        if self.if_zero:
            z.zero_()
        ll, hidden = self.decoder(trg, z, encoded_src_t, hidden)
        return ll, hidden

    def forward(self, src, trg):
        encoded_src_t = self.src_encoder_t(src)
        encoded_src_i = self.src_encoder_i(src)
        encoded_trg = self.trg_encoder(trg)

        mu_prior, log_var_prior = self.p(src, encoded_src_i)
        mu_posterior, log_var_posterior = self.q(src, encoded_src_i, trg, encoded_trg)

#         p_normal = Normal(loc=torch.zeros(mu_prior.size()).cuda(), scale =torch.ones(log_var_prior.size()).cuda())

        p_normal = Normal(loc=mu_prior, scale=log_var_prior.mul(0.5).exp())
        q_normal = Normal(loc=mu_posterior, scale=log_var_posterior.mul(0.5).exp())
        kl = kl_divergence(q_normal, p_normal)

        z = mu_posterior #q_normal.rsample()

        ll, hidden = self.decoder(trg, z, encoded_src_t)

        return ll, kl, hidden




