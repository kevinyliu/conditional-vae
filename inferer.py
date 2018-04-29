import torch
import torch.nn as nn
import torch.nn.functional as F

class Prior(nn.Module):
    def __init__(self, hidden_size, latent_size, dpt=0.3):
        super(Prior, self).__init__()

        self.latent_size = latent_size
        self.hidden_size = hidden_size

        self.linear = nn.Linear(2*hidden_size, latent_size)
        self.linear_mu = nn.Linear(latent_size, latent_size)
        self.linear_var = nn.Linear(latent_size, latent_size)
        
        self.dropout = nn.Dropout(p=dpt)


    def forward(self, src, encoded_src):
        encoded_src = encoded_src.transpose(0,1).transpose(1,2)
        h_src = F.avg_pool1d(encoded_src, encoded_src.size(2)).view(encoded_src.size(0), -1)
        h_src = self.dropout(h_src)
        h_z = F.tanh(self.linear(h_src))
        h_z = self.dropout(h_z)
        mu = self.linear_mu(h_z)
        log_var = self.linear_var(h_z)

        return mu, log_var


class ApproximatePosterior(nn.Module):
    def __init__(self, hidden_size, latent_size, dpt=0.3):
        super(ApproximatePosterior, self).__init__()

        self.latent_size = latent_size
        self.hidden_size = hidden_size

        self.linear = nn.Linear(4*hidden_size, latent_size)
        self.linear_mu = nn.Linear(latent_size, latent_size)
        self.linear_var = nn.Linear(latent_size, latent_size)
        
        self.dropout = nn.Dropout(p=dpt)

    def forward(self, src, encoded_src, trg, encoded_trg):
        encoded_src = encoded_src.transpose(0,1).transpose(1,2)
        encoded_trg = encoded_trg.transpose(0,1).transpose(1,2)

        h_src = F.avg_pool1d(encoded_src, encoded_src.size(2)).view(encoded_src.size(0), -1)
        h_trg = F.avg_pool1d(encoded_trg, encoded_trg.size(2)).view(encoded_trg.size(0), -1)
        
        h_src = self.dropout(h_src)
        h_trg = self.dropout(h_trg)
        
        h_z = F.tanh(self.linear(torch.cat((h_src, h_trg), dim=1)))
        h_z = self.dropout(h_z)
        mu = self.linear_mu(h_z)
        log_var = self.linear_var(h_z)

        return mu, log_var

class AttentionApproximatePosterior(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_size, hidden_size, latent_size, dpt=0.3, src_embedding=None, trg_embedding=None):
        super(AttentionApproximatePosterior, self).__init__()

        self.latent_size = latent_size
        self.hidden_size = hidden_size
        
        if src_embedding is not None:
            self.src_embedding = src_embedding
        else:
            self.src_embedding = nn.Embedding(src_vocab_size, embed_size)
            self.src_embedding.weight.data.copy_((torch.rand(src_vocab_size, embed_size) - 0.5) * 2)        
        
        if trg_embedding is not None:
            self.trg_embedding = trg_embedding
        else:
            self.trg_embedding = nn.Embedding(trg_vocab_size, embed_size)
            self.trg_embedding.weight.data.copy_((torch.rand(trg_vocab_size, embed_size) - 0.5) * 2)   
        
        self.linear_src = nn.Linear(2*embed_size, hidden_size)
        self.linear_trg = nn.Linear(2*embed_size, hidden_size)

        self.linear = nn.Linear(2*hidden_size, latent_size)
        self.linear_mu = nn.Linear(latent_size, latent_size)
        self.linear_var = nn.Linear(latent_size, latent_size)
        
        self.dropout = nn.Dropout(p=dpt)

    def forward(self, src, encoded_src, trg, encoded_trg):
        
        x_src = self.src_embedding(src).transpose(0, 1) # b x t_i x e
        x_trg = self.trg_embedding(trg).transpose(0, 1) # b x t_o x e
        
        # Currently just basic dot attention on embeddings. May want to change later
        
        attn_src = torch.bmm(x_trg, x_src.transpose(1, 2)) # b x t_o x t_i
        attn_src = F.softmax(attn_src, dim=2)
        c_src = torch.bmm(attn_src, x_src) # b x t_o x e
        
        attn_trg = torch.bmm(x_src, x_trg.transpose(1, 2)) # b x t_i x t_o
        attn_trg = F.softmax(attn_trg, dim=2)
        c_trg = torch.bmm(attn_trg, x_trg) # b x t_i x e
        
        c_src = self.dropout(c_src)
        c_trg = self.dropout(c_trg)
        
        v_src = F.tanh(self.linear_src(torch.cat((c_trg, x_src), dim=2)).sum(dim=1)) # b x h
        v_trg = F.tanh(self.linear_trg(torch.cat((c_src, x_trg), dim=2)).sum(dim=1)) # b x h
        
        h_z = F.tanh(self.linear(torch.cat((v_src, v_trg), dim=1)))
        h_z = self.dropout(h_z)
        mu = self.linear_mu(h_z)
        log_var = self.linear_var(h_z)

        return mu, log_var

class LSTMAttentionApproximatePosterior(nn.Module):
    def __init__(self, hidden_size, latent_size, dpt=0.3):
        super(LSTMAttentionApproximatePosterior, self).__init__()

        self.latent_size = latent_size
        self.hidden_size = hidden_size
        
        self.linear_src1 = nn.Linear(2*hidden_size, 2*hidden_size)
        self.linear_trg1 = nn.Linear(2*hidden_size, 2*hidden_size)
        
        self.linear_src2 = nn.Linear(4*hidden_size, hidden_size)
        self.linear_trg2 = nn.Linear(4*hidden_size, hidden_size)

        self.linear = nn.Linear(2*hidden_size, latent_size)
        self.linear_mu = nn.Linear(latent_size, latent_size)
        self.linear_var = nn.Linear(latent_size, latent_size)
        
        self.dropout = nn.Dropout(p=dpt)

    def forward(self, src, encoded_src, trg, encoded_trg):
        
        h_src = F.tanh(self.linear_src1(encoded_src.transpose(0, 1))) # b x t_i x h
        h_trg = F.tanh(self.linear_trg1(encoded_trg.transpose(0, 1))) # b x t_o x h        
        
        attn_src = torch.bmm(h_trg, h_src.transpose(1, 2)) # b x t_o x t_i
        attn_src = F.softmax(attn_src, dim=2)
        c_src = torch.bmm(attn_src, h_src) # b x t_o x h
        
        attn_trg = torch.bmm(h_src, h_trg.transpose(1, 2)) # b x t_i x t_o
        attn_trg = F.softmax(attn_trg, dim=2)
        c_trg = torch.bmm(attn_trg, h_trg) # b x t_i x h
        
        c_src = self.dropout(c_src)
        c_trg = self.dropout(c_trg)
        
        v_src = F.tanh(self.linear_src2(torch.cat((c_trg, h_src), dim=2)).sum(dim=1)) # b x h
        v_trg = F.tanh(self.linear_trg2(torch.cat((c_src, h_trg), dim=2)).sum(dim=1)) # b x h
        
        h_z = F.tanh(self.linear(torch.cat((v_src, v_trg), dim=1)))
        h_z = self.dropout(h_z)
        mu = self.linear_mu(h_z)
        log_var = self.linear_var(h_z)

        return mu, log_var

    