import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder

# vanilla seq2seq model
class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_size, hidden_size, num_layers, dpt=0.3):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, hidden_size, num_layers, dpt)
        #self.decoder = BasicDecoder(trg_vocab_size, embed_size, hidden_size, num_layers, dpt)
        self.decoder = BasicAttentionDecoder(trg_vocab_size, embed_size, 2 * hidden_size, num_layers, dpt)        
        #self.decoder = BasicBahdanauAttnDecoder(trg_vocab_size, embed_size, hidden_size, num_layers, dpt)

    def encode(self, src):
        return self.encoder(src) 
    
    def generate(self, trg, src, enc_output, hidden=None):
        return self.decoder(trg, enc_output, hidden)
        
    def forward(self, src, trg):
        enc_output = self.encoder(src)
        output, hidden = self.decoder(trg, enc_output)
        return output, hidden

# vanilla seq2seq decoder
class BasicDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dpt=0.3):
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

class BasicAttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dpt=0.3):
        super(BasicAttentionDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # test
        self.embedding.weight.data.copy_((torch.rand(vocab_size, embed_size) - 0.5) * 2)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dpt)
        self.linear1 = nn.Linear(2 * hidden_size, embed_size)
        self.linear2 = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(p=dpt)
        
        # weight tying
        self.linear2.weight = self.embedding.weight
    
    
    def forward(self, trg, encoded_src, hidden=None):
        trg_len = trg.size(0)
        batch_size = trg.size(1)

        x = self.embedding(trg)
        x = self.dropout(x)
        output, hidden = self.lstm(x, hidden)
        h_e = encoded_src.transpose(0, 1)
        h_d = output.transpose(0, 1)
        
        attn = torch.bmm(h_d, h_e.transpose(1, 2))
        attn = F.softmax(attn, dim=2)
        
        context = torch.bmm(attn, h_e).transpose(0, 1) # t_o x b x d
        
        output = torch.cat((context, output), dim=2)
        
        output = torch.tanh(self.linear1(output))
        output = self.dropout(output) 
        output = F.log_softmax(self.linear2(output), dim=2)
        
        return output, hidden
    
class BasicBahdanauAttnDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dpt=0.3):
        super(BasicBahdanauAttnDecoder, self).__init__()
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + 2 * hidden_size, hidden_size, num_layers, dropout=dpt)
        # dropout for LSTM
        self.dropout = nn.Dropout(p=dpt)
        # for calculating attention scores
        self.attn_annot = nn.Linear(2 * hidden_size, hidden_size)  # input is |F| x B x 2N, output |F| x B x N
        self.attn_hidden = nn.Linear(hidden_size, hidden_size)  # each input is |F| x B x N, output |F| x B x N
        self.other = nn.Linear(hidden_size, 1)
        # final linear layer before applying (Log) Softmax
        self.penult = nn.Linear(hidden_size + 2 * hidden_size, embed_size)
        self.out = nn.Linear(embed_size, vocab_size)
        # weight tying
        self.out.weight = self.embedding.weight

    def step_forward(self, word_input, last_hidden, last_context, annot_scores, annotations):
        # TODO: return attention scores as well for visualization later
        # input: word vec, h_{t-1}, c_{t-1}, annotation scores
        # output: h_t, c_t

        # construct new input by concatenating word vec with context vec
        # dimension: 1 x B x (M + 2N)
        new_input = torch.cat((word_input, last_context), dim=1).unsqueeze(0)
        # calculate new hidden vector
        new_output, new_hidden = self.lstm(new_input, last_hidden)
        # scores computed using current hidden state, dimension: 1 x B x N
        hidden_scores = self.attn_hidden(new_output)
        # calculate attention weights. weight matrix size is |F| x B x 1
        attn_weights = self.other(torch.tanh(hidden_scores + annot_scores))
        attn_weights = F.softmax(attn_weights, dim=0)
        # calculate new context vector, (size is |F| x B x 2N)
        # by multiplying matrices of dimensions
        new_context = torch.bmm(attn_weights.permute(1, 2, 0),
                                annotations.transpose(1, 0))[:, 0, :]
        return new_output, new_hidden, new_context

    def forward(self, trg, encoded_src, hidden=None, return_attn=False, word_dpt=0):
        # embed the target words
        trg_embeddings = self.embedding(trg)
        # pre-compute annotation scores to save resources. dimension: |F| x B x N
        annotations = encoded_src
        annot_scores = self.attn_annot(encoded_src)
        # init context vector as all 0s (dimension is B x 2N)
        context = torch.zeros(encoded_src.size()[1:]).type_as(annotations)
        all_scores = None
        for trg in trg_embeddings:
            output, hidden, context = self.step_forward(trg, hidden, context, annot_scores, annotations)
            # append output (h_t) and context to the overall matrix
            stacked = torch.cat((output, context.unsqueeze(0)), dim=2)
            all_scores = stacked if all_scores is None else torch.cat((all_scores, stacked))
        # apply final linear layer and then softmax
        scores = F.log_softmax(self.out(self.penult(all_scores)), dim=2)
        return scores, hidden