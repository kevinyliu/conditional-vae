import utils
import train
import cvae

train_iter, val_iter, test, DE, EN = utils.torchtext_extract()

model_name = "vae_simple_noprior_30ep"

gpu = True

num_layers = 2
embed_size = 100
hidden_size = 100
latent_size = 100

num_epochs=30

anneal = utils.kl_anneal_linear

eos = EN.vocab.stoi["</s>"]
bos = EN.vocab.stoi["<s>"]
pad = EN.vocab.stoi["<pad>"]

filter_beam = [pad, bos]


model = cvae.CVAE(len(DE.vocab), len(EN.vocab), embed_size, hidden_size, latent_size, num_layers)
if gpu:
    model.cuda()

    
train.train(model, model_name, train_iter, val_iter, DE, EN, anneal, num_epochs, gpu)