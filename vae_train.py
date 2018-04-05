import utils
import train
import cvae

train_iter, val_iter, test, DE, EN = utils.torchtext_extract()

model_name = "vae_simple"

gpu = True

num_layers = 2
embed_size = 100
hidden_size = 100
latent_size = 100

num_epochs=100


model = cvae.CVAE(len(DE.vocab), len(EN.vocab), embed_size, hidden_size, latent_size, num_layers)
if gpu:
    model.cuda()

    
train.train(model, model_name, train_iter, val_iter, DE, EN, num_epochs, gpu)