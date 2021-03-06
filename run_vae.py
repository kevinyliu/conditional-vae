import utils
import train
import cvae

model_name = "cvae_wdp"

gpu = True
device = 0
max_len = 100

num_layers = 2
embed_size = 300
hidden_size = 300
latent_size = 32

lr = 0.002
dpt = 0.3
word_dpt = 0.25
min_kl = 0.0
kl_coef = 1

num_epochs = 50
batch_size = 64

share_encoder = True

train_iter, val_iter, test, DE, EN = utils.torchtext_extract(d=device, MAX_LEN=max_len, BATCH_SIZE=batch_size)

anneal = utils.kl_anneal_custom

model = cvae.CVAE(len(DE.vocab), len(EN.vocab), embed_size, hidden_size, latent_size, num_layers, dpt,
                  share_params=share_encoder)
if gpu:
    model.cuda()

print("Number of parameters: {}".format(utils.count_parameters(model)))

train.train(model, model_name, train_iter, val_iter, DE, EN, anneal=anneal, num_epochs=num_epochs, gpu=gpu, lr=lr,
            kl_coef=kl_coef, min_kl=min_kl, word_dpt=word_dpt, checkpoint=True)
