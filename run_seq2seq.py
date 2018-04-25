import utils
import vanilla_train
import vanilla_seq2seq

model_name = "seq2seq_dotattn_initembed_100"

gpu = True
device = 0

num_layers = 2
embed_size = 300
hidden_size = 300

lr = 0.002
dpt = 0.3

num_epochs = 50
batch_size = 64

train_iter, val_iter, test, DE, EN = utils.torchtext_extract(d=device, BATCH_SIZE=batch_size)

model = vanilla_seq2seq.Seq2Seq(len(DE.vocab), len(EN.vocab), embed_size, hidden_size, num_layers, dpt)
if gpu:
    model.cuda()

print("Number of parameters: {}".format(utils.count_parameters(model)))
    
vanilla_train.train(model, model_name, train_iter, val_iter, DE, EN, num_epochs, gpu, lr, checkpoint=True)
