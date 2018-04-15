import utils
import vanilla_train
import vanilla_seq2seq

train_iter, val_iter, test, DE, EN = utils.torchtext_extract()

model_name = "test_vanilla_4_500_750"

gpu = True
checkpt = True

num_layers = 4
embed_size = 500
hidden_size = 500

num_epochs=100


model = vanilla_seq2seq.Seq2Seq(len(DE.vocab), len(EN.vocab), embed_size, hidden_size, num_layers)
if gpu:
    model.cuda()

vanilla_train.train(model, model_name, train_iter, val_iter, DE, EN, num_epochs, gpu, checkpoint=checkpt)
