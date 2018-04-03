import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import spacy

def torchtext_extract(d=-1, MAX_LEN=20, MIN_FREQ=5, BATCH_SIZE=32):
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    DE = data.Field(tokenize=tokenize_de)
    EN = data.Field(tokenize=tokenize_en, init_token=BOS_WORD, eos_token=EOS_WORD)  # only target needs BOS/EOS

    train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN), 
                                         filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
                                         len(vars(x)['trg']) <= MAX_LEN)

    DE.build_vocab(train.src, min_freq=MIN_FREQ)
    EN.build_vocab(train.trg, min_freq=MIN_FREQ)

    train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=BATCH_SIZE, device=d,
                                                  repeat=False, sort_key=lambda x: len(x.src))

    return train_iter, val_iter, test, DE, EN

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# TODO
def perplexity():
    pass

# TODO
def bleu():
    pass