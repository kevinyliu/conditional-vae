import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

from nltk.translate import bleu_score
import spacy
import numpy as np


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


def perp_bound(model, val_iter, gpu=True):
    """
    Calculates bound on perplexity using ELBO.
    This only works for VAE models.
    """
    model.eval()
    loss = nn.NLLLoss(size_average=True)  # ignore <pad> TODO check that this is the right index for pad
    val_nre = 0
    val_kl = 0
    for batch in tqdm(val_iter):
        src, trg = (batch.src.cuda(), batch.trg.cuda()) if gpu else (batch.src, batch.trg)

        re, kl, hidden = model(src, trg)
        
        kl = kl.sum() / len(kl)
        nre = loss(re[:-1, :, :].view(-1, re.size(2)), trg[1:, :].view(-1))

        neg_elbo = nre + kl

        val_nre += nre.item()
        val_kl += kl.item()

    val_nre /= len(val_iter)
    val_kl /= len(val_iter)
    val_elbo = val_nre + val_kl
    model.train()
    return np.exp(val_elbo), val_elbo, val_nre, val_kl  


def perplexity(model, val_iter, gpu=True):
    """
    Calculates perplexity.
    This does not work for VAE.
    """
    model.eval()
    loss = nn.NLLLoss(size_average=True)  # TODO remove pad form this calc
    val_loss = 0
    for batch in tqdm(val_iter):
        if gpu:
            src, trg = batch.src.cuda(), batch.trg.cuda()

        else:
            src, trg = batch.src, batch.trg

        ll, _ = model.forward(src, trg)
        # we have to eliminate the <s> start of sentence token in the trg, otherwise it will not be aligned
        nll = loss(ll[:-1, :, :].view(-1, ll.size(2)), trg[1:, :].view(-1))
        val_loss += nll.item()
    val_loss /= len(val_iter)
    model.train()
    return np.exp(val_loss), val_loss


def bleu(reference, predict):
    """Compute sentence-level bleu score.
    Args:
        reference (list[str])
        predict (list[str])
    """

    if len(predict) == 0:
        if len(reference) == 0:
            return 1.0
        else:
            return 0.0

    # use a maximum of 4-grams. If 4-grams aren't present, use only lower n-grams.
    n = min(4, len(reference), len(predict))
    weights = tuple([1. / n] * n)  # uniform weight on n-gram precisions
    return bleu_score.sentence_bleu([reference], predict, weights, emulate_multibleu=False)


def rouge(reference, predict, rouge_type='rouge-1'):
    """
    Compute rouge score.
    Args:
        reference (list[str])
        predict (list[str])
        rouge_type 'rouge-1', 'rouge-2', 'rouge-l'
    """
    from rouge import Rouge
    rouge = Rouge()
    scores = rouge.get_scores(' '.join(predict), ' '.join(reference))
    return scores[0][rouge_type]['f']