import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

import itertools, os, re
from nltk.translate import bleu_score
import tempfile, subprocess
import spacy
import numpy as np

import beam_search


def torchtext_extract(DATASET="IWSLT", d=-1, MAX_LEN=100, MIN_FREQ=5, BATCH_SIZE=32):
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

    if DATASET == "IWSLT":
        train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN),
                                             filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                                             len(vars(x)['trg']) <= MAX_LEN)
    elif DATASET == "WMT14":
        train, val, test = datasets.WMT14.splits(exts=('.de', '.en'), fields=(DE, EN),
                                             filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                                             len(vars(x)['trg']) <= MAX_LEN)

    DE.build_vocab(train.src, min_freq=MIN_FREQ)
    EN.build_vocab(train.trg, min_freq=MIN_FREQ)

    train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=BATCH_SIZE, device=d,
                                                  repeat=False, sort_key=lambda x: len(x.src))

    return train_iter, val_iter, test, DE, EN


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def kl_anneal_sigmoid(epoch, gpu=True):
    """
    Sigmoidal annealing schedule for KL weight
    """
    alpha = torch.tensor(2 * (1/(1 + np.exp(-epoch/2)) - 1/2), requires_grad=False)
    if gpu: alpha = alpha.cuda()
    return alpha


def kl_anneal_linear(epoch, epoch_full=15, gpu=True):
    """
    Linear annealing schedule for KL weight
    """
    alpha = min(1/epoch_full * epoch, 1)
    alpha = torch.tensor(alpha, requires_grad=False)
    if gpu: alpha = alpha.cuda()
    return alpha

def kl_anneal_custom(epoch):
    if epoch < 5:
        return 0
    if epoch < 15:
        return (epoch - 5) / 10
    return 1.0

def eval_vae(model, val_iter, pad, gpu=True):
    """
    Calculates bound on perplexity using ELBO.
    This only works for VAE models.
    """
    model.eval()
    loss = nn.NLLLoss(size_average=True, ignore_index=pad)
    val_nre = 0
    val_kl_word = 0
    val_kl_sent = 0
    for batch in tqdm(val_iter):
        src, trg = (batch.src.cuda(), batch.trg.cuda()) if gpu else (batch.src, batch.trg)
        
        trg_word_cnt = (trg != pad).float().sum() - trg.size(1)
        
        re, kl, hidden = model(src, trg)
        
        kl_word = kl.sum() / trg_word_cnt # KL by word
        kl_sent = kl.sum() # KL by sent

        nre = loss(re[:-1, :, :].view(-1, re.size(2)), trg[1:, :].view(-1))

        neg_elbo = nre + kl_word

        val_nre += nre.item()
        val_kl_word += kl_word.item()
        val_kl_sent += kl_sent.item()

    val_nre /= len(val_iter)
    val_kl_word /= len(val_iter)
    val_kl_sent /= len(val_iter)
    val_elbo = val_nre + val_kl
    model.train()
    return np.exp(val_elbo), val_elbo, val_nre, val_kl_word, val_kl_sent


def eval_seq2seq(model, val_iter, pad, gpu=True):
    """
    Calculates perplexity.
    This does not work for VAE.
    """
    model.eval()
    loss = nn.NLLLoss(size_average=True, ignore_index=pad)
    val_loss = 0
    for batch in tqdm(val_iter):
        src, trg = (batch.src.cuda(), batch.trg.cuda()) if gpu else (batch.src, batch.trg)

        ll, _ = model(src, trg)
        # we have to eliminate the <s> start of sentence token in the trg, otherwise it will not be aligned
        nll = loss(ll[:-1, :, :].view(-1, ll.size(2)), trg[1:, :].view(-1))
        val_loss += nll.item()
    val_loss /= len(val_iter)
    model.train()
    return np.exp(val_loss), val_loss


def bleu(reference, predict):
    """
    Compute sentence-level bleu score.
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


def moses_multi_bleu(outputs, references, lw=False):
    '''Outputs, references are lists of strings. Calculates BLEU score using https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl -- Python function from Google '''

    # Save outputs and references as temporary text files
    out_file = tempfile.NamedTemporaryFile()
    out_file.write('\n'.join(outputs).encode('utf-8'))
    out_file.write(b'\n')
    out_file.flush()  # ?
    ref_file = tempfile.NamedTemporaryFile()
    ref_file.write('\n'.join(references).encode('utf-8'))
    ref_file.write(b'\n')
    ref_file.flush()  # ?
    # Use moses multi-bleu script
    with open(out_file.name, 'r') as read_pred:
        bleu_cmd = ['./multi-bleu.perl']
        bleu_cmd = bleu_cmd + ['-lc'] if lw else bleu_cmd
        bleu_cmd = bleu_cmd + [ref_file.name]
        try:
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode('utf-8')
            # print(bleu_out)
            bleu_score = float(re.search(r'BLEU = (.+?),', bleu_out).group(1))
        except subprocess.CalledProcessError as error:
            print(error)
            raise Exception('Something wrong with bleu script')
            bleu_score = 0.0

    # Close temporary files
    out_file.close()
    ref_file.close()

    return bleu_score

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


def generate(model, val_iter, TRG_TEXT, k=10, max_len=100, gpu=True):
    """
    Generates top k best sentences given trained model.
    """
    bos = TRG_TEXT.vocab.stoi['<s>']
    eos = TRG_TEXT.vocab.stoi['</s>']
    pad = TRG_TEXT.vocab.stoi['<pad>']
    
    filter_token = [pad] 
    
    output = []
    
    for batch in tqdm(val_iter):
        trg = batch.trg
        src = batch.src
        for i in range(src.size(1)):
            src_sent = src[:, i:i+1]
            best_options = beam_search.beam_search(model, src_sent, bos, eos, k, max_len, filter_token, gpu)
            
            sentence_trg = ""
            sentence_src = ""
            sentence = []
            for word in best_options[0][1]:
                sentence += [TRG_TEXT.vocab.itos[word]]
                sentence_src += TRG_TEXT.vocab.itos[word] + " "
            for word in trg[: , i]:
                sentence_trg += TRG_TEXT.vocab.itos[word] + " "
            
#             print(sentence_src + "  |  " + sentence_trg)
            output.append(sentence)
    
    return output

def strip(sentence):
    while '<pad>' in sentence:
        sentence.remove('<pad>')
    while '<s>' in sentence:
        sentence.remove('<s>')
    while '</s>' in sentence:
        sentence.remove('</s>')
        
def test_generation(model, val_iter, TRG_TEXT, k=10, max_len=100, gpu=True):
    """
    Calls generate to get the generated sentences from beam search.
    Then evaluates them with blue and rouge.
    """
    sentences = generate(model, val_iter, TRG_TEXT, k, max_len, gpu)
    for s in sentences:
        strip(s)
    b = 0
    r = 0
    index = 0
    for batch in val_iter:
        trg = batch.trg
        for i in range(trg.size(1)):
            t = []
            for word in trg[:, i]:
                t += [TRG_TEXT.vocab.itos[word]]
            strip(t)
            b += bleu(t, sentences[index])
            r += rouge(t, sentences[index])
        index += 1
    b /= len(sentences)
    r /= len(sentences)
    
    return b, r

def test_multibleu(model, val_iter, TRG_TEXT, k=10, max_len=120, gpu=True):

    sentences = generate(model, val_iter, TRG_TEXT, k, max_len, gpu)

    sentences_out = []
    for s in sentences:
        strip(s)
        sent = ' '.join(j for j in s)
        sentences_out.append(sent)

    sentences_ref = []
    for batch in val_iter:
        trg = batch.trg
        for i in range(trg.size(1)):
            t = []
            for word in trg[:, i]:
                t += [TRG_TEXT.vocab.itos[word]]
            strip(t)
            sent_ref = ' '.join(j for j in t)
            sentences_ref.append(sent_ref)

    return moses_multi_bleu(sentences_out, sentences_ref)
