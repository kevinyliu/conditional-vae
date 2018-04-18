import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from tqdm import tqdm
import os
import numpy as np


def train(model, model_name, train_iter, val_iter, SRC_TEXT, TRG_TEXT, num_epochs=20, gpu=False, lr=0.001, weight_decay=0, checkpoint=False):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=30, factor=0.25, verbose=True, cooldown=6)
    pad = TRG_TEXT.vocab.stoi['<pad>']
    loss = nn.NLLLoss(size_average=True, ignore_index=pad)
    cur_best = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_nll = 0
        for batch in tqdm(train_iter):
            src, trg = (batch.src.cuda(), batch.trg.cuda()) if gpu else (batch.src, batch.trg)

            ll, hidden = model(src, trg)

            nll = loss(ll[:-1, :, :].view(-1, ll.size(2)), trg[1:, :].view(-1))

            train_nll += nll.item()

            optimizer.zero_grad()
            nll.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()

        train_nll /= len(train_iter)
        train_perp = np.exp(train_nll)

        val_perp, val_nll = utils.eval_seq2seq(model, val_iter, pad, gpu)

        # greedy search
        bleu_val = utils.test_multibleu(model, val_iter, TRG_TEXT, k=1, gpu=gpu)
        scheduler.step(bleu_val)  
        
        results = 'Epoch: {}\n' \
                  '\tVALID PPL: {:.4f} NLL: {:.4f}\n'\
                  '\tTRAIN PPL: {:.4f} NLL: {:.4f}\n'\
                  '\tBLEU Greedy: {:.4f}'\
                  .format(epoch+1, val_perp, val_nll, train_perp, train_nll, bleu_val)
        
        if not (epoch + 1) % 2:
            bleu, _ = utils.test_multibleu(model, val_iter, TRG_TEXT, gpu=gpu)
            results += '\n\tBLEU: {:.4f}'.format(bleu)

        print(results)
        
        if not (epoch + 1) % 1:
            local_path = os.getcwd()
            model_path = local_path + "/" + model_name
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            eval_file = model_path + "/" + "eval.txt"

            if epoch == 0:
                f = open(eval_file, "w")
                f.write("{}".format(model))
                f.close()

            with open(eval_file, "a") as f:
                f.write("{}\n".format(results))

            if checkpoint and bleu_val > cur_best:
                model_file = model_path + "/" + str(epoch + 1) + ".pt"
                torch.save(model, model_file)
