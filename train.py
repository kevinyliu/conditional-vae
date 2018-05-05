import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from tqdm import tqdm
import os
import numpy as np


def train(model, model_name, train_iter, val_iter, SRC_TEXT, TRG_TEXT, anneal, num_epochs=20, gpu=False, lr=0.001, weight_decay=0, checkpoint=False):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=30, factor=0.25, verbose=True, cooldown=6)
    pad = TRG_TEXT.vocab.stoi['<pad>']
    loss = nn.NLLLoss(size_average=True, ignore_index=pad)
    cur_best = 0
    
    for epoch in range(num_epochs):
        model.train()
        
        alpha = anneal(epoch, gpu=gpu)
        train_nre = 0
        train_kl_word = 0
        train_kl_sent = 0
        
        train_mu_dist = 0
        train_p_scale = 0
        train_q_scale = 0
        
        for batch in tqdm(train_iter):
            src, trg = (batch.src.cuda(), batch.trg.cuda()) if gpu else (batch.src, batch.trg)
            
            trg_word_cnt = (trg != pad).float().sum() - trg.size(1)
            
            re, kl, hidden, mu_prior, log_var_prior, mu_posterior, log_var_posterior = model(src, trg)
            
            kl_word = kl.sum() / trg_word_cnt # KL by word
            kl_sent = kl.sum() / len(kl) # KL by sent
            nre = loss(re[:-1, :, :].view(-1, re.size(2)), trg[1:, :].view(-1))
             
            neg_elbo = nre + alpha * kl_word.clamp(0.2)

            train_nre += nre.item()
            train_kl_word += kl_word.item()
            train_kl_sent += kl_sent.item()
            
            train_mu_dist += (mu_prior - mu_posterior).abs().mean().item()
            train_p_scale += log_var_prior.mul(0.5).exp().mean().item()
            train_q_scale += log_var_posterior.mul(0.5).exp().mean().item()

            optimizer.zero_grad()
            neg_elbo.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()
                    
        train_nre /= len(train_iter)
        train_kl_word /= len(train_iter)
        train_kl_sent /= len(train_iter)
        train_mu_dist /= len(train_iter)
        train_p_scale /= len(train_iter)
        train_q_scale /= len(train_iter)
        train_elbo = train_nre + train_kl_word
        train_perp = np.exp(train_elbo)

        val_perp, val_elbo, val_nre, val_kl_word, val_kl_sent, val_mu_dist, val_p_scale, val_q_scale = utils.eval_vae(model, val_iter, pad, gpu)

        # greedy search
        model.if_zero = False
        bleu_greedy = utils.test_multibleu(model, val_iter, TRG_TEXT, k=1, gpu=gpu)
        
        scheduler.step(bleu_greedy)
        #scheduler.step(val_nre)
        
        # greedy search - zeroed out latent vector
        model.if_zero = True
        bleu_zero = utils.test_multibleu(model, val_iter, TRG_TEXT, k=1, gpu=gpu)
        
        results = 'Epoch: {}\n' \
                  '\tVALID PB: {:.4f} NELBO: {:.4f} RE: {:.4f} KL/W: {:.4f} KL/S: {:.4f}\n' \
                  '\tTRAIN PB: {:.4f} NELBO: {:.4f} RE: {:.4f} KL/W: {:.4f} KL/S: {:.4f}\n'\
                  '\tBLEU Greedy: {:.4f}\n\tBLEU Zero Greedy: {:.4f}\n'\
                    '\tVALID MU_DIST: {:.4f} P_SCALE: {:.4f} Q_SCALE: {:.4f}\n'\
                    '\tTRAIN MU_DIST: {:.4f} P_SCALE: {:.4f} Q_SCALE: {:.4f}'\
            .format(epoch+1, val_perp, val_elbo, val_nre, val_kl_word, val_kl_sent,
                    np.exp(train_elbo), train_elbo, train_nre, train_kl_word, train_kl_sent, bleu_greedy, bleu_zero, val_mu_dist, val_p_scale, val_q_scale, train_mu_dist, train_p_scale, train_q_scale)

#        if not (epoch + 1) % 5:
#            model.if_zero = False
#            bleu = utils.test_multibleu(model, val_iter, TRG_TEXT, gpu=gpu)
#            results += '\n\tBLEU: {:.4f}'.format(bleu)

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
                f.write("Number of parameters: " + str(utils.count_parameters(model)) + "\n")
                f.close()

            with open(eval_file, "a") as f:
                f.write("{}\n".format(results))

            if (not (epoch + 1) % 2) and checkpoint and bleu_greedy > cur_best:
                model_file = model_path + "/" + str(epoch + 1) + ".pt"
                torch.save(model, model_file)
                cur_best = bleu_greedy
