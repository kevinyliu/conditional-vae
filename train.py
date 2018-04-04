import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils
from tqdm import tqdm
import os


def train(model, model_name, train_iter, val_iter, SRC_TEXT, TRG_TEXT, num_epochs=20, gpu=False, lr=0.001, weight_decay=0, checkpoint=False):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5, threshold=1e-3)
    loss = nn.NLLLoss(size_average=False)
    for epoch in range(num_epochs):
        model.train()
        train_nll = 0
        train_kl = 0
        for batch in tqdm(train_iter):
            src, trg = (batch.src.cuda(), batch.trg.cuda()) if gpu else (batch.src, batch.trg)

            ll, hidden, mu_prior, log_var_prior, mu_posterior, log_var_posterior = model(src, trg)

            nll = loss(ll[:-1, :, :].view(-1, ll.size(2)), trg[1:, :].view(-1))
            
            kl = torch.sum(0.5 * (((mu_prior - mu_posterior)**2 + torch.exp(log_var_posterior)) / torch.exp(log_var_prior) + (log_var_prior - log_var_posterior) - 1))
            
            neg_elbo = nll + kl

            train_nll += nll.data
            train_kl += kl.data

            optimizer.zero_grad()
            neg_elbo.backward()
            optimizer.step()

        train_nll /= len(train_iter.dataset)
        train_kl /= len(train_iter.dataset)
        train_loss = train_nll + train_kl

        results = 'Epoch: {} Loss: {:.4f} NLL: {:.4f} KL: {:.4f}'.format(epoch+1, train_loss, train_nll, train_kl)
        print(results)

        if not (epoch + 1) % 10:
            local_path = os.getcwd()
            model_path = local_path + "/" + model_name
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            eval_file = model_path + "/" + "eval.txt"

            if epoch == 0:
                f = open(eval_file, "w")
                f.close()

            with open(eval_file, "a") as f:
                f.write("{}: {}\n".format(epoch + 1, results))

            if checkpoint:
                model_file = model_path + "/" + str(epoch + 1) + ".pt"
                torch.save(model, model_file)
