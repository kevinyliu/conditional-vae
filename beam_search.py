import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm


class Beam(object):
    """
    Beam objects decodes step by step and keeps the best beam_size viable translations around.
    Important representations:
        `preds' is a list of indices in vocab space for each time step of the current best words
        `prevs' is a list of the indices of the word at the previous time step in the n-gram sequence. The entries are
            in width space, so they are indices in the beam_size array.
    Currently does not allow batching.
    """

    def __init__(self, width, v_size, model, src, bos, eos, gpu=True):
        self.width = width  # beam width
        self.v_size = v_size  # vocab size
        self.prevs = []  # list of previous word indices in this strand, each element is 'width' long
        self.preds = []  # list of previous word predictions
        self.scs = []  # list of cumulative scores
        self.scores = None  # dimension: width (singleton)
        self.model = model
        self.eos = eos
        self.pad = pad

        # run encoder
        self.encoded_src, self.p_normal = self.model.encode(src)

        init_pred = Variable(torch.ones(self.width) * bos)
        if gpu:
            init_pred = init_pred.type_as(torch.cuda.LongTensor())
        else:
            init_pred = init_pred.type_as(torch.LongTensor())

        cur_scores, self.hidden = self.model.generate(init_pred.view(1, -1), self.p_normal, self.encoded_src)

        # TODO: filter function for list of tokens like bellow
        # # filter out pad and end of sentence tokens
        # cur_scores[:, :, pad] = float("-inf")
        # cur_scores[:, :, eos] = float("-inf")

        best_scores, best_scores_id = cur_scores[0, 0, :].squeeze().topk(self.width)
        self.scores = best_scores
        self.preds.append(best_scores_id)
        self.cur_pred = best_scores_id  # dimension after init becomes: width (singleton)
        self.scs.append(self.scores)

    def advance(self):
        # cur_scores dimension is 1 x width x |V|
        # run decoder step to get the softmax over the next word for each word in the beam
        cur_scores, self.hidden = self.model.generate(self.cur_pred.view(1, -1), self.p_normal, self.encoded_src, self.hidden)
        # add to each softmax the score of the sequence it came from
        new_scores = self.scores.unsqueeze(1).expand_as(cur_scores) + cur_scores

        # TODO filter
        # if len(self.preds) <= 3:
        #     new_scores[:, :, self.pad] = float("-inf")
        #     new_scores[:, :, self.eos] = float("-inf")

        # flatten all the softmaxes in one vector
        scores_flat = new_scores.view(-1)
        # get the top, in log likelihood space
        best_scores, best_scores_id = scores_flat.topk(self.width)
        self.scores = best_scores
        self.scs.append(self.scores)
        # get the index at the previous step in the beam that let to the current beam
        prev = best_scores_id / self.v_size  # INDICES in the previous results, ranging from 0 to (width - 1)
        self.prevs.append(prev)
        # take the modulo vocab size to get the index of the words in the current beam
        pred = best_scores_id - prev * self.v_size  # actual word indices in embedding
        self.preds.append(pred)
        self.cur_pred = pred

    def run_search(self, n):
        # TODO run search will do the work of modifying the search so that the stuff that finished is taken out
        for _ in range(n - 1):
            self.advance()

    #         print("preds", torch.stack(self.preds, 1).data.cpu().numpy())
    #         print("prevs", torch.stack(self.preds, 1).data.cpu().numpy())
    #         print("scores", torch.stack(self.scs, 1).data.cpu().numpy())

    # def build_hyp(self, length=3):
    #     # prevs helps lookup the words at the previous step in the beam
    #     lookup = self.prevs[len(self.prevs) - 1]
    #     # hypothesis builds a list of words at each time step
    #     hyp = [self.preds[len(self.prevs)]]
    #     for j in range(len(self.prevs) - 1, -1, -1):
    #         # lookup into the preds at the previous step
    #         hyp.append(self.preds[j][lookup])
    #         # the new lookup is the prevs of the previous step, so you need to lookup in the prevs
    #         lookup = self.prevs[j - 1][lookup]
    #     # reverse the generated hypothesis
    #     hyp = hyp[-length:]
    #     hyp = list(map(lambda l: l.view(-1, 1), hyp[::-1]))
    #     return torch.cat(hyp, 1)

    # def get_hyp(self, length=3):
    #     cur_hyp = self.build_hyp(len(self.preds) - 1, length)
    #
    #     for i in range(length - 1, len(self.preds)):
    #         hyp = self.build_hyp(i, length)
    #         cur_hyp = torch.cat([cur_hyp, hyp], 0)
    #
    #     cur_hyp = cur_hyp.data.cpu().numpy()
    #     indexes = np.unique(cur_hyp, axis=0, return_index=True)[1]
    #     hyp = [cur_hyp[index, :] for index in sorted(indexes)]
    #     return hyp
