import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm


def beam_search(model, src, bos, eos, k=10, max_len=100, filter_token=None, gpu=True):
    """
    works if batch size = 1
    """
    model.eval()
    if gpu: 
        src = src.cuda()

    if filter_token is None:
        filter_token = []

    # run encoder
    encoded_src = model.encode(src)  # batch size = 1

    # init 
    init_prob = 0
    best_options = [(init_prob, [bos], None)]  # beam
    
    for length in range(max_len):  # maximum target length
        options = []  # candidates
        
        for lprob, sentence, hidden in best_options:
            # Prepare last word
            last_word = sentence[-1]

            # keep sentences ending in '</s>' as candidates
            if last_word == eos:
                options.append((lprob, sentence, hidden))

            else:
                last_word_input = torch.tensor([last_word], requires_grad=False).long().view(1, 1)
                if gpu: last_word_input = last_word_input.cuda()

                # Decode
                lprobs, new_hidden = model.generate(last_word_input, src, encoded_src, hidden)
                lprobs = lprobs.squeeze()

                # filter out unwanted tokens such as <pad>
                lprobs[filter_token] = float("-inf")

                # Add top k candidates to options list for next word
                for index in torch.topk(lprobs, k)[1]:
                    option = (lprobs[index].item() + lprob, sentence + [index], new_hidden)
                    options.append(option)

        options.sort(key=lambda x: x[0], reverse=True)  # sort by lprob
        best_options = options[:k]  # place top candidates in beam

    return best_options
