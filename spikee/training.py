import torch
import time
import numpy as np
from spikee.utils import evaluate
import os

def train(optimizer, batcher, model, delta, steps, data = None):
    '''
    Training the model. To monitor the training, mean scores for train, valid and negative triples can be shown.

    optimizer: torch optimizer
    batcher: batcher object returning mini-batches
    model: scorer object
    delta: strength of weight regularization pushing the weights above threshold
    steps: number of training steps
    data: list containing train, valid data and negative examples
    '''
    if data is not None:
        train_data, valid_data, neg_data = data[0], data[1], data[2]
    lossf = torch.nn.SoftMarginLoss()

    starttime = time.time()
    for k in range(steps):
        if k%100 == 0:
            estimate = (time.time()-starttime)/(k+1)*(steps-k)/60.
            if data is not None:
                tscore = float(torch.mean(model.score(train_data[:,0], train_data[:,1], train_data[:,2])).detach().numpy())
                vscore = float(torch.mean(model.score(valid_data[:,0], valid_data[:,1], valid_data[:,2])).detach().numpy())
                nscore = float(torch.mean(model.score(neg_data[:,0], neg_data[:,1], neg_data[:,2])).detach().numpy())
                print('SpikE {}: ETA {}min ~~~~~~~~~ train: {} __valid: {} __neg: {}'.format(k, np.round(estimate,2), tscore, vscore, nscore))
            else:
                print('SpikE {}: ETA {}min ~~~~~~~~~'.format(k, np.round(estimate,2)))
        optimizer.zero_grad()
        databatch = batcher.next_batch()

        prediction = model.score(databatch[0], databatch[1], databatch[2])
        weight_reg = model.entities.weight_loss()

        loss = lossf(prediction, torch.tensor(databatch[-1]))
        loss = loss + delta*weight_reg
        loss.backward()

        optimizer.step()
        model.update_embeddings()


def train_and_evaluate(optimizer, batcher, model, delta, steps, eval_points, datapath, data):
    '''
    Training the model. To monitor the training, hits@k, mean rank and MRR are shown.

    optimizer: torch optimizer
    batcher: batcher object returning mini-batches
    model: scorer object
    delta: strength of weight regularization pushing the weights above threshold
    steps: number of training steps
    eval_points: list of steps when metrics should be calculated
    datapath: path to the data
    data: list containing train and validation data
    '''
    train_data, valid_data = data[0], data[1]
    lossf = torch.nn.SoftMarginLoss()

    starttime = time.time()
    for k in range(steps):
        if k in eval_points:
            print('{}:\n'.format(k))
            _, ranks_modes, hitsAt = evaluate.get_rank_sp(model, train_data, 'train', datapath)
            print('train ~~~~SP~~~~ hits@1: {0:.4f} __hits@3: {1:.4f} __ mean: {2:.4f} __ MRR: {3:.4f}\n'.format(hitsAt[0], hitsAt[1], ranks_modes[3], ranks_modes[4]))
            _, ranks_modes, hitsAt = evaluate.get_rank_po(model, train_data, 'train', datapath)
            print('train ~~~~PO~~~~ hits@1: {0:.4f} __hits@3: {1:.4f} __ mean: {2:.4f} __ MRR: {3:.4f}\n'.format(hitsAt[0], hitsAt[1], ranks_modes[3], ranks_modes[4]))
            _, ranks_modes, hitsAt = evaluate.get_rank_sp(model, valid_data, 'valid', datapath)
            print('valid ~~~~SP~~~~ hits@1: {0:.4f} __hits@3: {1:.4f} __ mean: {2:.4f} __ MRR: {3:.4f}\n'.format(hitsAt[0], hitsAt[1], ranks_modes[3], ranks_modes[4]))
            _, ranks_modes, hitsAt = evaluate.get_rank_po(model, valid_data, 'valid', datapath)
            print('valid ~~~~PO~~~~ hits@1: {0:.4f} __hits@3: {1:.4f} __ mean: {2:.4f} __ MRR: {3:.4f}\n'.format(hitsAt[0], hitsAt[1], ranks_modes[3], ranks_modes[4]))

            estimate = (time.time()-starttime)/(k+1)*(steps-k)/60.
            print('ETA {0:.2f}min \n\n'.format(estimate))

        optimizer.zero_grad()
        databatch = batcher.next_batch()

        prediction = model.score(databatch[0], databatch[1], databatch[2])
        weight_reg = model.entities.weight_loss()

        loss = lossf(prediction, torch.tensor(databatch[-1]))
        loss = loss + delta*weight_reg
        loss.backward()

        optimizer.step()
        model.update_embeddings()
