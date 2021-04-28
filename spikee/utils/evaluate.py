import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from spikee.utils.conv import get_ent2id
from spikee.utils.conv import get_rel2id

def score_triple(rel2id, ent2id, model, subj, pred, obj):
    '''
    Score single triple. Takes text name of graph entities as input.
    '''
    subj = torch.tensor([ent2id[subj]])
    obj = torch.tensor([ent2id[obj]])
    pred = torch.tensor([rel2id[pred]])
    return model.score(subj, pred, obj)

def get_rank_sp(model, data, whichone, datapath, savepath = None):
    '''
    Get metrics like mean rank, MRR, hits@k when the object is replaced.

    model: trained model used to evaluate data
    data: triples to be tested
    whichone: which set is used (train, valid)
    datapath: path of data folder
    savepath: if not None, path where results are stored
    '''
    ranks = []
    filtered_sp = np.load('{}/{}_filter_sp.npy'.format(datapath, whichone), allow_pickle=True)
    for i in range(len(data)):
        subj = data[i][0]
        pred = data[i][1]
        rankings = model.score([subj for j in range(len(filtered_sp[i]))], [pred for j in range(len(filtered_sp[i]))], filtered_sp[i]).detach().numpy()
        rank= ss.rankdata(-rankings)[0]
        ranks.append(rank)
    ranks_modes = [np.percentile(ranks, 25), np.median(ranks), np.percentile(ranks, 75), np.mean(ranks), np.mean(1/(np.array(ranks)))]
    hitsAt = []
    for hit in [1, 3, 10, 100, 500]:
        hitsAt.append(np.mean(np.array(ranks) <= hit))
    if savepath is None:
        return ranks, ranks_modes, hitsAt
    else:
        save_ranks(ranks, ranks_modes, hitsAt, savepath, whichone, 'sp')

def get_rank_po(model, data, whichone, datapath, savepath = None):
    '''
    Get metrics like mean rank, MRR, hits@k when the subject is replaced.

    model: trained model used to evaluate data
    data: triples to be tested
    whichone: which set is used (train, valid)
    datapath: path of data folder
    savepath: if not None, path where results are stored
    '''
    ranks = []
    filtered_po = np.load('{}/{}_filter_po.npy'.format(datapath, whichone), allow_pickle=True)
    for i in range(len(filtered_po)):
        pred = data[i][1]
        obj = data[i][2]
        rankings = model.score(filtered_po[i], [pred for j in range(len(filtered_po[i]))], [obj for j in range(len(filtered_po[i]))]).detach().numpy()
        rank= ss.rankdata(-rankings)[0]
        ranks.append(rank)
    ranks_modes = [np.percentile(ranks, 25), np.median(ranks), np.percentile(ranks, 75), np.mean(ranks), np.mean(1/(np.array(ranks)))]
    hitsAt = []
    for hit in [1, 3, 10, 100, 500]:
        hitsAt.append(np.mean(np.array(ranks) <= hit))
    if savepath is None:
        return ranks, ranks_modes, hitsAt
    else:
        save_ranks(ranks, ranks_modes, hitsAt, savepath, whichone, 'po')

def save_ranks(ranks, ranks_modes, hitsAt, savepath, whichone, mode):
    '''
    Save rank metric results.
    '''
    plt.close()
    plt.bar(np.arange(len(hitsAt)+1), hitsAt+[ranks_modes[-1]])
    plt.xticks(np.arange(len(hitsAt)+1), [1, 3, 10, 100, 500, 'MRR'])
    plt.ylim(0,1)
    plt.savefig('{}/{}_{}_hits_and_RMR.png'.format(savepath, whichone, mode))
    np.savetxt('{}/{}_{}_hits.txt'.format(savepath, whichone, mode), [[1, 3, 10, 100, 500], hitsAt])
    np.savetxt('{}/{}_{}_ranking.txt'.format(savepath, whichone, mode), ranks)
    np.savetxt('{}/{}_{}_ranking_modes.txt'.format(savepath, whichone, mode), ranks_modes)

def get_scores(model, train_data, valid_data, neg_data, savepath, post=''):
    '''
    Calculate and save scores of training data, validation data and a set of negative triples.
    '''
    train_scores = model.score(train_data[:,0], train_data[:,1], train_data[:,2]).detach().numpy()
    valid_scores = model.score(valid_data[:,0], valid_data[:,1], valid_data[:,2]).detach().numpy()
    neg_scores = model.score(neg_data[:,0], neg_data[:,1], neg_data[:,2]).detach().numpy()

    plt.close()
    _ = plt.hist(train_scores, bins = 100, alpha = 0.35, color = 'gray', density=True, label = 'train')
    _ = plt.hist(neg_scores, bins = 100, density=True, alpha = 0.7, histtype='step', linewidth = 2.5, label = 'neg')
    _ = plt.hist(valid_scores, bins = 100, density=True, alpha = 0.7, histtype='step', linewidth = 2.5, label = 'pos')
    plt.legend()
    plt.xlim(-10,.1)
    plt.savefig('{}/{}score_histogram.png'.format(savepath, post))
    np.save('{}/{}train_scores.npy'.format(savepath, post), train_scores)
    np.save('{}/{}valid_scores.npy'.format(savepath, post), valid_scores)
    np.save('{}/{}neg_scores.npy'.format(savepath, post), neg_scores)
