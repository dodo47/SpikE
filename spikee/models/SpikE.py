import torch
import numpy as np
import matplotlib.pyplot as plt
from spikee.models.neurons.nLIF import nLIF
from spikee.utils.scorer import SYMmetric_score
from spikee.utils.scorer import ASYmmetric_score

class SpikE_Scorer_S:
    def __init__(self, nodes, relations, dim, input_size, tau, maxspan = 2, seed = 1231245):
        '''
        Implementation of the SpikE-S model.
        '''
        torch.manual_seed(seed)
        self.entities = nLIF(nodes, dim, input_size, tau, maxspan, seed)
        self.predicates = torch.nn.Embedding(relations, dim)
        self.num_nodes = nodes

    def score(self, subj, pred, obj):
        '''
        Calculate the score of a list of triples.

        Input: list of subjects, predicates, objects, e.g., [s0, s1, ...], [p0, p1, ...], [o0, o1, ...]
        Output: list of scores
        '''
        s_emb, o_emb = self.entities.embeddings(subj, obj)
        p_emb = self.predicates(torch.tensor(pred).long())

        return SYMmetric_score(s_emb, o_emb, p_emb)

    def update_embeddings(self):
        '''
        Calculate spike embeddings.
        '''
        self.entities.update_embeddings()

    def save(self, savepath, appdix = ''):
        '''
        Save some stuff.
        '''
        np.save('{}/weights_{}.npy'.format(savepath, appdix), self.entities.weights.weight.data.detach().numpy())
        np.save('{}/input_spikes_{}.npy'.format(savepath, appdix), self.entities.input_spikes.detach().numpy())
        pred_embs = self.predicates.weight.data.detach().numpy()
        np.save('{}/predicate_embeddings_{}.npy'.format(savepath, appdix), pred_embs)
        ent_embs = self.entities.get_spike_times().detach().numpy()
        np.save('{}/entity_embeddings_{}.npy'.format(savepath, appdix), ent_embs)

        plt.close()
        for j in range(50):
            plt.vlines(ent_embs[j], j+0.1, (j+1)-0.1)
        plt.savefig('{}/entity_embeddings_{}.png'.format(savepath, appdix))

        plt.close()
        for j in range(len(pred_embs)):
            plt.vlines(pred_embs[j], j+0.1, (j+1)-0.1)
        plt.savefig('{}/predicate_embeddings_{}.png'.format(savepath, appdix))


class SpikE_Scorer_AS(SpikE_Scorer_S):
    def __init__(self, nodes, relations, dim, input_size, tau, maxspan = 2, seed = 1231245):
        '''
        Implementation of the SpikE model.
        '''
        super().__init__(nodes, relations, dim, input_size, tau, maxspan, seed)

    def score(self, subj, pred, obj):
        '''
        Calculate the score of a list of triples.

        Input: list of subjects, predicates, objects, e.g., [s0, s1, ...], [p0, p1, ...], [o0, o1, ...]
        Output: list of scores
        '''
        s_emb, o_emb = self.entities.embeddings(subj, obj)
        p_emb = self.predicates(torch.tensor(pred).long())

        return ASYmmetric_score(s_emb, o_emb, p_emb)
