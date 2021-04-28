import torch
from torch.nn import functional as F

def ASYmmetric_score(s_emb, o_emb, p_emb):
    '''
    TransE score (Bordes et al., 2013).
    '''
    return -F.pairwise_distance(s_emb - o_emb, p_emb, p=1)

def SYMmetric_score(s_emb, o_emb, p_emb):
    '''
    Symmetric version of TransE score.
    '''
    return -F.pairwise_distance((s_emb - o_emb).abs(), p_emb, p=1)
