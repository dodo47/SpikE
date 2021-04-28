import numpy as np

def get_rel2id(datapath):
    '''
    Convenience function returning a dictionary that turns relation names into ids.
    '''
    file = open('{}/{}'.format(datapath, 'relation_ids.del'))
    content = file.readlines()
    ids = []
    nodenames = []
    for i in range(len(content)):
        a = content[i].split('\t')
        ids.append(int(a[0]))
        nodenames.append(a[-1][:-1])
    rel2id = dict(zip(nodenames, ids))

    return rel2id

def get_id2rel(datapath):
    '''
    Convenience function returning a dictionary that turns ids into relation names.
    '''
    rel2id = get_rel2id(datapath)
    id2rel = dict(zip(rel2id.values(), rel2id.keys()))

    return id2rel

def get_ent2id(datapath):
    '''
    Convenience function returning a dictionary that turns entity names into ids.
    '''
    file = open('{}/{}'.format(datapath, 'entity_ids.del'))
    content = file.readlines()
    ids = []
    nodenames = []
    for i in range(len(content)):
        a = content[i].split('\t')
        ids.append(int(a[0]))
        nodenames.append(a[-1][:-1])
    ent2id = dict(zip(nodenames, ids))

    return ent2id

def get_id2ent(datapath):
    '''
    Convenience function returning a dictionary that turns ids into entity names.
    '''
    ent2id = get_ent2id(datapath)
    id2ent = dict(zip(ent2id.values(), ent2id.keys()))

    return id2ent

def load_data(datapath):
    '''
    Load the data from datapath.

    datapath: path to the data folder

    Output
    training triples, validation triples, negative triples,
    number of nodes, number of relations
    '''
    train_data = np.array(np.loadtxt('{}/{}'.format(datapath, 'train.del')), dtype=int)
    valid_data = np.array(np.loadtxt('{}/{}'.format(datapath, 'valid.del')), dtype=int)

    num_nodes = np.max([np.max(train_data[:,0]), np.max(train_data[:,2])])+1
    num_predicates = np.max(train_data[:,1])+1

    return train_data, valid_data, num_nodes, num_predicates
