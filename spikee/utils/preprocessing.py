from spikee.utils import conv
import numpy as np

def filter_data_sp(train_data, valid_data, num_nodes):
    '''
    Filters for objects.

    Given a triple spo, all objects x are removed that
    appear as spx in the training or validation data.
    '''
    filtered_sp = [[] for i in range(len(valid_data))]
    for i in range(len(valid_data)):
        subj = valid_data[i][0]
        pred = valid_data[i][1]
        obj = valid_data[i][2]
        for tr in train_data:
            if tr[0] == subj and tr[1] == pred:
                filtered_sp[i].append(tr[2])
        for tr in valid_data:
            if tr[0] == subj and tr[1] == pred:
                filtered_sp[i].append(tr[2])
        filtered_sp[i] = list(set(range(num_nodes)).difference(set(filtered_sp[i])))
        filtered_sp[i] = [obj] + filtered_sp[i]
    return filtered_sp

def filter_data_po(train_data, valid_data, num_nodes):
    '''
    Filters for subjects.

    Given a triple spo, all subjects x are removed that
    appear as xpo in the training or validation data.
    '''
    filtered_po = [[] for i in range(len(valid_data))]
    for i in range(len(valid_data)):
        subj = valid_data[i][0]
        pred = valid_data[i][1]
        obj = valid_data[i][2]
        for tr in train_data:
            if tr[2] == obj and tr[1] == pred:
                filtered_po[i].append(tr[0])
        for tr in valid_data:
            if tr[2] == obj and tr[1] == pred:
                filtered_po[i].append(tr[0])
        filtered_po[i] = list(set(range(num_nodes)).difference(set(filtered_po[i])))
        filtered_po[i] = [subj] + filtered_po[i]
    return filtered_po

def filter_data(datapath):
    '''
    Create filters for data (neglect known statements that are ranked
    higher during testing than the evaluated triple).
    '''
    train_data, valid_data, num_nodes, _ = conv.load_data(datapath)

    valid_filter_sp = filter_data_sp(train_data, valid_data, num_nodes)
    valid_filter_po = filter_data_po(train_data, valid_data, num_nodes)
    train_filter_sp = filter_data_sp(valid_data, train_data, num_nodes)
    train_filter_po = filter_data_po(valid_data, train_data, num_nodes)

    np.save('{}/valid_filter_sp'.format(datapath), valid_filter_sp)
    np.save('{}/valid_filter_po'.format(datapath), valid_filter_po)
    np.save('{}/train_filter_sp'.format(datapath), train_filter_sp)
    np.save('{}/train_filter_po'.format(datapath), train_filter_po)
