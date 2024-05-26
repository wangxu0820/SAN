import os

import torch
from torch.utils.data import Dataset

from options import opt

class FduMtlDataset(Dataset):
    num_labels = 2
    def __init__(self, X, Y, max_seq_len):
        self.X = X
        self.Y = Y
        self.num_labels = 2
        if max_seq_len > 0:
            self.set_max_seq_len(max_seq_len)
        assert len(self.X) == len(self.Y), 'X and Y have different lengths'

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

    def set_max_seq_len(self, max_seq_len):
        for x in self.X:
            x['tokens'] = x['tokens'][:max_seq_len]
        self.max_seq_len = max_seq_len

    def get_max_seq_len(self):
        if not hasattr(self, 'max_seq_len'):
            self.max_seq_len = max([len(x) for x in self.X])
        return self.max_seq_len

class FduMtlDataset_pl(Dataset):
    num_labels = 2
    def __init__(self, X, max_seq_len, pseu_label, weights):
        self.X = X
        self.num_labels = 2
        self.pseu_label = pseu_label
        self.weights = weights
        if max_seq_len > 0:
            self.set_max_seq_len(max_seq_len)
        assert len(self.X) == len(self.pseu_label), 'X and Y have different lengths'

    def __len__(self):
        return len(self.pseu_label)

    def __getitem__(self, idx):
        return (self.X[idx], self.pseu_label[idx], self.weights[idx])

    def set_max_seq_len(self, max_seq_len):
        for x in self.X:
            x['tokens'] = x['tokens'][:max_seq_len]
        self.max_seq_len = max_seq_len

    def get_max_seq_len(self):
        if not hasattr(self, 'max_seq_len'):
            self.max_seq_len = max([len(x) for x in self.X])
        return self.max_seq_len


def read_mtl_file(filename):
    X = []
    Y = []
    with open(filename, 'r', encoding='ISO-8859-2') as inf:
        for line in inf.readlines():
            parts = line.split('\t')
            if len(parts) == 2:
                Y.append(int(parts[0]))
            elif len(parts) == 1:
                Y.append(0)
            else:
                raise Exception('Unknown format')
            words = parts[-1].split(' ')
            X.append({'tokens': words})
    Y = torch.LongTensor(Y).to(opt.device)
    return (X, Y)

def read_mtl_file_pl(domain, id, pseu_label, weights):
    X = []
    id = torch.tensor(id).to(opt.device)

    with open(os.path.join(opt.fdu_mtl_dir, f'{domain}.task.train'), 'r', encoding='ISO-8859-2') as inf:
        for line in inf.readlines():
            parts = line.split('\t')
            words = parts[-1].split(' ')
            X.append({'tokens': words})

    X = X[-200:]
    with open(os.path.join(opt.fdu_mtl_dir, f'{domain}.task.test'), 'r', encoding='ISO-8859-2') as inf:
        for line in inf.readlines():
            parts = line.split('\t')
            words = parts[-1].split(' ')
            X.append({'tokens': words})

    with open(os.path.join(opt.fdu_mtl_dir, f'{domain}.task.unlabel'), 'r', encoding='ISO-8859-2') as inf:
        for line in inf.readlines():
            parts = line.split('\t')
            words = parts[-1].split(' ')
            X.append({'tokens': words})

    X = [X[i] for i in id]

    pseu_label = torch.tensor(pseu_label).to(opt.device)
    weights = torch.tensor(weights).to(opt.device)
    return (X, pseu_label, weights)


def get_fdu_mtl_datasets(vocab, data_dir, domain, max_seq_len):
    print(f'Loading FDU MTL data for {domain} Domain')
    # train and dev set
    train_X, train_Y = read_mtl_file(os.path.join(data_dir, f'{domain}.task.train'))
    dev_X, dev_Y = train_X[-200:], train_Y[-200:]  # 后200作为验证集
    train_X, train_Y = train_X[:-200], train_Y[:-200]  # 剩下的作为训练集
    train_set = FduMtlDataset(train_X, train_Y, max_seq_len)
    dev_set = FduMtlDataset(dev_X, dev_Y, max_seq_len)
    # pre-compute embedding indices
    vocab.prepare_inputs(train_set)
    vocab.prepare_inputs(dev_set)

    # test set
    test_X, test_Y = read_mtl_file(os.path.join(data_dir, f'{domain}.task.test'))
    test_set = FduMtlDataset(test_X, test_Y, max_seq_len)
    vocab.prepare_inputs(test_set)

    # unlabeled set
    unlabeled_X, unlabeled_Y = read_mtl_file(os.path.join(data_dir, f'{domain}.task.unlabel'))
    unlabeled_set = FduMtlDataset(unlabeled_X, unlabeled_Y, max_seq_len)
    vocab.prepare_inputs(unlabeled_set)

    return train_set, dev_set, test_set, unlabeled_set
