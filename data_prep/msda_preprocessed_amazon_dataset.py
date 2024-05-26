import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset  # 对tensor进行打包
from .folded_dataset import FoldedDataset

from options import opt


def get_msda_amazon_datasets(data_file, domain, kfold, feature_num):
    print(f'Loading mSDA Preprocessed Multi-Domain Amazon data for {domain} Domain')
    dataset = pickle.load(open(data_file, 'rb'))[domain]

    lx, ly = dataset['labeled']
    if feature_num > 0:
        lx = lx[:, : feature_num]
    lx = torch.from_numpy(lx.toarray()).float().to(opt.device)
    ly = torch.from_numpy(ly).long().to(opt.device)
    print(f'{domain} Domain has {len(ly)} labeled instances.')
    labeled_set = TensorDataset(lx, ly)
    # only for train_man_exp1:
    if kfold > 1:
        folded_labeled_set = FoldedDataset(kfold, labeled_set)

    ux, uy = dataset['unlabeled']
    if feature_num > 0:
        ux = ux[:, : feature_num]
    ux = torch.from_numpy(ux.toarray()).float().to(opt.device)
    uy = torch.from_numpy(uy).long().to(opt.device)
    print(f'{domain} Domain has {len(uy)} unlabeled instances.')
    # if opt.use_cuda:
    #     ux, uy = ux.cuda(), uy.cuda()
    unlabeled_set = TensorDataset(ux, uy)

    # only for train_man_exp1:
    if kfold == 1:
        return labeled_set, unlabeled_set
    else:
        return folded_labeled_set, unlabeled_set