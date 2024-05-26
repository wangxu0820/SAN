import os
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from mixed_gaussian_uniform import *
from options import opt
from data_prep.msda_preprocessed_amazon_dataset import get_msda_amazon_datasets
import pickle
from torch.utils.data import TensorDataset
from data_prep.fdu_mtl_dataset import *
from vocab import Vocab
import utils

def sample_weighting(l_features, l_pseu_labels,features,labels,pseu_labels):
    l_features = l_features.numpy()
    l_pseu_labels = l_pseu_labels.numpy()
    features = features.numpy()
    labels = labels.numpy()
    pseu_labels = pseu_labels.numpy()

    id = np.arange(len(features))
    sort_index = np.argsort(pseu_labels)
    clust_features = features[sort_index]
    clust_pseu_labels = pseu_labels[sort_index]
    clust_labels = labels[sort_index]
    clust_id = id[sort_index]

    l_sort_index = np.argsort(l_pseu_labels)
    l_clust_features = l_features[l_sort_index]
    l_clust_pseu_labels = l_pseu_labels[l_sort_index]

    weighted_id = np.empty([0], dtype=int)
    weighted_pseu_label = np.empty([0], dtype=int)
    weights = np.empty([0])

    for i in range(2):
        class_feature = clust_features[clust_pseu_labels == i]
        class_label = clust_labels[clust_pseu_labels == i]
        class_id = clust_id[clust_pseu_labels == i]
        if len(class_id) == 0:
            continue
        class_mean = np.mean(class_feature, axis=0)  # (192,)
        class_mean = class_mean / (np.linalg.norm(class_mean) + 1e-10)
        l_class_feature = l_clust_features[l_clust_pseu_labels == i]
        # labeled data的类中心
        l_class_mean = np.mean(l_class_feature, axis=0)  # (192,)
        l_class_mean = l_class_mean / (np.linalg.norm(class_mean) + 1e-10)

        R = np.linalg.norm(class_feature, axis=1)
        class_feature_normalized = class_feature / R[:, None]
        class_dist = np.arccos(np.sum(class_feature_normalized * l_class_mean.reshape(-1, 128 + 64), axis=1))
        class_dist = np.nan_to_num(class_dist, nan=1e-10)

        class_dist = class_dist - np.min(class_dist)
        class_dist[2 * np.arange(len(class_dist) // 2)] = -1 * class_dist[2 * np.arange(len(class_dist) // 2)]
        weight, pi = gauss_unif(class_dist.reshape(-1, 1))

        weights = np.hstack((weights, weight))
        weighted_id = np.hstack((weighted_id, class_id))
        weighted_pseu_label = np.hstack((weighted_pseu_label, np.ones_like(class_id, dtype=int) * i))

    return weighted_id, weighted_pseu_label, weights


def make_udata(vocab, domain, id, pseu_label,weights):
    unlabeled_X, pseu_label, weights = read_mtl_file_pl(domain, id, pseu_label, weights)
    unlabeled_set = FduMtlDataset_pl(unlabeled_X, opt.max_seq_len, pseu_label, weights)
    vocab.prepare_inputs(unlabeled_set)
    return unlabeled_set

def make_new_list(vocab, domain):
    # upload unlabeled data
    my_collate = utils.unsorted_collate
    train_sets, dev_set, test_set, unlabeled_sets = get_fdu_mtl_datasets(vocab, opt.fdu_mtl_dir, domain, opt.max_seq_len)
    uset = ConcatDataset([dev_set, test_set, unlabeled_sets])
    unlabeled_loaders = DataLoader(uset, opt.batch_size, shuffle=False, collate_fn=my_collate)
    train_loaders = DataLoader(train_sets, opt.batch_size, shuffle=False, collate_fn=my_collate)

    F_s = torch.load(opt.init_save_file + '/netF_s.pkl')
    F_d = torch.load(opt.init_save_file + '/netF_d.pkl')
    C = torch.load(opt.init_save_file + '/netC.pkl')

    features = torch.Tensor([])
    labels = torch.LongTensor([])
    pseu_labels = torch.LongTensor([])
    with torch.no_grad():
        for data in unlabeled_loaders:
            input = data[0]  # tuple
            label = data[1]  # torch.Size([8])
            shared_feat = F_s(input)
            domain_feat = F_d(input)
            feature = torch.cat((shared_feat, domain_feat), dim=1)
            outputs = C(feature)

            features = torch.cat([features, feature.cpu()], dim=0)
            labels = torch.cat([labels, label.cpu()], dim=0)
            pseu_labels = torch.cat([pseu_labels, torch.argmax(outputs.cpu(), dim=1)], dim=0)

    # label data
    l_features = torch.Tensor([])
    l_pseu_labels = torch.LongTensor([])
    with torch.no_grad():
        for data in train_loaders:
            input = data[0]
            shared_feat = F_s(input)
            domain_feat = F_d(input)
            feature = torch.cat((shared_feat, domain_feat), dim=1)
            outputs = C(feature)
            l_features = torch.cat([l_features, feature.cpu()], dim=0)
            l_pseu_labels = torch.cat([l_pseu_labels, torch.argmax(outputs.cpu(), dim=1)], dim=0)

    weighted_id, weighted_pseu_label, weights = sample_weighting(l_features, l_pseu_labels,features, labels, pseu_labels)
    return make_udata(vocab, domain, weighted_id, weighted_pseu_label, weights)





