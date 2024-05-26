import torch
from torch.utils.data import DataLoader, ConcatDataset
from mixed_gaussian_uniform import *

from options import opt
from data_prep.msda_preprocessed_amazon_dataset import get_msda_amazon_datasets
import pickle
from torch.utils.data import TensorDataset

def sample_weighting(features,labels,pseu_labels):
    features = features.numpy()
    labels = labels.numpy()
    pseu_labels = pseu_labels.numpy()

    id = np.arange(len(features))
    sort_index = np.argsort(pseu_labels)
    clust_features = features[sort_index]
    clust_pseu_labels = pseu_labels[sort_index]
    clust_labels = labels[sort_index]
    clust_id = id[sort_index]

    weighted_id = np.empty([0], dtype=int)
    weighted_pseu_label = np.empty([0], dtype=int)
    weights = np.empty([0])
    for i in range(2):
        class_feature = clust_features[clust_pseu_labels == i]
        class_id = clust_id[clust_pseu_labels == i]
        if len(class_id) == 0:
            continue
        class_mean = np.mean(class_feature, axis=0)
        class_mean = class_mean / (np.linalg.norm(class_mean) + 1e-10)

        R = np.linalg.norm(class_feature, axis=1)
        class_feature_normalized = class_feature / R[:, None]
        class_dist = np.arccos(np.sum(class_feature_normalized * class_mean.reshape(-1, 128 + 64), axis=1))
        class_dist = class_dist - np.min(class_dist)
        class_dist[2 * np.arange(len(class_dist) // 2)] = -1 * class_dist[2 * np.arange(len(class_dist) // 2)]

        weight, pi = gauss_unif(class_dist.reshape(-1, 1))

        weights = np.hstack((weights, weight))
        weighted_id = np.hstack((weighted_id, class_id))
        weighted_pseu_label = np.hstack((weighted_pseu_label, np.ones_like(class_id, dtype=int) * i))

    return weighted_id, weighted_pseu_label, weights

# rewrite dataset['unlabeled']
def make_udata(domain, id,pseu_label,weights):
    dataset = pickle.load(open(opt.prep_amazon_file, 'rb'))[domain]
    ux, uy = dataset['unlabeled']

    if opt.feature_num > 0:
        ux = ux[:, : opt.feature_num]
    ux = torch.from_numpy(ux.toarray()).float().to(opt.device)
    uy = torch.from_numpy(uy).long().to(opt.device)
    id = torch.tensor(id).to(opt.device)
    ux = ux.index_select(0, id)
    pseu_label = torch.tensor(pseu_label).to(opt.device)
    weights = torch.tensor(weights).to(opt.device)
    unlabeled_set = TensorDataset(ux, pseu_label, weights)
    return unlabeled_set

def make_new_list(fold, epoch, domain):
    # upload unlabeled data
    _, unlabeled_sets = get_msda_amazon_datasets(opt.prep_amazon_file, domain, opt.kfold, opt.feature_num)
    unlabeled_loaders = DataLoader(unlabeled_sets, opt.batch_size, shuffle=False)

    if epoch==0:
        F_s = torch.load(opt.init_save_file + '/netF_s_fold0.pkl')
        F_d = torch.load(opt.init_save_file + '/netF_d_fold0.pkl')
        C = torch.load(opt.init_save_file + '/netC_fold0.pkl')
    if epoch>0:
        F_s = torch.load(opt.model_save_file + '/netF_s_fold0.pkl')
        F_d = torch.load(opt.model_save_file + '/netF_d_fold0.pkl')
        C = torch.load(opt.model_save_file + '/netC_fold0.pkl')

    features = torch.Tensor([])
    labels = torch.LongTensor([])
    pseu_labels = torch.LongTensor([])
    with torch.no_grad():
        for data in unlabeled_loaders:
            input = data[0]
            label = data[1]
            input = input.cuda()
            shared_feat = F_s(input)
            domain_feat = F_d(input)
            feature = torch.cat((shared_feat, domain_feat), dim=1)
            outputs = C(feature)

            features = torch.cat([features, feature.cpu()], dim=0)  # 连接feature --> torch.Size([498, 256])
            labels = torch.cat([labels, label.cpu()], dim=0)  # 连接label --> torch.Size([498])
            pseu_labels = torch.cat([pseu_labels, torch.argmax(outputs.cpu(), dim=1)], dim=0)

    weighted_id, weighted_pseu_label, weights= sample_weighting(features, labels, pseu_labels)
    return make_udata(domain, weighted_id, weighted_pseu_label, weights)






