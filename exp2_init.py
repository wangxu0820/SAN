from collections import defaultdict
import itertools
import logging
import os
import pickle
import random
import sys
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from torchnet.meter import ConfusionMeter

from options import opt
random.seed(opt.random_seed)
np.random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)
torch.cuda.manual_seed_all(opt.random_seed)

from data_prep.msda_preprocessed_amazon_dataset import get_msda_amazon_datasets
from models import *
import utils

from pseudo_labeling_exp1and2 import make_new_list_exp2

# save models and logging
if not os.path.exists(opt.model_save_file):
    os.makedirs(opt.model_save_file)
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
log = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(opt.model_save_file, 'log.txt'))
log.addHandler(fh)

# output options
log.info(opt)


def train(train_sets, dev_sets, test_sets, unlabeled_sets):
    """
    train_sets, dev_sets, test_sets: dict[domain] -> AmazonDataset
    For unlabeled domains, no train_sets are available
    """
    # dataset loaders
    train_loaders, unlabeled_loaders = {}, {}
    train_iters, unlabeled_iters = {}, {}
    dev_loaders, test_loaders = {}, {}
    for domain in opt.domains:
        train_loaders[domain] = DataLoader(train_sets[domain], opt.batch_size, shuffle=True)
        train_iters[domain] = iter(train_loaders[domain])
    for domain in opt.dev_domains:
        dev_loaders[domain] = DataLoader(dev_sets[domain], opt.batch_size, shuffle=False)
        test_loaders[domain] = DataLoader(test_sets[domain], opt.batch_size, shuffle=False)
    for domain in opt.all_domains:
        if domain in opt.unlabeled_domains:
            uset = unlabeled_sets[domain]
        else:
            # for labeled domains, consider which data to use as unlabeled set
            if opt.unlabeled_data == 'both':
                uset = ConcatDataset([train_sets[domain], unlabeled_sets[domain]])
            elif opt.unlabeled_data == 'unlabeled':
                uset = unlabeled_sets[domain]
            elif opt.unlabeled_data == 'train':
                uset = train_sets[domain]
            else:
                raise Exception(f'Unknown options for the unlabeled data usage: {opt.unlabeled_data}')
        unlabeled_loaders[domain] = DataLoader(uset, opt.batch_size, shuffle=True)
        unlabeled_iters[domain] = iter(unlabeled_loaders[domain])

    # models
    F_s, F_d, C, D = None, None, None, None
    if opt.model.lower() == 'mlp':
        F_s = Mlp(opt.feature_num, opt.F_hidden_sizes, opt.shared_hidden_size, opt.dropout, opt.F_bn)
        F_d = SMlp(opt.dropout, opt.F_bn)
    else:
        raise Exception(f'Unknown model architecture {opt.model}')
    C = SentimentClassifier(opt.C_layers, opt.shared_hidden_size + opt.domain_hidden_size,
            opt.shared_hidden_size + opt.domain_hidden_size, opt.num_labels,
            opt.dropout, opt.C_bn)
    D = DomainClassifier(opt.D_layers, opt.shared_hidden_size, opt.shared_hidden_size,
            len(opt.all_domains), opt.dropout, opt.D_bn)

    F_s, F_d, C, D = F_s.to(opt.device), F_d.to(opt.device), C.to(opt.device), D.to(opt.device)
    params = [p for model in [F_s, C, F_d] if model for p in model.parameters()]
    optimizer = optim.Adam(params, lr=opt.learning_rate)
    optimizerD = optim.Adam(D.parameters(), lr=opt.D_learning_rate)

    # training
    best_acc, best_avg_acc = defaultdict(float), 0.0
    for epoch in range(opt.max_epoch):
        F_s.train()
        C.train()
        D.train()
        F_d.train()
            
        # training accuracy
        correct, total = defaultdict(int), defaultdict(int)
        # D accuracy
        d_correct, d_total = 0, 0
        # conceptually view 1 epoch as 1 epoch of the first domain
        num_iter = len(train_loaders[opt.domains[0]])
        for i in tqdm(range(num_iter)):
            # D iterations
            utils.freeze_net(F_s)
            utils.freeze_net(F_d)
            utils.freeze_net(C)
            utils.unfreeze_net(D)
            # WGAN n_critic trick since D trains slower
            n_critic = opt.n_critic
            if opt.wgan_trick:
                if opt.n_critic>0 and ((epoch==0 and i<25) or i%500==0):
                    n_critic = 100

            for _ in range(n_critic):
                D.zero_grad()
                loss_d = {}
                # train on both labeled and unlabeled domains
                for domain in opt.all_domains:
                    # targets not used
                    d_inputs, _ = utils.endless_get_next_batch(unlabeled_loaders, unlabeled_iters, domain)
                    d_targets = utils.get_domain_label(domain, len(d_inputs))
                    shared_feat = F_s(d_inputs)
                    d_outputs = D(shared_feat)
                    # D accuracy
                    _, pred = torch.max(d_outputs, 1)
                    d_total += len(d_inputs)
                    d_correct += (pred==d_targets).sum().item()
                    if opt.label_smooth is True:
                        # domain label smoothing
                        l_d = crossentropylabelsmooth(d_outputs, d_targets, opt.eps, alpha=0.2, reduction='True')
                    else:
                        l_d = functional.nll_loss(d_outputs, d_targets)
                    l_d.backward()
                    loss_d[domain] = l_d.item()
                optimizerD.step()

            # F&C iteration
            utils.unfreeze_net(F_s)
            utils.unfreeze_net(F_d)
            utils.unfreeze_net(C)
            utils.freeze_net(D)
            F_s.zero_grad()
            F_d.zero_grad()
            C.zero_grad()
            for domain in opt.domains:
                inputs, targets = utils.endless_get_next_batch(train_loaders, train_iters, domain)
                targets = targets.to(opt.device)
                shared_feat = F_s(inputs)
                domain_feat = F_d(inputs)
                features = torch.cat((shared_feat, domain_feat), dim=1)
                c_outputs = C(features)
                l_c = functional.nll_loss(c_outputs, targets)
                l_c.backward(retain_graph=True)
                _, pred = torch.max(c_outputs, 1)
                total[domain] += targets.size(0)
                correct[domain] += (pred == targets).sum().item()
            # update F with D gradients on all domains
            for domain in opt.all_domains:
                d_inputs, _ = utils.endless_get_next_batch(
                        unlabeled_loaders, unlabeled_iters, domain)
                shared_feat = F_s(d_inputs)
                d_outputs = D(shared_feat)
                d_targets = utils.get_domain_label(domain, len(d_inputs))
                l_d = functional.nll_loss(d_outputs, d_targets)
                if opt.lambd > 0:
                    l_d *= -opt.lambd
                l_d.backward()

            optimizer.step()

        # end of epoch
        log.info('Ending epoch {}'.format(epoch+1))
        if d_total > 0:
            log.info('D Training Accuracy: {}%'.format(100.0*d_correct/d_total))
        log.info('Training accuracy:')
        log.info('\t'.join(opt.domains))
        log.info('\t'.join([str(100.0*correct[d]/total[d]) for d in opt.domains]))
        log.info('Evaluating validation sets:')
        acc = {}
        for domain in opt.dev_domains:
            acc[domain] = evaluate(domain, dev_loaders[domain], F_s, F_d, C)
        avg_acc = sum([acc[d] for d in opt.dev_domains]) / len(opt.dev_domains)
        log.info(f'Average validation accuracy: {avg_acc}')
        log.info('Evaluating test sets:')
        test_acc = {}
        for domain in opt.dev_domains:
            test_acc[domain] = evaluate(domain, test_loaders[domain], F_s, F_d, C)
        avg_test_acc = sum([test_acc[d] for d in opt.dev_domains]) / len(opt.dev_domains)
        log.info(f'Average test accuracy: {avg_test_acc}')

        if avg_acc > best_avg_acc:
            log.info(f'New best average validation accuracy: {avg_acc}')
            best_acc['valid'] = acc
            best_acc['test'] = test_acc
            best_avg_acc = avg_acc
            with open(os.path.join(opt.model_save_file, 'options.pkl'), 'wb') as ouf:
                pickle.dump(opt, ouf)
            torch.save(F_s, '{}/netF_s.pkl'.format(opt.init_save_file))
            torch.save(F_d, '{}/netF_d.pkl'.format(opt.init_save_file))
            torch.save(C, '{}/netC.pkl'.format(opt.init_save_file))
            torch.save(D, '{}/netD.pkl'.format(opt.init_save_file))

    # end of training
    log.info(f'Best average validation accuracy: {best_avg_acc}')
    return best_acc

def nll_loss(output,label,weight):
    one_hot_label = torch.zeros(output.size()).scatter_(1, label.cpu().view(-1, 1).long(), 1).cuda()
    nllloss = (-one_hot_label*output).sum(dim=1)
    return torch.sum(nllloss*weight)/(torch.sum(weight)+1e-10)

def crossentropylabelsmooth(y_pred, y_true, eps, alpha=0.2, reduction='None'):
    num_classes = y_pred.size(1)
    if eps >= 0:
        smooth_param = eps
    else:
        # Adaptive label smooth regularization
        soft_label = functional.softmax(y_pred, dim=1)
        smooth_param = alpha * soft_label[torch.arange(soft_label.size(0)), y_true].unsqueeze(1)

    log_probs = y_pred
    with torch.no_grad():
        targets = torch.ones_like(log_probs)
        targets *= smooth_param / (num_classes - 1)
        targets.scatter_(1, y_true.data.unsqueeze(1), (1 - smooth_param))

    loss = (-targets * log_probs).sum(dim=1)

    with torch.no_grad():
        non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)
    if reduction is not None:
        loss = loss.sum() / non_zero_cnt
    return loss

def evaluate(name, loader, F_s, F_d, C):
    F_s.eval()
    F_d.eval()
    C.eval()
    it = iter(loader)
    correct = 0
    total = 0
    confusion = ConfusionMeter(opt.num_labels)
    for inputs, targets in tqdm(it):
        targets = targets.to(opt.device)
        if not F_d:
            # unlabeled domain
            d_features = torch.zeros(len(targets), opt.domain_hidden_size).to(opt.device)
        else:
            d_features = F_d(inputs)
        features = torch.cat((F_s(inputs), d_features), dim=1)
        outputs = C(features)
        _, pred = torch.max(outputs, 1)
        confusion.add(pred.data, targets.data)
        total += targets.size(0)
        correct += (pred == targets).sum().item()
    acc = correct / total
    log.info('{}: Accuracy on {} samples: {}%'.format(name, total, 100.0*acc))
    log.debug(confusion.conf)
    return acc


def main():
    if not os.path.exists(opt.model_save_file):
        os.makedirs(opt.model_save_file)
    datasets = {}
    raw_unlabeled_sets = {}
    log.info(f'Loading {opt.dataset} Datasets...')
    for domain in opt.all_domains:
        datasets[domain], raw_unlabeled_sets[domain] = get_msda_amazon_datasets(
                opt.prep_amazon_file, domain, 1, opt.feature_num)
    opt.num_labels = 2
    log.info(f'Done Loading {opt.dataset} Datasets.')
    log.info(f'Domains: {opt.domains}')

    train_sets, dev_sets, test_sets, unlabeled_sets = {}, {}, {}, {}
    for domain in opt.domains:
        train_sets[domain] = datasets[domain]
        unlabeled_sets[domain] = raw_unlabeled_sets[domain]
    # in this setting, dev_domains should only contain unlabeled domains
    for domain in opt.dev_domains:
        dev_sets[domain] = datasets[domain]
        test_sets[domain] = raw_unlabeled_sets[domain]
        unlabeled_sets[domain] = datasets[domain]

    cv = train(train_sets, dev_sets, test_sets, unlabeled_sets)
    log.info(f'Training done...')
    acc = sum(cv['valid'].values()) / len(cv['valid'])
    log.info(f'Validation Set Domain Average\t{acc}')
    test_acc = sum(cv['test'].values()) / len(cv['test'])
    log.info(f'Test Set Domain Average\t{test_acc}')
    return cv

if __name__ == '__main__':
    main()
