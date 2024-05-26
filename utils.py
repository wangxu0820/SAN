from collections import defaultdict

import numpy as np
import torch
from torch import autograd
from options import opt


def freeze_net(net):
    if not net:
        return
    for p in net.parameters():
        p.requires_grad = False


def unfreeze_net(net):
    if not net:
        return
    for p in net.parameters():
        p.requires_grad = True


def sorted_collate(batch):
    return my_collate(batch, sort=True)


def unsorted_collate(batch):
    return my_collate(batch, sort=False)

def unsorted_collate_pl(batch):
    return my_collate_pl(batch, sort=False)

def my_collate(batch, sort):
    x, y = zip(*batch)
    # extract input indices
    x = [s['inputs'] for s in x]
    x, y = pad(x, y, opt.eos_idx, sort)
    if torch.cuda.is_available():
        x = (x[0].cuda(0), x[1].cuda(0))
        y = y.cuda(0)
    return (x, y)

def my_collate_pl(batch, sort):
    x, y, gammas = zip(*batch)

    # extract input indices
    x = [s['inputs'] for s in x]
    x, y = pad(x, y, opt.eos_idx, sort)
    gammas = torch.FloatTensor(gammas).view(-1).to(opt.device)
    if torch.cuda.is_available():
        x = (x[0].to(opt.device), x[1].to(opt.device))
        y = y.to(opt.device)
    return (x, y, gammas)

def pad(x, y, eos_idx, sort):
    lengths = [len(row) for row in x]
    max_len = max(lengths)
    # if using CNN, pad to at least the largest kernel size
    if opt.model.lower() == 'cnn':
        max_len = max(max_len, opt.max_kernel_size)
    # pad sequences
    padded_x = np.zeros((len(x), max_len), dtype=np.int)
    padded_x.fill(eos_idx)
    for i, row in enumerate(x):
        assert eos_idx not in row, f'EOS in sequence {row}'
        padded_x[i][:len(row)] = row
    padded_x, lengths = torch.LongTensor(padded_x), torch.LongTensor(lengths)
    y = torch.LongTensor(y).view(-1)
    if sort:
        # sort by length
        sort_len, sort_idx = lengths.sort(0, descending=True)
        padded_x = padded_x.index_select(0, sort_idx)
        y = y.index_select(0, sort_idx)
        return (padded_x, sort_len), y
    else:
        return (padded_x, lengths), y


def calc_gradient_penalty(D, features, onesided=False, interpolate=True):
    feature_vecs = list(features.values())

    if interpolate:
        alpha = torch.rand(feature_vecs[0].size())
        alpha /= torch.sum(alpha, dim=1, keepdim=True)
        alpha = alpha.cuda() if opt.use_cuda else alpha
        interpolates = sum([f*alpha[i] for (i,f) in enumerate(feature_vecs)])
    else:
        feature = torch.cat(feature_vecs, dim=0)
        alpha = torch.rand(len(feature), 1).expand(feature.size())
        noise = torch.rand(feature.size())
        alpha = alpha.cuda() if opt.use_cuda else alpha
        noise = noise.cuda() if opt.use_cuda else noise
        interpolates = alpha*feature + (1-alpha)*(feature+0.5*feature.std()*noise)

    interpolates = interpolates.to(opt.device, require_grad=True)
    disc_interpolates = D(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() \
                                    if opt.use_cuda else torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    if onesided:
        clip_fn = lambda x: x.clamp(min=0)
    else:
        clip_fn = lambda x: x
    gradient_penalty = (clip_fn(gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.gp_lambd
    return gradient_penalty


def calc_orthogality_loss(shared_features, domain_features):
    assert len(shared_features) == len(domain_features)
    loss = None
    for i, sf in enumerate(shared_features):
        df = domain_features[i]
        prod = torch.mm(sf.t(), df)
        loss = torch.sum(prod*prod) + (loss if loss is not None else 0)
    return loss


def average_cv_accuracy(cv):
    """
    cv[fold]['valid'] contains CV accuracy for validation set
    cv[fold]['test'] contains CV accuracy for test set
    """
    avg_acc = {'valid': defaultdict(float), 'test': defaultdict(float)}
    for fold, foldacc in cv.items():
        for dataset, cv_acc in foldacc.items():
            for domain, acc in cv_acc.items():
                avg_acc[dataset][domain] += acc
    for domain in avg_acc['valid']:
        avg_acc['valid'][domain] /= opt.kfold
        avg_acc['test'][domain] /= opt.kfold
    # overall average
    return avg_acc


def endless_get_next_batch(loaders, iters, domain):
    try:
        inputs, targets = next(iters[domain])
    except StopIteration:
        iters[domain] = iter(loaders[domain])
        inputs, targets = next(iters[domain])
    # In PyTorch 0.3, Batch Norm no longer works for size 1 batch,
    # so we will skip leftover batch of size < batch_size 将跳过最后剩余的小于batch_size的一批
    if len(targets) < opt.batch_size:
        return endless_get_next_batch(loaders, iters, domain)
    return (inputs, targets)

def endless_get_next_batch_pl(loaders, iters, domain):
    try:
        inputs_tuple = next(iters[domain])
    except StopIteration:
        iters[domain] = iter(loaders[domain])
        inputs_tuple = next(iters[domain])
    # In PyTorch 0.3, Batch Norm no longer works for size 1 batch,
    # so we will skip leftover batch of size < batch_size
    uninputs, pseu_labels, gammas = inputs_tuple
    if len(pseu_labels) < opt.batch_size:
        return endless_get_next_batch_pl(loaders, iters, domain)
    return inputs_tuple


domain_labels = {}
def get_domain_label(domain, size):
    if (domain, size) in domain_labels:
        return domain_labels[(domain, size)]
    idx = opt.all_domains.index(domain)
    labels = torch.LongTensor(size)
    labels.fill_(idx)
    labels = labels.to(opt.device)
    domain_labels[(domain, size)] = labels
    return labels


random_domain_labels = {}
def get_random_domain_label(loss, size):
    if size in random_domain_labels:
        return random_domain_labels[size]
    labels = torch.FloatTensor(size, len(opt.all_domains))
    labels.fill_(1 / len(opt.all_domains))
    labels = labels.to(opt.device)
    random_domain_labels[size] = labels
    return labels
