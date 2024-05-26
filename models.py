import torch
import torch.nn.functional as functional
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.distributions.normal as normal
from options import opt

class CNN(nn.Module):
    def __init__(self,
                 vocab,
                 num_layers,
                 hidden_size,
                 kernel_num,
                 kernel_sizes,
                 dropout):
        super(CNN, self).__init__()
        self.word_emb = vocab.init_embed_layer()
        self.hidden_size = hidden_size
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes

        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, vocab.emb_size)) for K in kernel_sizes])
        
        # at least 1 hidden layer so that the output size is hidden_size
        assert num_layers > 0, 'Invalid layer numbers'
        self.fcnet = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.fcnet.add_module('f-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.fcnet.add_module('f-linear-{}'.format(i),
                        nn.Linear(len(kernel_sizes)*kernel_num, hidden_size))  # nn.Linear(600, 128)
            else:
                self.fcnet.add_module('f-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            # if batch_norm:
            #     self.fcnet.add_module('f-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.fcnet.add_module('f-relu-{}'.format(i), opt.act_unit)

    def forward(self, input):
        data, _ = input
        data = autograd.Variable(data)
        batch_size = len(data)
        embeds = self.word_emb(data)

        # conv
        embeds = embeds.unsqueeze(1)
        x = [functional.relu(conv(embeds)).squeeze(3) for conv in self.convs]
        x = [functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        # fcnet
        return self.fcnet(x)

class SCNN(nn.Module):
    def __init__(self,
                 vocab,
                 num_layers,
                 hidden_size,
                 kernel_num,
                 kernel_sizes,
                 dropout):
        super(SCNN, self).__init__()
        self.word_emb = vocab.init_embed_layer()
        self.hidden_size = hidden_size
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes

        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, vocab.emb_size)) for K in kernel_sizes])

        self.fc_mu = Parameter(torch.zeros(64, 600))
        self.fc_sigma = Parameter(torch.randn(64, 600))
        self.fc_bias = Parameter(torch.zeros(64))

    def forward(self, input):
        data, _ = input
        data = autograd.Variable(data)
        batch_size = len(data)
        embeds = self.word_emb(data)

        # conv
        embeds = embeds.unsqueeze(1)
        x = [functional.relu(conv(embeds)).squeeze(3) for conv in self.convs]
        x = [functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        # fcnet
        fc_sigma_pos = F.softplus(self.fc_sigma - 2)
        fc_distribution = normal.Normal(self.fc_mu, fc_sigma_pos)

        # train
        if self.training:
            fc_w = fc_distribution.rsample()
            return F.linear(x, fc_w, self.fc_bias)

        else:
            return F.linear(x, self.fc_mu, self.fc_bias)

# Mlp FeatureExtractor
class Mlp(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 output_size,
                 dropout,
                 batch_norm=False):
        super(Mlp, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.net = nn.Sequential()
        num_layers = len(hidden_sizes)
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('f-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.net.add_module('f-linear-{}'.format(i), nn.Linear(input_size, hidden_sizes[0]))
            else:
                self.net.add_module('f-linear-{}'.format(i), nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            if batch_norm:
                self.net.add_module('f-bn-{}'.format(i), nn.BatchNorm1d(hidden_sizes[i]))
            self.net.add_module('f-relu-{}'.format(i), opt.act_unit)

        if dropout > 0:
            self.net.add_module('f-dropout-final', nn.Dropout(p=dropout))
        self.net.add_module('f-linear-final', nn.Linear(hidden_sizes[-1], output_size))
        if batch_norm:
            self.net.add_module('f-bn-final', nn.BatchNorm1d(output_size))
        self.net.add_module('f-relu-final', opt.act_unit)

    def forward(self, input):
        return self.net(input)

# Stochastic Mlp FeatureExtractor
class SMlp(nn.Module):
    def __init__(self, dropout, batch_norm=False):
        super(SMlp, self).__init__()
        self.dropout = dropout

        self.fc1 = nn.Linear(5000, 1000)
        self.bn1_fc = nn.BatchNorm1d(1000)

        self.fc2 = nn.Linear(1000, 500)
        self.bn2_fc = nn.BatchNorm1d(500)

        self.fc3_mu = Parameter(torch.zeros(64, 500))
        self.fc3_sigma = Parameter(torch.randn(64, 500))
        self.fc3_bias = Parameter(torch.zeros(64))

    def forward(self, x, only_mu=True):
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = F.relu(self.fc2(x))

        # Distribution sample for the fc layer
        fc3_sigma_pos = F.softplus(self.fc3_sigma - 2)
        fc3_distribution = normal.Normal(self.fc3_mu, fc3_sigma_pos)

        # train
        if self.training:
            fc3_w = fc3_distribution.rsample()
            return F.linear(x, fc3_w, self.fc3_bias)

        # test
        else:
            return F.linear(x, self.fc3_mu, self.fc3_bias)


class SentimentClassifier(nn.Module):
    def __init__(self,
                 num_layers,
                 input_size,
                 hidden_size,
                 output_size,
                 dropout,
                 batch_norm=False):
        super(SentimentClassifier, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.hidden_size = hidden_size
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('p-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(input_size, hidden_size))
            else:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('p-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.net.add_module('p-relu-{}'.format(i), opt.act_unit)

        self.net.add_module('p-linear-final', nn.Linear(hidden_size, output_size))
        self.net.add_module('p-logsoftmax', nn.LogSoftmax(dim=-1))

    def forward(self, input):
        return self.net(input)


class DomainClassifier(nn.Module):
    def __init__(self,
                 num_layers,
                 input_size,
                 hidden_size,
                 num_domains,
                 dropout,
                 batch_norm=False):
        super(DomainClassifier, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.num_domains = num_domains
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('q-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.net.add_module('q-linear-{}'.format(i), nn.Linear(input_size, hidden_size))
            else:
                self.net.add_module('q-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('q-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.net.add_module('q-relu-{}'.format(i), opt.act_unit)

        self.net.add_module('q-linear-final', nn.Linear(hidden_size, num_domains))
        self.net.add_module('q-logsoftmax', nn.LogSoftmax(dim=-1))

    def forward(self, input):
        return self.net(input)
