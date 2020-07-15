import math
import torch
import torch.nn as nn
import torchvision

import sys
sys.path.append('.')

from .util import gen_A, gen_adj

class GraphConvolution(nn.Module):
    """ Imported from here: https://github.com/tkipf/pygcn
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCNResnet(nn.Module):
    ''' Paper1: https://res.mdpi.com/d_attachment/futureinternet/futureinternet-11-00245/article_deploy/futureinternet-11-00245.pdf
        Paper2: https://arxiv.org/pdf/1904.03582.pdf
    '''
    __model_factory = {
        'resnet50': torchvision.models.resnet50,
        'resnet101': torchvision.models.resnet101
    }
    def __init__(self, num_classes, backbone, concur, sums, in_channel, threshold):
        super(GCNResnet, self).__init__()
        self.num_classes = num_classes
        self.resnet = self.__model_factory[backbone](pretrained=True)
        self.avgpool = nn.AdaptiveMaxPool2d(1)

        self.A = nn.Parameter(torch.from_numpy(gen_A(concur, sums, threshold=threshold)).float())
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, inp):
        feature = self.resnet.conv1(x)
        feature = self.resnet.bn1(feature)
        feature = self.resnet.relu(feature)
        feature = self.resnet.maxpool(feature)
        feature = self.resnet.layer1(feature)
        feature = self.resnet.layer2(feature)
        feature = self.resnet.layer3(feature)
        feature = self.resnet.layer4(feature)
        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        # feature.size() = (batch_size, 2048)

        # x = x[0]
        # inp.size() = (batch_size, num_classes, in_channel)
        inp = inp[0] # all inp are the same for each batch
        # inp.size() = (num_classes, in_channel)
        adj = gen_adj(self.A).detach()
        # adj.size() = (num_classes, num_classes)
        x = self.gc1(inp, adj)
        # x.size() = (num_classes, 1024)
        x = self.relu(x)
        x = self.gc2(x, adj)
        # x.size() = (num_classes, 2048)
        x = torch.t(x)
        # x.size() = (2048, num_classes)
        x = torch.matmul(feature, x)
        # x.size() = (batch_size, num_classes)
        return x
