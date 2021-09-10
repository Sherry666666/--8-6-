import tqdm
import pgl
import paddle
import paddle.nn as nn
from pgl.utils.logger import log
import numpy as np
import time
import argparse
from paddle.optimizer import Adam
import paddle.nn.functional as F
class GCNII(nn.Layer):
    """Implement of GCNII
    """

    def __init__(self,
                 input_size,
                 num_class,
                 num_layers=1,
                 hidden_size=64,
                 dropout=0.6,
                 lambda_l=0.5,
                 alpha=0.1,
                 k_hop=64,
                 **kwargs):
        super(GCNII, self).__init__()
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.lambda_l = lambda_l
        self.alpha = alpha
        self.k_hop = k_hop

        self.mlps = nn.LayerList()
        self.mlps.append(nn.Linear(input_size, self.hidden_size))
        self.drop_fn = nn.Dropout(self.dropout)
        for _ in range(self.num_layers - 1):
            self.mlps.append(nn.Linear(self.hidden_size, self.hidden_size))

        self.output = nn.Linear(self.hidden_size, num_class)
        self.gcnii = pgl.nn.GCNII(
            hidden_size=self.hidden_size,
            activation="relu",
            lambda_l=self.lambda_l,
            alpha=self.alpha,
            k_hop=self.k_hop,
            dropout=self.dropout)

    def forward(self, graph, feature):
        for m in self.mlps:
            feature = m(feature)
            feature = F.relu(feature)
            feature = self.drop_fn(feature)
        feature = self.gcnii(graph, feature)
        feature = self.output(feature)
        return feature