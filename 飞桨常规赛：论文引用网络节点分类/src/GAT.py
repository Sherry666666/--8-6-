import tqdm
import pgl
import paddle
import paddle.nn as nn
from pgl.utils.logger import log
import numpy as np
import time
import argparse
from paddle.optimizer import Adam


class GAT(nn.Layer):
    """Implement of GAT
    """

    def __init__(
            self,
            input_size,
            num_class,
            num_layers=1,
            feat_drop=0.6,
            attn_drop=0.6,
            num_heads=8,
            hidden_size=8, ):
        super(GAT, self).__init__()
        self.num_class = num_class
        self.num_layers = num_layers
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.gats = nn.LayerList()
        for i in range(self.num_layers):
            if i == 0:
                self.gats.append(
                    pgl.nn.GATConv(
                        input_size,
                        self.hidden_size,
                        self.feat_drop,
                        self.attn_drop,
                        self.num_heads,
                        activation='elu'))
            elif i == (self.num_layers - 1):
                self.gats.append(
                    pgl.nn.GATConv(
                        self.num_heads * self.hidden_size,
                        self.num_class,
                        self.feat_drop,
                        self.attn_drop,
                        1,
                        concat=False,
                        activation=None))
            else:
                self.gats.append(
                    pgl.nn.GATConv(
                        self.num_heads * self.hidden_size,
                        self.hidden_size,
                        self.feat_drop,
                        self.attn_drop,
                        self.num_heads,
                        activation='elu'))

    def forward(self, graph, feature):
        for m in self.gats:
            feature = m(graph, feature)
        return feature