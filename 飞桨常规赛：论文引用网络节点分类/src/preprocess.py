from collections import namedtuple
import pgl
import paddle.fluid as fluid
import numpy as np
import time
import pandas as pd
import paddle

# print("--------图的数据----------")
# edges_data = pd.read_csv('data/data61620/edges.csv',header=None)
# print(edges_data.head())
# print("--------训练数据----------")
# train_data = pd.read_csv('data/data61620/train.csv')
# print(train_data.head())
# print("--------node features----------")
# node_feat = np.load("data/data61620/feat.npy")
# print(node_feat.shape)


# 参考地址：https://aistudio.baidu.com/aistudio/projectdetail/1199841
# PGL文档：https://pgl.readthedocs.io/en/latest/quick_start/instruction.html

from collections import namedtuple
import pgl
import paddle.fluid as fluid
import numpy as np
import time
import pandas as pd

Dataset = namedtuple("Dataset", 
               ["graph", "num_classes", "train_index",
                "train_label", "valid_index", "valid_label", "test_index"])

def load_edges(num_nodes, self_loop=True, add_inverse_edge=True):
    # 从数据中读取边
    edges = pd.read_csv("data/data61620/edges.csv", header=None, names=["src", "dst"]).values

    if add_inverse_edge:
        edges = np.vstack([edges, edges[:, ::-1]])

    # 增加结点自己到自己的边
    if self_loop:
        src = np.arange(0, num_nodes)
        dst = np.arange(0, num_nodes)
        self_loop = np.vstack([src, dst]).T
        edges = np.vstack([edges, self_loop])
    
    return edges
def send_func(src_feat, dst_feat, edge_feat):
    return { "out": src_feat["feature"] }
def recv_func(msg):
    value = msg["out"]
    max_value = msg.reduce_max(value)
    # We want to subscribe the max_value correspond to the destination node.
    max_value = msg.edge_expand(max_value)
    value = value - max_value
    return msg.reduce_sum(value)
def load():
    # 从数据中读取点特征和边，以及数据划分
    node_feat = np.load("data/data61620/feat.npy") # shape:(130644, 100)
    num_nodes = node_feat.shape[0]
    edges = load_edges(num_nodes=num_nodes, self_loop=True, add_inverse_edge=True)
    graph = pgl.graph.Graph(num_nodes=num_nodes, edges=edges, node_feat={"feat": node_feat})  
    indegree = graph.indegree() #indegree,每个结点的度数。shape:(130644,)  graph.indegree(3)表示取编号为3的结点的度数
    norm = np.maximum(indegree.astype("float32"), 1) # 保证每个结点至少有1度
    norm = np.power(norm, -0.5)
    graph.node_feat["norm"] = np.expand_dims(norm, -1) # graph.node_feat["norm"] 的 shape:(130644, 1)
    graph.tensor()
    message = graph.send(send_func, src_feat={"feature": graph.node_feat["feat"]})
    graph.node_feat["send"] = graph.recv(recv_func, message)
    df = pd.read_csv("data/data61620/train.csv")
    node_index = df["nid"].values # 结点的索引
    node_label = df["label"].values # 结点的类型
    train_part = int(len(node_index) * 0.8) # 取训练集80%用作训练，20%用作验证
    train_index = node_index[:train_part]
    train_label = node_label[:train_part]
    valid_index = node_index[train_part:]
    valid_label = node_label[train_part:]
    test_index = pd.read_csv("data/data61620/test.csv")["nid"].values
    dataset = Dataset(graph=graph.tensor(), 
                    train_label=paddle.to_tensor(np.expand_dims(train_label,-1)),
                    train_index=paddle.to_tensor(train_index),
                    valid_index=paddle.to_tensor(valid_index),
                    valid_label=paddle.to_tensor(np.expand_dims(valid_label,-1)),
                    test_index=paddle.to_tensor(test_index), num_classes=35)
    return dataset


if __name__=="__main__":
    node_feat = np.load("data/data61620/feat.npy") # shape:(130644, 100)
    num_nodes = node_feat.shape[0]
    edges = load_edges(num_nodes=num_nodes, self_loop=True, add_inverse_edge=True)
    graph = pgl.graph.Graph(num_nodes=num_nodes, edges=edges, node_feat={"feat": node_feat})
    indegree = graph.indegree()
    norm = np.maximum(indegree.astype("float32"), 1) # 保证每个结点至少有1度
    norm = np.power(norm, -0.5)
    graph.node_feat["norm"] = np.expand_dims(norm, -1)
    print(graph.node_feat["feat"].shape)
    print(graph.node_feat["send"].shape)