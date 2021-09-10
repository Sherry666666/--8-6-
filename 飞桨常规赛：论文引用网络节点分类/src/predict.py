import paddle
import argparse
from preprocess import load
from GCN import GCN
from pgl.utils.logger import log
from paddle.optimizer import Adam
import pandas as pd
from GAT import GAT
def predict(args):
    # step1. 数据加载
    dataset = load()
    test_index = dataset.test_index
    graph = dataset.graph
    if args.model_name == 'GCN':
        gnn_model = GCN(input_size=graph.node_feat["feat"].shape[1],
                                num_class=dataset.num_classes,
                                num_layers=args.num_layers,
                                dropout=0.5,
                                hidden_size=128)
    if args.model_name == 'GAT':
        gnn_model = GAT(input_size=graph.node_feat["feat"].shape[1],
                        num_class=dataset.num_classes,
                        num_layers=args.num_layers,
                        feat_drop=0.6,
                        attn_drop=0.6,
                        num_heads=8,
                        hidden_size=8)
    layer_state_dict =  paddle.load("work/GCN.pdparams")                           
    gnn_model.set_state_dict(layer_state_dict) 
    pred = gnn_model(dataset.graph, dataset.graph.node_feat["feat"])
    pred = paddle.gather(pred, test_index)
    pred = paddle.argmax(pred,axis=1)
    df = pd.DataFrame({'nid':paddle.tolist(test_index),'label':paddle.tolist(pred)})
    print(df.head())
    df.to_csv('work/submission.csv',index=None)
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Benchmarking Citation Network')
    parser.add_argument("--model_name", type=str,default='GCN', help="model_name")
    parser.add_argument("--num_layers",type=int,default=1,help="layers number")
    parser.add_argument("--lr",type=float,default=0.01,help="learning_rate")
    parser.add_argument("--epoch", type=int, default=200, help="Epoch")
    parser.add_argument("--runs", type=int, default=10, help="runs")
    parser.add_argument(
        "--feature_pre_normalize",
        type=bool,
        default=True,
        help="pre_normalize feature")
    args = parser.parse_args()
    log.info(args)
    predict(args = args)