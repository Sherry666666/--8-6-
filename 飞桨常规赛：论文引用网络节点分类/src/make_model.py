# 参考地址：https://github.com/PaddlePaddle/PGL/blob/main/examples/gcn/train.py
# 模型的保存与加载：https://www.paddlepaddle.org.cn/documentation/docs/zh/tutorial/quick_start/save_model/save_model.html
import paddle
import argparse
from preprocess import load
from GCN import GCN
from GAT import GAT
from GCNII import GCNII
from pgl.utils.logger import log
from paddle.optimizer import Adam
def train(node_index, node_label, gnn_model, graph, criterion, optim):
    gnn_model.train()
    pred = gnn_model(graph, graph.node_feat["feat"])
    pred = paddle.gather(pred, node_index)
    loss = criterion(pred, node_label)
    loss.backward()
    acc = paddle.metric.accuracy(input=pred, label=node_label, k=1)
    optim.step()
    optim.clear_grad()
    return loss, acc
@paddle.no_grad()
def eval(node_index, node_label, gnn_model, graph, criterion):
    gnn_model.eval()
    pred = gnn_model(graph, graph.node_feat["feat"])
    pred = paddle.gather(pred, node_index)
    loss = criterion(pred, node_label)
    acc = paddle.metric.accuracy(input=pred, label=node_label, k=1)
    return loss, acc
def main(args):
    # 初始化数组
    dur = []
    best_test = []
    cal_val_acc = []
    cal_test_acc = []
    cal_val_loss = []
    cal_test_loss = []
    # step1. 数据加载
    dataset = load()
    # dataset.graph.tensor()
    graph = dataset.graph
    train_index = dataset.train_index
    train_label = dataset.train_label
    val_index = dataset.valid_index
    val_label = dataset.valid_label
    test_index = dataset.test_index
    # step2. 损失函数配置
    criterion = paddle.nn.loss.CrossEntropyLoss()
    # step3. 模型配置
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
                        hidden_size=64)
    if args.model_name == 'GCNII':
        gnn_model = GCNII(input_size=graph.node_feat["feat"].shape[1],
                        num_class=dataset.num_classes,
                        num_layers=args.num_layers,
                        )
    # step4. 优化方法配置
    optim = Adam(
            learning_rate=args.lr,
            parameters=gnn_model.parameters(),
            weight_decay=0.0005)
    # step4. 模型训练
    gnn_model.train()
    for epoch in range(args.epoch):
        train_loss, train_acc = train(train_index, train_label, gnn_model,
                                          graph, criterion, optim)
        
        val_loss, val_acc = eval(val_index, val_label, gnn_model, graph,
                                    criterion)
        cal_val_acc.append(val_acc.numpy())
        cal_val_loss.append(val_loss.numpy())
        layer_state_dict = gnn_model.state_dict()
        paddle.save(layer_state_dict,"work/GCN.pdparams")
        # paddle.save(gnn_model.weight,"work/GCN.weight.pdtensor")
        log.info(f"epoch{epoch}:Model:val Accuracy:{val_acc.numpy()}   Loss:{val_loss.numpy()}")
        if val_acc.numpy()>0.71:
            print(val_acc.numpy())
            break
    # log.info("Runs %s: Model: GCN Best Test Accuracy: %f" %
    #              (run, cal_test_acc[np.argmin(cal_val_loss)]))
    # best_test.append(cal_test_acc[np.argmin(cal_val_loss)])
    # log.info("Average Speed %s sec/ epoch" % (np.mean(dur)))
    # log.info("Dataset: %s Best Test Accuracy: %f ( stddev: %f )" %
    #          (args.dataset, np.mean(best_test), np.std(best_test)))

        
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
    main(args = args)

