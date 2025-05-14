import time
from tools import prepro, prepare_h5,prepare_nested_h5
from tools import data_process
from moco_sc import MoCo_SC
from scMDC_distiller import Cluster
import random
import yaml
from save import save_model
from evaluation import *

import torch
from torch import nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# 主要功能是加载和合并 YAML 配置文件。
def yaml_config_hook(config_file):
    # config_file: 这是函数的输入参数，指定要加载的主 YAML 配置文件的路径。
    # with open(config_file) as f: 打开主配置文件并将其文件对象赋值给 f。
    # yaml.safe_load(f): 使用 PyYAML 库的 safe_load 函数解析 YAML 文件内容，将其转换为 Python 字典 cfg。
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")

            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg


# 定义了一个 pretrain_epoch 函数，用于在单个训练 epoch 中训练模型。主要功能是处理批次数据、计算损失、执行反向传播并更新模型参数。
def pretrain_epoch(model,
                   data,
                   neighbors,
                   batch_size,
                   criterion,
                   optimizer,
                   device,
                   c=1,
                   flag='aug_nn'):
    loss_epoch = 0.0
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    count = 0

    # model.train()
    print(f"Model Traning Phase: {model.training}")
    for step, pre_index in enumerate(range(data.shape[0] // batch_size + 1)):
        indices_idx = np.arange(pre_index * batch_size, min(data.shape[0], (pre_index + 1) * batch_size))

        if len(indices_idx) < batch_size:
            continue

        count += 1

        batch_indices = indices[indices_idx]
        x = data[batch_indices]
        x = torch.FloatTensor(x).to(device)

        # Use Neighbors as positive instances
        # 如果 neighbors 不为 None，则根据邻接信息选取邻居数据。
        if neighbors is not None:
            batch_nei = neighbors[batch_indices]  # 当前批次的邻接信息
            batch_nei_idx = np.array([np.random.choice(nei, c) for nei in batch_nei])  # 从邻居中随机选择 c 个样本。
            batch_nei_idx = batch_nei_idx.flatten()

            x_nei = data[batch_nei_idx]
            x_nei = torch.FloatTensor(x_nei).to(device)

        assert x_nei.size(0) // x.size(0) == c

        # 'aug_nn': 使用模型将样本和其邻居数据传递，计算moco_sc模型中的对比损失，即计算查询样本和键样本之间的对比损失，以优化模型的特征表示。
        if flag == 'aug_nn':  # Using its augmentation counterpart and neighbor to form positive pairs
            loss = model(x, x_nei, flag=flag)
        else:  # 如果 flag 不是 'aug_nn'，损失通常是 交叉熵损失。
            out1, out2 = model(x, x, flag=flag)
            loss = criterion(out1, out2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step [{step}/{data.shape[0]}]\t loss_instance: {loss.item()}")

        loss_epoch += loss.item()

    loss_epoch = loss_epoch / count

    return loss_epoch


# 主要功能是初始化和训练一个 MoCo_sc 模型。
def pretrain(args, device):
    # data_process: 该函数负责加载和处理数据。它从指定的目录中读取数据，并进行预处理（如缩放、选择特征等）。
    # 函数返回三个对象：data: 处理后的数据特征。_: 可能是处理后的标签或其他信息，但在此处未使用。neighbors: 数据的邻接信息（用于图数据处理或最近邻搜索）。
    data, _, neighbors = data_process(args,args.data_file,
                                      args.num_genes,
                                      k=args.n,
                                      max_element=args.max_element,
                                      scale=False)

    print(f"Data Size: {data.shape}")
    print(f"Neighbors Size: {neighbors.shape}")
    in_features = data.shape[1]
    #从文件中加载数据
    x1, y, cell_type = prepare_nested_h5(args.data_file)
    x1 = np.ceil(x1).astype(np.int32)
    adata1 = sc.AnnData(x1)
    n_vars = adata1.n_vars
    encodeLayer = list(map(int, args.encodeLayer))
    decodeLayer1 = list(map(int, args.decodeLayer1))
    # 初始化MoCo_SC模型
    model = MoCo_SC(args=args,
                    encoder=Cluster,
                    input_size=args.f1, #输入数据的特征数量
                    num_cluster=args.classnum,  # 聚类数
                    encodeLayer=encodeLayer,
                    decodeLayer=decodeLayer1,
                    # latent_features=args.latent_feature,  # 潜在特征维度
                    device=args.device,
                    mlp=True,  # 是否使用多层感知机（MLP）作为投影头。
                    K=args.K,  # K: 队列的大小
                    m=args.m,  # 动量系数（用于更新键编码器）。
                    T=args.T,  # 温度系数
                    p=args.p,
                    lam=args.lam,
                    alpha=args.alpha)

    model = model.to(device)
    # nn.CrossEntropyLoss: 定义交叉熵损失函数，这是用于分类任务的常见损失函数。
    # 它的主要作用是度量模型预测的概率分布与真实标签之间的差距。
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=0.0)

    model_path = os.path.join(args.model_path, f"seed_{args.seed}")
    # 如果 args.reload 为真，加载之前保存的模型和优化器的状态，并从指定的 epoch 继续训练。
    if args.reload:
        model_fp = os.path.join(model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1

    # 模型预训练
    # pretrain
    t0 = time.time()
    for epoch in range(args.start_epoch, args.epochs + 1):
        loss_epoch = pretrain_epoch(model,
                                    data,
                                    neighbors,
                                    args.batch_size,
                                    criterion,
                                    optimizer,
                                    device,
                                    c=args.c,
                                    flag=args.flag)

        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch}")
        print('-' * 60)

        if epoch % 10 == 0:
            model.eval()

    # save_model: 将训练好的模型及其优化器的状态保存到指定的路径，以便以后恢复和使用。
    save_model(model_path, model, optimizer, args.epochs)

    return model, t0,data, neighbors

def save_preprocessed_data(data, neighbors, filename="preprocessed_data.npz"):
    np.savez_compressed(filename, data=data, neighbors=neighbors)
    print(f"Preprocessed data saved to {filename}")
