from sklearn.metrics.pairwise import paired_distances
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from layers import NBLoss, ZINBLoss, MeanAct, DispAct
import numpy as np

import math, os

def buildNetwork2(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):#使用 for 循环遍历 layers 列表中的元素，从索引1开始（即从第二层开始）：
        net.append(nn.Linear(layers[i-1], layers[i]))#对于每一层，使用 nn.Linear 创建一个线性层，将前一层的维度作为输入特征数量，将当前层的维度作为输出特征数量。
        net.append(nn.BatchNorm1d(layers[i], affine=True))#接着，使用 nn.BatchNorm1d 创建一个批归一化层，传入当前层的维度，并将 affine 参数设置为 True 以允许对均值和标准差进行学习。
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="selu":
            net.append(nn.SELU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        elif activation=="elu":
            net.append(nn.ELU())
    return nn.Sequential(*net)

class scMultiClusterLinear(nn.Module):
    def __init__(self, input_dim1, input_dim2, n_cluster=10,
            encodeLayer=[], decodeLayer1=[], decodeLayer2=[], tau=1., t=10, device="cpu",
            activation="elu", sigma1=2.5, sigma2=.1, alpha=1., gamma=1., phi1=0.0001, phi2=0.0001, cutoff = 0.5):
        super(scMultiClusterLinear, self).__init__()
        self.tau=tau
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.cutoff = cutoff
        self.activation = activation
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.alpha = alpha
        self.gamma = gamma
        self.phi1 = phi1
        self.phi2 = phi2
        self.t = t
        self.device = device
        self.encoder = buildNetwork2([input_dim1+input_dim2]+encodeLayer, type="encode", activation=activation)#创建一个编码器网络，使用 buildNetwork2 函数构建
        self.decoder1 = buildNetwork2(decodeLayer1, type="decode", activation=activation)#self.decoder1 和 self.decoder2：分别为数据集1和数据集2创建解码器网络。
        self.decoder2 = buildNetwork2(decodeLayer2, type="decode", activation=activation)
        self.dec_mean1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), MeanAct())
        self.dec_disp1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), DispAct())
        self.dec_mean2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), MeanAct())
        self.dec_disp2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), DispAct())
        self.dec_pi1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), nn.Sigmoid())
        self.dec_pi2 = nn.Sequential(nn.Linear(decodeLayer2[-1], input_dim2), nn.Sigmoid())
        self.zinb_loss = ZINBLoss()
        self.z_dim = encodeLayer[-1]#编码器输出的维度，即潜在空间的维度。

        self.fc = nn.Linear(encodeLayer[-1], n_cluster)

    #自动编码器（AutoEncoder, AE）模型的前向传播函数。
    def forwardAE(self, x1, x2):
        x = torch.cat([x1+torch.randn_like(x1)*self.sigma1, x2+torch.randn_like(x2)*self.sigma2], dim=-1)#将两个输入数据集 x1 和 x2 沿着最后一个维度（dim=-1）拼接起来。在拼接之前，每个数据集都加上了高斯噪声（torch.randn_like 生成与输入相同形状的随机数），其标准差由 self.sigma1 和 self.sigma2 控制。
        h = self.encoder(x)#将拼接和噪声化后的数据 x 通过编码器网络 self.encoder 进行处理，得到编码后的特征表示 h。

        h1 = self.decoder1(h)#将编码后的特征表示 h 通过解码器网络 self.decoder1 进行处理。
        mean1 = self.dec_mean1(h1)#均值
        disp1 = self.dec_disp1(h1)#离散度参数
        pi1 = self.dec_pi1(h1)#dropout概率参数

        h2 = self.decoder2(h)
        mean2 = self.dec_mean2(h2)
        disp2 = self.dec_disp2(h2)
        pi2 = self.dec_pi2(h2)

        x0 = torch.cat([x1, x2], dim=-1)#将原始数据集 x1 和 x2 沿着最后一个维度拼接起来
        h0 = self.encoder(x0)#将拼接后的数据 x0 通过编码器网络 self.encoder 进行处理。
        num, lq = self.cal_latent(h0)
        return h0, num, lq, mean1, mean2, disp1, disp2, pi1, pi2

    def cal_latent(self, z):
        sum_y = torch.sum(torch.square(z), dim=1)
        num = -2.0 * torch.matmul(z, z.t()) + torch.reshape(sum_y, [-1, 1]) + sum_y
        num = num / self.alpha
        num = torch.pow(1.0 + num, -(self.alpha + 1.0) / 2.0)
        zerodiag_num = num - torch.diag(torch.diag(num))
        latent_p = (zerodiag_num.t() / torch.sum(zerodiag_num, dim=1)).t()
        return num, latent_p


    #定义了一个名为 encodeBatch 的方法，它是一个用于批量编码数据的函数，通常在深度学习模型中用于将输入数据转换到潜在空间。
    def encodeBatch(self, X1, X2, batch_size=256):
        use_cpu = torch.cuda.is_available()
        if use_cpu:
            self.to(self.device)
        encoded = []
        self.eval()#将模型设置为评估模式，这会关闭特定于训练的层，如Dropout。
        num = X1.shape[0]
        num_batch = int(math.ceil(1.0*X1.shape[0]/batch_size))#计算需要多少批次来处理所有数据，使用 math.ceil 函数确保即使不能整除也能完整处理所有数据。
        for batch_idx in range(num_batch):
            x1batch = X1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            x2batch = X2[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            # inputs1 = Variable(x1batch)
            # inputs2 = Variable(x2batch)
            inputs1 = torch.tensor(x1batch, dtype=torch.float32)
            inputs2 = torch.tensor(x2batch, dtype=torch.float32)
            z,_,_,_,_,_,_,_,_ = self.forwardAE(inputs1, inputs2)#调用模型的 forwardAE 方法对当前批次的数据进行编码，得到潜在表示 z。其他返回值被忽略。
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)#将所有批次的编码结果合并成一个张量。
        return encoded

    def forward(self, x1, x2):
        encoder_feature = self.encodeBatch(x1, x2)
        out = self.fc(encoder_feature)

        return out
