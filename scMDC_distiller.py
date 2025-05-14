import pandas as pd
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
#from utils import print_log

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

#编码器: 从输入数据中提取特征。
#解码器: 将特征映射回原始输入空间，并生成均值、离散度和概率。
#损失计算: 使用 ZINBLoss 计算模型的损失。
#聚类: 将编码器的输出映射到聚类空间，以进行最终的聚类任务。
class Cluster(nn.Module):
    def __init__(self, input_dim1, n_cluster=10,
            encodeLayer=[], decodeLayer1=[], tau=1., t=10, device="cpu",
            activation="elu", sigma1=2.5, alpha=1., gamma=1., phi1=0.0001, cutoff = 0.5):
        super(Cluster, self).__init__()
        self.tau=tau
        self.input_dim1 = input_dim1

        self.cutoff = cutoff
        self.activation = activation
        self.sigma1 = sigma1

        self.alpha = alpha
        self.gamma = gamma
        self.phi1 = phi1

        self.t = t
        self.device = device
        self.encoder = buildNetwork2([input_dim1]+encodeLayer, type="encode", activation=activation)#创建一个编码器网络，使用 buildNetwork2 函数构建
        self.decoder1 = buildNetwork2(decodeLayer1, type="decode", activation=activation)#self.decoder1：为数据集创建解码器网络。

        self.dec_mean1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), MeanAct())
        self.dec_disp1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), DispAct())

        self.dec_pi1 = nn.Sequential(nn.Linear(decodeLayer1[-1], input_dim1), nn.Sigmoid())

        self.zinb_loss = ZINBLoss()
        self.z_dim = encodeLayer[-1]#编码器输出的维度，即潜在空间的维度。
        #这层的输入是编码器的输出，即 encodeLayer[-1] 维度的向量。encodeLayer[-1] 表示编码器最后一层的输出维度(潜在空间的维度)。
        #这层的输出是一个维度为 n_cluster 的向量，其中 n_cluster 表示要聚类的类别数目。
        #主要目的是将编码器的输出映射到聚类的类别数目上
        self.fc = nn.Linear(encodeLayer[-1], n_cluster)
       # self.log_path = log_path

    def fc_get_pred(self, x):
        out = self.fc(x)
        return out

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def cal_latent(self, z):
        sum_y = torch.sum(torch.square(z), dim=1)
        num = -2.0 * torch.matmul(z, z.t()) + torch.reshape(sum_y, [-1, 1]) + sum_y
        num = num / self.alpha
        num = torch.pow(1.0 + num, -(self.alpha + 1.0) / 2.0)
        zerodiag_num = num - torch.diag(torch.diag(num))
        latent_p = (zerodiag_num.t() / torch.sum(zerodiag_num, dim=1)).t()
        return num, latent_p

    def kmeans_loss(self, z):
        dist1 = self.tau*torch.sum(torch.square(z.unsqueeze(1) - self.mu), dim=2)
        temp_dist1 = dist1 - torch.reshape(torch.mean(dist1, dim=1), [-1, 1])
        q = torch.exp(-temp_dist1)
        q = (q.t() / torch.sum(q, dim=1)).t()
        q = torch.pow(q, 2)
        q = (q.t() / torch.sum(q, dim=1)).t()
        dist2 = dist1 * q
        return dist1, torch.mean(torch.sum(dist2, dim=1))

    def target_distribution(self, q):
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

    def forward(self, x1):
        x = torch.cat([x1+torch.randn_like(x1)*self.sigma1], dim=-1)
        h = self.encoder(x)

        h1 = self.decoder1(h)
        mean1 = self.dec_mean1(h1)
        disp1 = self.dec_disp1(h1)
        pi1 = self.dec_pi1(h1)



        x0 = torch.cat([x1], dim=-1)
        h0 = self.encoder(x0)
        num, lq = self.cal_latent(h0)
        return h0, num, lq, mean1, disp1, pi1,

    #自动编码器（AutoEncoder, AE）模型的前向传播函数。
    def forwardAE(self, x1):
        # 将两个输入数据集 x1 和 x2 沿着最后一个维度（dim=-1）拼接起来。在拼接之前，每个数据集都加上了高斯噪声（torch.randn_like 生成与输入相同形状的随机数），其标准差由 self.sigma1 和 self.sigma2 控制。
        x = torch.cat([x1+torch.randn_like(x1)*self.sigma1], dim=-1)
        h = self.encoder(x)#将拼接和噪声化后的数据 x 通过编码器网络 self.encoder 进行处理，得到编码后的特征表示 h。

        h1 = self.decoder1(h)#将编码后的特征表示 h 通过解码器网络 self.decoder1 进行处理，得到解码后的特征 h1。
        mean1 = self.dec_mean1(h1)#均值
        disp1 = self.dec_disp1(h1)#离散度参数
        pi1 = self.dec_pi1(h1)#dropout概率参数

#处理原始输入数据
#通过将原始数据（x1）传入编码器并得到 h0，模型可以进行一个对比，比较噪声化后的数据与原始数据的编码特征。
#h0 用于计算潜在变量（num）和潜在分布（lq）。这些潜在变量和分布通常是用于进一步的模型分析和损失计算。
        x0 = torch.cat([x1], dim=-1)
        h0 = self.encoder(x0)
        num, lq = self.cal_latent(h0)
        return h0, num, lq, mean1,  disp1,  pi1,

    #定义了一个名为 encodeBatch 的方法，它是一个用于批量编码数据的函数，通常在深度学习模型中用于将输入数据转换到潜在空间。
    def encodeBatch(self, X1,batch_size=256):
        use_cpu = torch.cuda.is_available()
        if use_cpu:
            self.to(self.device)
        encoded = []
        self.eval()#将模型设置为评估模式，这会关闭特定于训练的层，如Dropout。
        num = X1.shape[0]
        num_batch = int(math.ceil(1.0*X1.shape[0]/batch_size))#计算需要多少批次来处理所有数据，使用 math.ceil 函数确保即使不能整除也能完整处理所有数据。
        for batch_idx in range(num_batch):
            x1batch = X1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]

            # inputs1 = Variable(x1batch)
            # inputs2 = Variable(x2batch)
            inputs1 = torch.tensor(x1batch, dtype=torch.float32)

            z,_,_,_,_,_ = self.forwardAE(inputs1)#调用模型的 forwardAE 方法对当前批次的数据进行编码，得到潜在表示 z。其他返回值被忽略。
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)#将所有批次的编码结果合并成一个张量。
        return encoded

    def kldloss(self, p, q):
        c1 = -torch.sum(p * torch.log(q), dim=-1)
        c2 = -torch.sum(p * torch.log(p), dim=-1)
        return torch.mean(c1 - c2)

#自动编码器（autoencoder）模型的预训练过程
#数据准备；前向传播: 计算编码器的输出，并得到重建损失和潜在分布。计算损失: 计算重建损失和 KL 散度损失。
#反向传播和优化: 更新模型参数以最小化损失。
#累积和计算平均损失；
    def pretrain_autoencoder(self, X1, X_raw1, sf1,
            batch_size=256, lr=0.001, epochs=400, ae_save=True, ae_weights='AE_weights.pth.tar'):
            #batch_size=256, lr=0.001, epochs=400, ae_save=True, ae_weights=r'D:\python learning\Pydata\SCMDC\Share\results\AE_weights.pth.tar'):
        num_batch = int(math.ceil(1.0*X1.shape[0]/batch_size))
        dataset = TensorDataset(torch.Tensor(X1), torch.Tensor(X_raw1), torch.Tensor(sf1))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        #print_log("Pretraining stage", self.log_path)
        print("Pretraining stage")
    #计算和打印模型中所有可训练参数的总数
        num_params = 0
        for param in self.parameters():
            if param.requires_grad:
                num_params += param.numel()
        #打印模型中所有可训练参数的总数，并将信息记录到日志文件中 (self.log_path)。
        #print_log('Total number of parameters: %d' % num_params, self.log_path)
        print('Total number of parameters: %d' % num_params)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        num = X1.shape[0]
        for epoch in range(epochs):
            #初始化损失
            loss_val = 0
            recon_loss1_val = 0
            #recon_loss2_val = 0
            kl_loss_val = 0
            for batch_idx, (x1_batch, x_raw1_batch, sf1_batch) in enumerate(dataloader):
                x1_tensor = Variable(x1_batch).to(self.device)
                x_raw1_tensor = Variable(x_raw1_batch).to(self.device)
                sf1_tensor = Variable(sf1_batch).to(self.device)

#调用前向传播函数: self.forwardAE(x1_tensor) 计算编码器的输出，包括潜在特征 zbatch、潜在变量 z_num 和 lqbatch、解码后的均值 mean1_tensor、离散度 disp1_tensor 和 dropout 概率 pi1_tensor。
                zbatch, z_num, lqbatch, mean1_tensor, disp1_tensor, pi1_tensor = self.forwardAE(x1_tensor)
#计算重建损失 recon_loss1。这是自动编码器的主要损失，度量模型重建输入数据的效果。
                recon_loss1 = self.zinb_loss(x=x_raw1_tensor, mean=mean1_tensor, disp=disp1_tensor, pi=pi1_tensor, scale_factor=sf1_tensor)

                lpbatch = self.target_distribution(lqbatch)
                lqbatch = lqbatch + torch.diag(torch.diag(z_num))
                lpbatch = lpbatch + torch.diag(torch.diag(z_num))
# 计算 KL 散度损失 kl_loss，它衡量模型预测的潜在分布与目标分布之间的差异。
                kl_loss = self.kldloss(lpbatch, lqbatch)
                if epoch+1 >= epochs * self.cutoff:
                   loss = recon_loss1  + kl_loss * self.phi1
                else:
                   loss = recon_loss1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val += loss.item() * len(x1_batch)
                recon_loss1_val += recon_loss1.item() * len(x1_batch)

                if epoch+1 >= epochs * self.cutoff:
                    kl_loss_val += kl_loss.item() * len(x1_batch)

            loss_val = loss_val/num
            recon_loss1_val = recon_loss1_val/num

            kl_loss_val = kl_loss_val/num
            if epoch%self.t == 0:
               #print_log('Pretrain epoch {}, Total loss:{:.6f}, ZINB loss1:{:.6f}, ZINB loss2:{:.6f}, KL loss:{:.6f}'.format(epoch+1, loss_val, recon_loss1_val , kl_loss_val), self.log_path)
               print('Pretrain epoch {}, Total loss:{:.6f}, ZINB loss1:{:.6f}, KL loss:{:.6f}'.format(epoch + 1, loss_val, recon_loss1_val, kl_loss_val))

        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, ae_weights)


    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)

#student_output: 学生模型的预测输出；teacher_output: 教师模型的预测输出。labels: 实际的标签（ground truth）。
    #alpha: 权重系数，用于平衡硬损失（hard loss）和软损失（soft loss）。temperature: 温度参数，用于调节softmax函数的平滑度。
    def distillation_loss(self, student_output, teacher_output, labels, alpha=0.5, temperature=2.0):
        # 交叉熵损失
        #硬损失：计算学生模型输出与实际标签之间的交叉熵损失。这部分损失衡量学生模型在实际任务上的表现。
        #软损失：计算学生模型输出和教师模型输出之间的KL散度损失。这个损失衡量的是学生模型如何接近教师模型的“软标签”（即经过温度调节的概率分布）。
        hard_loss = F.cross_entropy(student_output, labels)
        # KL散度损失
        soft_loss = F.kl_div(F.log_softmax(student_output / temperature, dim=1),
                             F.softmax(teacher_output / temperature, dim=1)) * (temperature ** 2)
        # 总损失
        #将硬损失和软损失加权求和。
        total_loss = alpha * hard_loss + (1. - alpha) * soft_loss
        return total_loss

    # 用于训练一个聚类模型。该模型结合了自编码器（autoencoder）和K-means聚类，以对数据进行深度聚类
    # X1 数据集的特征矩阵；X_raw1 数据集的原始特征矩阵；
    # sf1 数据集的大小因子；n_clusters: 聚类数目；
    def fit(self, X1, X_raw1, sf1, y=None, lr=1., n_clusters = 4,
            batch_size=256, num_epochs=10, update_interval=1, tol=1e-3, save_dir="", teacher_model=None):
        '''X: tensor data'''
        use_cpu = torch.cuda.is_available()
        if use_cpu:
            self.to(self.device)
        #print_log("Clustering stage", self.log_path)
        print("Clustering stage")
        X1 = torch.tensor(X1).to(self.device)
        X_raw1 = torch.tensor(X_raw1).to(self.device)
        sf1 = torch.tensor(sf1).to(self.device)

        # 初始化簇中心参数
        # self.mu 是一个 PyTorch Parameter 对象，形状为 (n_clusters, self.z_dim)。requires_grad=True 表示这个参数需要在训练过程中更新。这是K-means的簇中心，n_clusters 是簇的数量，self.z_dim 是潜在空间的维度。
        self.mu = Parameter(torch.Tensor(n_clusters, self.z_dim), requires_grad=True)
        # 设置优化器，用于更新模型的参数。
        # filter(lambda p: p.requires_grad, self.parameters()) 选择所有需要梯度计算的参数（即 requires_grad=True 的参数）。lr=lr 设置学习率，rho=.95 是 Adadelta 优化器的一个超参数。
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)

        #print_log("Initializing cluster centers with kmeans.", self.log_path)
        print("Initializing cluster centers with kmeans.")
       #使用 K-Means 对数据进行初始化，找到初始的聚类中心。
        # 创建K-means模型，n_clusters 是簇的数量，n_init=20 表示K-means将运行20次并选择最佳结果。
        kmeans = KMeans(n_clusters, n_init=20)
        # 使用 encodeBatch 方法对数据进行编码，得到潜在表示 Zdata。，然后使用 K-Means 聚类这些潜在表示。
        Zdata = self.encodeBatch(X1, batch_size=batch_size)


        #latent
        Zdata_numpy = Zdata.data.cpu().numpy()
        # Replace NaNs and Infinities with zero
        Zdata_numpy[np.isnan(Zdata_numpy)] = 0
        Zdata_numpy[np.isinf(Zdata_numpy)] = 0
        # 将潜在表示 Zdata 转换为NumPy数组，然后用K-means对其进行拟合和预测，得到每个样本的聚类标签。
        # self.y_pred = kmeans.fit_predict(Zdata.data.cpu().numpy())
        self.y_pred = kmeans.fit_predict(Zdata_numpy)
        # 保存当前的聚类标签，以便在训练过程中进行比较。
        self.y_pred_last = self.y_pred
        # 将K-means的簇中心拷贝到模型的簇中心参数 self.mu 中。
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))#将 K-Means 找到的聚类中心 kmeans.cluster_centers_ 赋值给模型中的 mu。



        #如果提供了真实标签 y，则计算并打印初始的评估指标（AMI、NMI、ARI）。
        if y is not None:
            ami = np.round(metrics.adjusted_mutual_info_score(y, self.y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
           # print_log('Initializing k-means: AMI= %.4f, NMI= %.4f, ARI= %.4f' % (ami, nmi, ari), self.log_path)
            print('Initializing k-means: AMI= %.4f, NMI= %.4f, ARI= %.4f' % (ami, nmi, ari))

        # 准备训练，设置模型的训练模式，并计算每个训练周期的批次数量。
        self.train() #计算数据集中的样本总数。
        num = X1.shape[0]
        num_batch = int(math.ceil(1.0*X1.shape[0]/batch_size))

        # 初始化用于存储最终聚类评价指标和训练的最终周期数的变量。
        final_ami, final_nmi, final_ari, final_epoch = 0, 0, 0, 0

        # 训练过程中每个 epoch 执行的主要操作部分
        for epoch in range(num_epochs):
            if epoch%update_interval == 0:
                # 教师模型预测的聚类标签：如果有教师模型，则使用教师模型对输入数据 (X1) 进行编码，获取编码后的数据 Zdata_teacher。
                if teacher_model is not None:
                    teacher_model.eval()
                    Zdata_teacher = teacher_model.encodeBatch(X1,  batch_size=batch_size)
                    with torch.no_grad():
                        #获取教师模型的预测结果logits (y_pred_teacher_logits)，这些 logits 表示教师模型对每个样本的类别预测分数。
                        #这些分数可以用于计算教师模型对样本的类别概率或进行进一步的损失计算。
                        #在蒸馏过程中，教师模型的预测 logits被用来计算学生模型的蒸馏损失，来帮助学生模型更好地学习。
                        #Zdata_teacher潜在表示随后被送入教师模型的全连接层（fc_get_pred）以获取教师模型的预测结果y_pred_teacher_logits。这些预测结果用于计算知识蒸馏损失，以指导学生模型学习。
                        y_pred_teacher_logits = teacher_model.fc_get_pred(Zdata_teacher)
                    # dist_teacher, _ = teacher_model.kmeans_loss(Zdata_teacher)
                    # y_pred_teacher = torch.argmin(dist_teacher, dim=1).data.cpu().numpy()

                # 学生模型预测的聚类标签
                #使用学生模型对输入数据进行编码，得到 Zdata。
                Zdata = self.encodeBatch(X1, batch_size=batch_size)
                #计算编码后的数据 Zdata 与聚类中心之间的距离 (dist)
                dist, _ = self.kmeans_loss(Zdata)
                #使用 fc_get_pred 方法获取学生模型的预测 logits (y_pred_student_logits)。
                y_pred_student_logits = self.fc_get_pred(Zdata)
                # y_pred_student = torch.argmin(dist, dim=1).data
                #根据距离 dist，将样本分配到最近的聚类中心，生成学生模型的最终预测结果 self.y_pred。
                self.y_pred = torch.argmin(dist, dim=1).data.cpu().numpy()
                #如果有标签 y，计算并记录 AMI、NMI 和 ARI。
                if y is not None:
                    #acc2 = np.round(cluster_acc(y, self.y_pred), 5)
                    final_ami = ami = np.round(metrics.adjusted_mutual_info_score(y, self.y_pred), 5)
                    final_nmi = nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                    final_ari = ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    final_epoch = epoch+1
                    #print_log('Clustering   %d: AMI= %.4f, NMI= %.4f, ARI= %.4f' % (epoch+1, ami, nmi, ari), self.log_path)
                    print('Clustering   %d: AMI= %.4f, NMI= %.4f, ARI= %.4f' % (epoch + 1, ami, nmi, ari))

                # check stop criterion
                #计算当前聚类结果 (self.y_pred) 与上一个聚类结果 (self.y_pred_last) 之间的变化量。
                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
                self.y_pred_last = self.y_pred
                if epoch>0 and delta_label < tol:
                   # print_log(f'delta_label , {delta_label}, < tol , {tol}', self.log_path)
                   # print_log("Reach tolerance threshold. Stopping training.", self.log_path)
                   print(f'delta_label , {delta_label}, < tol , {tol}')
                   print("Reach tolerance threshold. Stopping training.")
                   break

                #save current model
                # if (epoch>0 and delta_label < tol) or epoch%10 == 0:
                    # self.save_checkpoint({'epoch': epoch+1,
                            # 'state_dict': self.state_dict(),
                            # 'mu': self.mu,
                            # 'y_pred': self.y_pred,
                            # 'y_pred_last': self.y_pred_last,
                            # 'y': y
                            # }, epoch+1, filename=save_dir)

            # train 1 epoch for clustering loss
            loss_val = 0.0
            recon_loss1_val = 0.0

            cluster_loss_val = 0.0
            kl_loss_val = 0.0
            # 核心训练过程。它实现了模型在每个 epoch 的训练步骤，包括数据批处理、计算损失、更新模型参数等。
            # 批处理和数据准备
            # 这里的数据被分批次（batch_size）进行处理，以便处理大规模数据集。num_batch 计算了总共需要多少批次。
            # 使用 batch_idx 和 batch_size 从完整的数据集中切片出当前批次的数据。
            for batch_idx in range(num_batch):
                x1_batch = X1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                x_raw1_batch = X_raw1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                sf1_batch = sf1[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]


                # inputs1 = Variable(x1_batch)
                # rawinputs1 = Variable(x_raw1_batch)
                # sfinputs1 = Variable(sf1_batch)
                # inputs2 = Variable(x2_batch)
                # rawinputs2 = Variable(x_raw2_batch)
                # sfinputs2 = Variable(sf2_batch)

                inputs1 = torch.tensor(x1_batch, dtype=torch.float32)
                rawinputs1 = torch.tensor(x_raw1_batch, dtype=torch.float32)
                sfinputs1 = torch.tensor(sf1_batch, dtype=torch.float32)


                # 前向传播和损失计算
                # 计算当前批次的潜在表示 zbatch 和其他输出。self.forward() 是模型的前向传播方法。
                zbatch, z_num, lqbatch, mean1_tensor, disp1_tensor, pi1_tensor= self.forward(inputs1)
                _, cluster_loss = self.kmeans_loss(zbatch) #计算当前潜在表示 zbatch 与簇中心之间的损失。
                # self.zinb_loss() 计算重构损失，衡量模型重构输入数据的能力。recon_loss1 和 recon_loss2 分别对应数据集 1 和数据集 2。
                recon_loss1 = self.zinb_loss(x=rawinputs1, mean=mean1_tensor, disp=disp1_tensor, pi=pi1_tensor, scale_factor=sfinputs1)

                target2 = self.target_distribution(lqbatch)
                lqbatch = lqbatch + torch.diag(torch.diag(z_num))
                target2 = target2 + torch.diag(torch.diag(z_num))
                kl_loss = self.kldloss(target2, lqbatch)

                # 蒸馏损失
                if teacher_model is not None:
                    #从 y_pred_teacher_logits 和 y_pred_student_logits 中提取当前批次的预测结果。
                    y_pred_teacher_batch = y_pred_teacher_logits[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                    y_pred_student_batch = y_pred_student_logits[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                    label = y[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                    # label = label - 1

                    y_pred_teacher_batch = torch.tensor(y_pred_teacher_batch, dtype=torch.float32).to(self.device)
                    y_pred_student_batch = torch.tensor(y_pred_student_batch, dtype=torch.float32).to(self.device)
                    label = torch.tensor(label, dtype=torch.long).to(self.device)

                    #调用 self.distillation_loss 方法计算教师模型和学生模型之间的知识蒸馏损失 (loss_distiller)。
                    loss_distiller = self.distillation_loss(y_pred_student_batch, y_pred_teacher_batch, label)

                    #如果教师模型存在，将 distillation 损失加到总损失中；否则，只计算重建损失、KL 损失和聚类损失。
                    loss = recon_loss1 + kl_loss + cluster_loss * self.gamma + loss_distiller
                else:
                    loss = recon_loss1 + kl_loss + cluster_loss * self.gamma
                optimizer.zero_grad()
                loss.backward()
#                torch.nn.utils.clip_grad_norm_(self.mu, 1)
                optimizer.step()
                cluster_loss_val += cluster_loss.data * len(inputs1)
                recon_loss1_val += recon_loss1.data * len(inputs1)

                kl_loss_val += kl_loss.data * len(inputs1)
                loss_val = cluster_loss_val + recon_loss1_val  + kl_loss_val

            if epoch%self.t == 0:
               #print_log("#Epoch %d: Total: %.6f Clustering Loss: %.6f ZINB Loss1: %.6f KL Loss: %.6f" % (
                     #epoch + 1, loss_val / num, cluster_loss_val / num, recon_loss1_val / num, kl_loss_val / num), self.log_path)
                print("#Epoch %d: Total: %.6f Clustering Loss: %.6f ZINB Loss1: %.6f KL Loss: %.6f" % (
            epoch + 1, loss_val / num, cluster_loss_val / num, recon_loss1_val / num, kl_loss_val / num))


        #返回最终的预测标签 self.y_pred 和训练结束的最后一个 epoch final_epoch。
        return self.y_pred, final_epoch

