import torch
from torch import nn
import torch.nn.functional as F

class MoCo_SC(nn.Module):
    def __init__(self,
                 args,
                 encoder,  # 传入的编码器模型，用于生成特征表示。
                 input_size,  # 输入特征的维度。
                 num_cluster,  # 聚类的数量，通常用于训练中的分类任务或表示。
                 encodeLayer=[256,64,32,16],
                 decodeLayer=[16, 64, 256],
                 # latent_features=[1024, 512, 128],  # 特征维度的列表，表示网络的每一层输出的维度。
                 device="cpu",
                 mlp=True,
                 K=65536,
                 m=0.999,
                 T=0.9,  # 温度参数，用于对比损失计算。
                 p=0.0,
                 lam=0.1,
                 alpha=0.1):
        super().__init__()
        self.K = K  # 4096
        self.m = m
        self.T = T  # 0.3
        self.lam = lam  # 0.4
        self.alpha = alpha
        self.rep_dim = encodeLayer[-1]
        self.device = device

        # self.encoder_q 和 self.encoder_k: 两个编码器网络，
        # self.encoder_q ：查询（query）：用于计算查询样本的特征表示
        # self.encoder_k：键（key）：用于计算键样本的特征表示，键样本是与查询样本进行对比的样本。
        # encoder_k 的权重会通过动量更新来逐步接近 encoder_q 的权重。
        self.encoder_q = encoder(input_dim1=input_size, n_cluster=num_cluster, tau=args.tau,
                            encodeLayer=encodeLayer, decodeLayer1=decodeLayer,
                            activation='elu', sigma1=args.sigma1, gamma=args.gamma,
                            cutoff=args.cutoff, phi1=args.phi1, device=args.device)
        self.encoder_k = encoder(input_dim1=input_size, n_cluster=num_cluster, tau=args.tau,
                            encodeLayer=encodeLayer, decodeLayer1=decodeLayer,
                            activation='elu', sigma1=args.sigma1, gamma=args.gamma,
                            cutoff=args.cutoff, phi1=args.phi1, device=args.device)

        # Projection Head
        # 如果 mlp 为 True，则在编码器后添加一个多层感知机（MLP）头。
        # 这个头由两个线性层、一个批归一化层和一个 ReLU 激活函数组成。
        # 这有助于在特征空间中生成更有效的表示，通常用于提高对比学习的性能。
        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            print(f"dim_mlp: {dim_mlp}")

            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.BatchNorm1d(dim_mlp),
                nn.ReLU(),
                nn.Linear(dim_mlp, dim_mlp)
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.BatchNorm1d(dim_mlp),
                nn.ReLU(),
                nn.Linear(dim_mlp, dim_mlp)
            )

        # 初始化 encoder_k 的权重
        # 将 encoder_q 的权重复制到 encoder_k，并将 encoder_k 的权重设置为不需要梯度。这是为了初始化 encoder_k 的权重，使其与 encoder_q 相同，但在训练过程中不更新 encoder_k 的权重。
        for param_k, param_q in zip(self.encoder_k.parameters(), self.encoder_q.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # 队列: 用于存储负样本的特征向量。这个队列是一个大矩阵（大小为 K x rep_dim），存储着历史样本的特征向量，供当前样本进行对比。初始化时，使用正态分布随机生成，并进行归一化。
        # self.ptr: 队列的指针，用于管理队列中的位置，确保新生成的特征向量可以正确地替换掉旧的特征向量。
        self.register_buffer("queue",
                             F.normalize(torch.randn(self.K, self.rep_dim, requires_grad=False), dim=1))
        self.ptr = 0

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_k, param_q in zip(self.encoder_k.parameters(), self.encoder_q.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)
            param_k.requires_grad = False

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.size(0)

        self.queue[self.ptr: self.ptr + batch_size, :] = keys.detach()
        self.ptr = (self.ptr + batch_size) % self.K
        self.queue.requires_grad = False

    def forward_aug_nn(self, x1, x2):
        latent = self.encoder_q.encodeBatch(x1) #(batch_size, 16)
        q = self.encoder_q.fc_get_pred(latent) # (batch_size, 9)
        q = F.normalize(q, dim=1)

        c = x2.size(0) // x1.size(0)
        qc = q.unsqueeze(1)
        for _ in range(1, c):
            qc = torch.cat([qc, q.unsqueeze(1)], dim=1)
        qc = qc.reshape(-1, q.size(1))

        assert qc.size(0) == x2.size(0)

        with torch.no_grad():
            self._momentum_update_key_encoder()

            latent_k1 = self.encoder_q.encodeBatch(x1)
            k1 = self.encoder_k.fc_get_pred(latent_k1)
            latent_k2 = self.encoder_q.encodeBatch(x2)
            k2 = self.encoder_k.fc_get_pred(latent_k2)

            k1 = F.normalize(k1, dim=1)
            k2 = F.normalize(k2, dim=1)

        # 计算相似度
        pos_sim1 = (1 - self.lam) * torch.einsum("ic, ic -> i", [q, k1]).unsqueeze(-1)
        pos_sim2 = (self.lam / c) * torch.einsum("ic, ic -> i", [qc, k2]).unsqueeze(-1)
        pos_sim2 = pos_sim2.reshape(-1, c)

        assert pos_sim2.size(0) == pos_sim1.size(0)

        pos_sim = torch.cat([pos_sim1, pos_sim2], dim=1)
        neg_sim = torch.einsum("ic, jc -> ij", [q, self.queue.clone().detach()])

        loss = -(torch.logsumexp(pos_sim / self.T, dim=1) - torch.logsumexp(neg_sim / self.T, dim=1)).mean()
        penalty = self.alpha * (torch.mean(torch.abs(latent)))
        loss += penalty

        self._dequeue_and_enqueue(k2)

        return loss

    def forward(self, x1, x2, flag="aug_nn"):
        if flag == 'aug_nn':
            return self.forward_aug_nn(x1, x2)

        latent = self.encoder_q.encodeBatch(x1)
        q = self.encoder_q.fc_get_pred(latent)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()

            latent = self.encoder_q.encodeBatch(x2)
            k = self.encoder_k.fc_get_pred(latent)
            k = F.normalize(k, dim=1)

        pos_sim = torch.einsum("ic, ic -> i", [q, k]).unsqueeze(-1)
        neg_sim = torch.einsum("ic, jc -> ij", [q, self.queue.clone().detach()])

        logits = torch.cat([pos_sim, neg_sim], dim=1) / self.T
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(self.device)

        self._dequeue_and_enqueue(k)

        return logits, labels

    def get_embedding(self, x):
        # out = self.encoder_k.get_embedding(x)
        out = self.encoder_q.encodeBatch(x)

        return out