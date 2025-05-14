import os
import sys
import math
import scanpy as sc
from scipy import stats, spatial, sparse
from scipy.linalg import norm
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data
from sklearn.neighbors import kneighbors_graph

import matplotlib.pyplot as plt #加
from munkres import Munkres, print_matrix #加
import seaborn as sns #加

def print_log(log_info, log_path, console=True):
    # print info onto the console
    if console:
        print(log_info)
    # write logs into log file
    if not os.path.exists(log_path):
        fp = open(log_path, "w")
        fp.writelines(log_info + "\n")
    else:
        with open(log_path, 'a+') as f:
            f.writelines(log_info + '\n')

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    #from sklearn.utils.linear_assignment_ import linear_assignment 原代码
    #ind = linear_assignment(w.max() - w) 原代码
    #return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size 原代码

    from scipy.optimize import linear_sum_assignment
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size

def GetCluster(X, res, n):#X 是输入的单细胞数据集
    adata0=sc.AnnData(X)
    if adata0.shape[0]>200000:#如果数据集中的样本数超过20万，执行以下操作以减少数据量。
       np.random.seed(adata0.shape[0])#set seed 
       adata0=adata0[np.random.choice(adata0.shape[0],200000,replace=False)] 
    sc.pp.neighbors(adata0, n_neighbors=n, use_rep="X")#使用 scanpy 的 pp.neighbors 函数计算每个细胞与其 n 个最近邻居的邻接图，使用数据集 X 作为表示。
    sc.tl.louvain(adata0,resolution=res)#根据Louvain算法和给定的分辨率 res 对数据进行聚类分析
    Y_pred_init=adata0.obs['louvain']
    Y_pred_init=np.asarray(Y_pred_init,dtype=int)
    if np.unique(Y_pred_init).shape[0]<=1:
        #avoid only a cluster
        exit("Error: There is only a cluster detected. The resolution:"+str(res)+"is too small, please choose a larger resolution!!")
    else: 
        print("Estimated n_clusters is: ", np.shape(np.unique(Y_pred_init))[0]) 
    return(np.shape(np.unique(Y_pred_init))[0])

def torch_PCA(X, k, center=True, scale=False):
    X = X.t()
    n,p = X.size()
    ones = torch.ones(n).cpu().view([n,1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center else torch.zeros(n*n).view([n,n])
    H = torch.eye(n).cpu() - h
    X_center =  torch.mm(H.double(), X.double())
    covariance = 1/(n-1) * torch.mm(X_center.t(), X_center).view(p,p)
    scaling = torch.sqrt(1/torch.diag(covariance)).double() if scale else torch.ones(p).cpu().double()
    scaled_covariance = torch.mm(torch.diag(scaling).view(p,p), covariance)
    eigenvalues, eigenvectors = torch.eig(scaled_covariance, True)
    components = (eigenvectors[:, :k])
    #explained_variance = eigenvalues[:k, 0]
    return components
    
def best_map(L1,L2):
            #L1 should be the groundtruth labels and L2 should be the clustering labels we got
            Label1 = np.unique(L1)
            nClass1 = len(Label1)
            Label2 = np.unique(L2)
            nClass2 = len(Label2)
            nClass = np.maximum(nClass1,nClass2)
            G = np.zeros((nClass,nClass))
            for i in range(nClass1):
                ind_cla1 = L1 == Label1[i]
                ind_cla1 = ind_cla1.astype(float)
                for j in range(nClass2):
                    ind_cla2 = L2 == Label2[j]
                    ind_cla2 = ind_cla2.astype(float)
                    G[i,j] = np.sum(ind_cla2 * ind_cla1)
            m = Munkres()
            index = m.compute(-G.T)
            index = np.array(index)
            c = index[:,1]
            newL2 = np.zeros(L2.shape)
            for i in range(nClass2):
                newL2[L2 == Label2[i]] = Label1[c[i]]
            return newL2

#用于将聚类标签 (L2) 映射到最优的真实标签 (L1) 上，以最大化标签的一致性。来评估聚类结果与真实标签的匹配情况。
#L1: 真实标签的数组。L2: 聚类算法生成的标签数组。
def best_map_low(L1, L2):
    # L1 should be the groundtruth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)

    # Initialize the cost matrix
    G = np.zeros((nClass, nClass))

    # Populate the cost matrix with the number of overlaps
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)

    # Use the Munkres algorithm to find the best mapping
    m = Munkres()
    # We need to handle the situation where nClass2 > nClass1
    if nClass2 > nClass1:
        padded_G = np.pad(G, ((0, nClass2 - nClass1), (0, 0)), 'constant')
        index = m.compute(-padded_G.T)
    else:
        index = m.compute(-G.T)

    index = np.array(index)
    c = index[:, 1]

    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        if i < nClass1:
            newL2[L2 == Label2[i]] = Label1[c[i]]
        else:
            newL2[L2 == Label2[i]] = Label1[c[i % nClass1]]  # Map extra clusters to existing labels

    return newL2

#根据基因表达数据选择具有高表达水平且检测到的基因，以进行后续分析。
#data：输入的基因表达数据矩阵。每一列对应一个基因，每一行对应一个细胞。threshold：表达量阈值。低于此阈值的表达量被认为是“未检测到”。
#atleast：至少需要被检测到的样本数。低于这个数的基因将被排除。yoffset：在选择基因时用于调整基因表达量的偏移量。
#xoffset:在选择基因时用于调整基因表达量的阈值。decay:决定基因选择阈值的衰减因子。n:要选择的基因数量。
def geneSelection(data, threshold=0, atleast=10, 
                  yoffset=.02, xoffset=5, decay=1.5, n=None, 
                  plot=True, markers=None, genes=None, figsize=(6,3.5),
                  markeroffsets=None, labelsize=10, alpha=1, verbose=1):

    # 数据预处理
    # 如果数据是稀疏矩阵（sparse.issparse(data)），计算每个基因的零率（即该基因在所有样本中未被检测到的比例）。将表达数据转换为对数形式，并计算每个基因的平均表达量。
    if sparse.issparse(data):
        zeroRate = 1 - np.squeeze(np.array((data>threshold).mean(axis=0)))
        A = data.multiply(data>threshold)
        A.data = np.log2(A.data)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.squeeze(np.array(A[:,detected].mean(axis=0))) / (1-zeroRate[detected])
    else:
        # 如果数据是稠密矩阵，计算每个基因的零率和对数平均表达量。
        zeroRate = 1 - np.mean(data>threshold, axis=0)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        mask = data[:,detected]>threshold
        logs = np.zeros_like(data[:,detected]) * np.nan
        logs[mask] = np.log2(data[:,detected][mask])
        meanExpr[detected] = np.nanmean(logs, axis=0)

    # 过滤基因:
    # 对于检测频率低于 atleast 的基因，将其零率和平均表达量设置为nan。
    lowDetection = np.array(np.sum(data>threshold, axis=0)).squeeze() < atleast
    zeroRate[lowDetection] = np.nan
    meanExpr[lowDetection] = np.nan

    # 基因选择:
    # 如果指定了 n（要选择的基因数量），通过二分查找法调整 xoffset 以选择正好 n 个基因。
    # 如果没有指定 n，则根据 xoffset 和 yoffset 选择基因。
    if n is not None:
        up = 10
        low = 0
        for t in range(100):
            nonan = ~np.isnan(zeroRate)
            selected = np.zeros_like(zeroRate).astype(bool)
            selected[nonan] = zeroRate[nonan] > np.exp(-decay*(meanExpr[nonan] - xoffset)) + yoffset
            if np.sum(selected) == n:
                break
            elif np.sum(selected) < n:
                up = xoffset
                xoffset = (xoffset + low)/2
            else:
                low = xoffset
                xoffset = (xoffset + up)/2
        if verbose>0:
            print('Chosen offset: {:.2f}'.format(xoffset))
    else:
        nonan = ~np.isnan(zeroRate)
        selected = np.zeros_like(zeroRate).astype(bool)
        selected[nonan] = zeroRate[nonan] > np.exp(-decay*(meanExpr[nonan] - xoffset)) + yoffset
                
    if plot:
        if figsize is not None:
            plt.figure(figsize=figsize)
        plt.ylim([0, 1])
        if threshold>0:
            plt.xlim([np.log2(threshold), np.ceil(np.nanmax(meanExpr))])
        else:
            plt.xlim([0, np.ceil(np.nanmax(meanExpr))])
        x = np.arange(plt.xlim()[0], plt.xlim()[1]+.1,.1)
        y = np.exp(-decay*(x - xoffset)) + yoffset
        if decay==1:
            plt.text(.4, 0.2, '{} genes selected\ny = exp(-x+{:.2f})+{:.2f}'.format(np.sum(selected),xoffset, yoffset), 
                     color='k', fontsize=labelsize, transform=plt.gca().transAxes)
        else:
            plt.text(.4, 0.2, '{} genes selected\ny = exp(-{:.1f}*(x-{:.2f}))+{:.2f}'.format(np.sum(selected),decay,xoffset, yoffset), 
                     color='k', fontsize=labelsize, transform=plt.gca().transAxes)

        plt.plot(x, y, color=sns.color_palette()[1], linewidth=2)
        xy = np.concatenate((np.concatenate((x[:,None],y[:,None]),axis=1), np.array([[plt.xlim()[1], 1]])))
        t = plt.matplotlib.patches.Polygon(xy, color=sns.color_palette()[1], alpha=.4)
        plt.gca().add_patch(t)
        
        plt.scatter(meanExpr, zeroRate, s=1, alpha=alpha, rasterized=True)
        if threshold==0:
            plt.xlabel('Mean log2 nonzero expression')
            plt.ylabel('Frequency of zero expression')
        else:
            plt.xlabel('Mean log2 nonzero expression')
            plt.ylabel('Frequency of near-zero expression')
        plt.tight_layout()
        
        if markers is not None and genes is not None:
            if markeroffsets is None:
                markeroffsets = [(0, 0) for g in markers]
            for num,g in enumerate(markers):
                i = np.where(genes==g)[0]
                plt.scatter(meanExpr[i], zeroRate[i], s=10, color='k')
                dx, dy = markeroffsets[num]
                plt.text(meanExpr[i]+dx+.1, zeroRate[i]+dy, g, color='k', fontsize=labelsize)
    
    return selected
