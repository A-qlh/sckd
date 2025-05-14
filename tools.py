import os
import h5py
import pandas as pd
import numpy as np
import scanpy as sc
import scipy as sp
import hnswlib
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from evaluation import *
import math
import random
from utils import geneSelection
# Some of the following codes are adapted according to the scziDesk Model github
# https://github.com/xuebaliang/scziDesk 


class CellDataset(Dataset):
    def __init__(self, data, target):
        super(CellDataset, self).__init__()
        self.data = torch.FloatTensor(data)  # data：输入数据
        self.target = torch.LongTensor(target)  # target:目标数据

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.target[index]

        return data, label


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn


decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
encode = empty_safe(np.vectorize(lambda _x: str(_x).encode("utf-8")), "S")
upper = empty_safe(np.vectorize(lambda x: str(x).upper()), str)
lower = empty_safe(np.vectorize(lambda x: str(x).lower()), str)
tostr = empty_safe(np.vectorize(str), str)


def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data


def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d

#从 HDF5 文件中读取和处理数据
def read_data(filename, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        #f["obs"], f["var"], 和 f["uns"] 分别代表 HDF5 文件中的数据组。
        obs = pd.DataFrame(dict_from_group(f["obs"]), index = decode(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index = decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])

        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.sparse.csr_matrix((exprs_handle["data"][...], 
                                            exprs_handle["indices"][...],
                                            exprs_handle["indptr"][...]), 
                                            shape = exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.sparse.csr_matrix((obs.shape[0], var.shape[0]))

    return mat, obs, var, uns


def prepro(filename):
    # data_path = os.path.join(filename, "data.h5")
    #调用 read_data 函数从文件中读取数据。这里的 read_data 函数返回了四个对象，分别是数据矩阵 mat，细胞信息 obs，变量信息 var 和其他未使用的信息 uns。
    mat, obs, var, uns = read_data(filename, sparsify=False, skip_exprs=False)

    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())

# 从 obs 中提取细胞类型信息，并将其转换为 NumPy 数组 cell_name。
    cell_name = np.array(obs["cell_type1"])
#获取细胞类型的唯一值 cell_type 及其对应的整数标签 cell_label。cell_label 是 cell_name 中每个细胞类型的数字编码，用于后续的数据处理。
    cell_type, cell_label = np.unique(cell_name, return_inverse=True)
#返回数据矩阵 X，细胞标签 cell_label 和细胞类型的原始名称 cell_name。
    return X, cell_label, cell_name


def normalize(adata, 
              copy=True,
              flavor=None, 
              highly_genes=None, 
              filter_min_counts=True, 
              normalize_input=True, 
              logtrans_input=True,
              scale_input=True,
              size_factors=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    else:
        raise NotImplementedError

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_cells=3)

    if normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if flavor == 'seurat_v3':
        print("seurat_v3")
        sc.pp.highly_variable_genes(adata, flavor=flavor, n_top_genes = highly_genes)

    if normalize_input:
        sc.pp.normalize_total(adata, target_sum=1e4)

    if logtrans_input:
        sc.pp.log1p(adata)

    if flavor is None:
        if highly_genes is not None:
            print("routine hvg")
            sc.pp.highly_variable_genes(adata, n_top_genes=highly_genes)
        else:
            sc.pp.highly_variable_genes(adata)
    
    adata_hvg = adata[:, adata.var.highly_variable].copy()

    if scale_input:
        sc.pp.scale(adata_hvg)

    return adata, adata_hvg

def prepare_nested_h5(file_name):
    X, Y, cell_type = prepro(file_name)

    X = np.ceil(X).astype(np.int32)


    return X, Y, cell_type

def prepare_h5(file_name):
    import h5py
    data_mat = h5py.File(file_name, "r")

    X = np.array(data_mat['X'])
    cell_name = np.array(data_mat['Y'])

    cell_type, Y = np.unique(cell_name, return_inverse=True)

    return X, Y, cell_name

def data_process(args, filename,
                 num_genes,
                 k=6,
                 max_element=95536,
                 scale=True):
    # file_name = os.path.join(root_dir, f"{data_type}_dir", dataset_name)
    print(f"Current Processed Dataset is: {filename}")
#从指定的 HDF5 文件 (filename) 中加载数据。
#X：特征矩阵，其中行表示细胞，列表示基因。Y：细胞的标签。
    X, Y, cell_type = prepare_nested_h5(filename)
    #基因选择
    #如果 args.filter1 为真，调用 geneSelection 函数来选择最重要的基因，并仅保留这些基因的特征。args.f1 指定了选择的重要基因数量。
    if args.filter1:
        importantGenes = geneSelection(X, n=args.f1, plot=False)
        X = X[:, importantGenes]

    adata = sc.AnnData(X, dtype=np.float32)
    adata.obs['Group'] = Y
    adata.obs['annotation'] = cell_type
#数据规范化
#标准化（normalize_input=True）；对数变换（logtrans_input=True）；可选的缩放（scale_input=scale）
    adata, adata_hvg = normalize(adata,
                                 copy=True,
                                 flavor=None,
                                 highly_genes=num_genes,
                                 normalize_input=True,
                                 logtrans_input=True,
                                 scale_input=scale)

    x_array = adata_hvg.to_df().values
    y_array = adata_hvg.obs['Group'].values

    print(f"X shape: {x_array.shape}")
    print(f"Y shape: {y_array.shape}")
#计算邻居
#如果 k 大于 0，调用 cal_nn 函数计算每个细胞的邻居。cal_nn 返回邻居信息。
    if k > 0:
        neighbors, _ = cal_nn(x_array, k=k, max_element=max_element)
    else:
        return x_array, y_array, None

    return x_array, y_array, neighbors


def cal_nn(x, k=500, max_element=95536):
    p = hnswlib.Index(space='cosine', dim=x.shape[1])
    p.init_index(max_elements=max_element,
                 ef_construction=600,
                 random_seed=600,
                 M=100)

    p.set_num_threads(20)
    p.set_ef(600)
    p.add_items(x)

    neighbors, distance = p.knn_query(x, k=k)
    neighbors = neighbors[:, 1:]
    distance = distance[:, 1:]

    return neighbors, distance


def prepareAll(args, filename, num_genes, scale=True):
    # file_name = os.path.join(root_dir, data_type + "_dir", dataset_name)
    print(f"Current Processed Dataset is: {filename}")

    # X 设为数据矩阵，Y 设为组标签，cell_type 设为细胞类型注释。
    X, Y, cell_type = prepare_nested_h5(filename)
    if args.filter1:
        importantGenes = geneSelection(X, n=args.f1, plot=False)
        X = X[:, importantGenes]


    adata = sc.AnnData(X, dtype=np.float32)
    adata.obs['Group'] = Y
    adata.obs['annotation'] = cell_type

    adata, adata_hvg = normalize(adata,
                                 copy=True,
                                 flavor=None,
                                 highly_genes=num_genes,
                                 normalize_input=True,
                                 logtrans_input=True,
                                 scale_input=scale)

    x_ndarray = adata_hvg.to_df().values.squeeze()
    y_ndarray = adata.obs["Group"].values

    print(f'X Shape: {x_ndarray.shape}, Y Shape: {y_ndarray.shape}')

    return x_ndarray, y_ndarray, adata_hvg, cell_type


def inference(args, model, device, type=1):
    model.eval()
    filename = args.data_file
    x, y, adata, cell_type = prepareAll(args=args, filename=filename,
                                        num_genes=args.num_genes,
                                        scale=True)

    val_datasets = CellDataset(x, y)
    in_features = val_datasets.data.size(1)

    print(f"Validation Dataset size: {len(val_datasets)}")
    print(f"The in_features is: {in_features}")

    val_loader = DataLoader(val_datasets,
                            batch_size=256,
                            shuffle=False,
                            num_workers=args.workers,
                            drop_last=False)

    labels_vector = [] #输入数据 x 对应的标签。
    latent_vector = [] #模型对输入数据 x 生成的嵌入向量（或特征向量）。

    for step, (x, y) in enumerate(val_loader):
        x = x.to(device)

        with torch.no_grad():
            latent = model.get_embedding(x) #调用模型的 get_embedding 方法获取数据的嵌入向量（特征表示）。

        latent = latent.detach()

        latent_vector.extend(latent.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 50 == 0:
            print(f"Step [{step}/{len(val_loader)}]\t Computing features...")

    labels_vector = np.array(labels_vector)
    latent_vector = np.array(latent_vector)

    # return labels_vector, latent_vector, cell_type, adata, val_loader
    return labels_vector, latent_vector, cell_type, adata, val_datasets


def get_pseudo_label(args, model, device, type=1):
    # 返回的结果包括 Y（标签）、latent（潜在特征向量）、cell_type细胞的真实类型（标签）、adata 和 val_loader(验证数据的加载器)。
    # Y, latent, cell_type, adata, val_loader, val_datasets = inference(args, model, device)
    Y, latent, cell_type, adata,  val_datasets = inference(args, model, device, type=type)

    print("### Performming Leiden clustering method on latent vector ###")
    # adata_embedding: 包含 Leiden 聚类结果的 AnnData 对象。, leiden_pred: Leiden 聚类的预测结果（每个数据点的簇标签）
    adata_embedding, leiden_pred = run_leiden(latent_vector=latent,
                                              resolution=args.resolution)  # resolution 参数控制聚类的细粒度。较高的分辨率会生成更多的簇。

    print("### Performming KMeans clustering method on latent vector ###")
    # run_kmeans: 使用 KMeans 算法对潜在特征向量进行聚类。args.classnum：指定 KMeans 算法中簇的数量。
    # kmeans_pred：KMeans 聚类的预测结果（每个数据点的簇标签）
    kmeans_pred = run_kmeans(latent, args.classnum, random_state=args.seed)

    # adata_embedding.obs['label']: 将原始标签 Y 添加到 adata_embedding 对象的 .obs 中，并将其转换为类别数据。
    # 将 cell_type 和 adata.obs['annotation'] 添加到 adata_embedding.obs 和 adata.obs 中。
    adata_embedding.obs['label'] = Y
    adata_embedding.obs['label'] = adata_embedding.obs['label'].astype("category")
    # 将真实的细胞类型 cell_type 添加到 adata_embedding 对象的 .obs 中，并将其转换为字符串数组。
    adata_embedding.obs['annotation'] = cell_type
    adata_embedding.obs['annotation'] = np.array(list(map(str, adata.obs['annotation'].values)))

    adata.obs['label'] = Y  # 将原始标签 Y 添加到 adata 对象的 .obs 中，并将其转换为类别数据。
    adata.obs['label'] = adata.obs['label'].astype("category")
    adata_embedding.obs['kmeans'] = kmeans_pred  # 将 KMeans 聚类结果 kmeans_pred 添加到 adata_embedding 对象的 .obs 中，并将其转换为类别数据。
    adata_embedding.obs['kmeans'] = adata_embedding.obs['kmeans'].astype("category")

    adata.obs['kmeans'] = kmeans_pred  # 将 KMeans 聚类结果 kmeans_pred 添加到 adata 对象的 .obs 中，并将其转换为类别数据。
    adata.obs['kmeans'] = adata.obs['kmeans'].astype("category")

    adata.obs['leiden'] = adata_embedding.obs['leiden'].values  # 将 Leiden 聚类结果添加到 adata 对象的 .obs 中。

    # # 将真实标签和潜在特征向量保存为CSV文件
    # np.savetxt(args.save_dir + "/" + str(args.run) + "_true_labels.csv", Y, delimiter=",")
    #
    # np.savetxt(args.save_dir + "/" + str(args.run) + "_latent.csv", latent, delimiter=",")

    # 包含 Leiden 聚类结果的 AnnData 对象。原始数据的 AnnData 对象。原始标签。 Leiden 聚类的预测结果。KMeans 聚类的预测结果。验证数据的加载器。
    # return adata_embedding, adata, Y, leiden_pred, kmeans_pred, val_loader
    return adata_embedding, adata, Y, leiden_pred, kmeans_pred, val_datasets


def get_anchor(adata,
               adata_embedding,  # 数据的嵌入表示（特征表示）
               pseudo_label='leiden',  # 通过leideng聚类算法生成的伪标签
               seed=42,
               k=30,  # 最近邻的数量，用于计算距离。
               percent=0.4,  # 每个类别中选择锚定细胞的比例，即每个类别中选择 40% 的细胞作为锚定细胞。
               max_element=95536):  # 最大元素数量，用于限制计算的邻居数量。

    # 计算最近邻距离
    # 使用 cal_nn 函数计算每个数据点的最近邻距离。cal_nn 返回的是每个点到其 k 个邻居的距离。
    max_element = max(max_element, adata_embedding.shape[0] + 1)
    _, distance = cal_nn(x=adata_embedding.X, k=k, max_element=max_element)

    # 计算每个数据点到其 k 个邻居的平均距离，并将其作为数据点的密度度量
    mean_distance = np.mean(distance, axis=1)

    # 添加距离列: 在 adata 和 adata_embedding 中添加一列 pseudo_label_distance，包含每个细胞的平均距离。
    # 指的是在数据集中增加一个新的列，用于存储每个细胞到其 k 个最近邻的平均距离。这一步骤帮助我们后续根据这些距离信息选择锚定细胞。
    dis_col_name = f"{pseudo_label}_distance"
    adata.obs[dis_col_name] = mean_distance
    adata_embedding.obs[dis_col_name] = mean_distance

    # 选择锚定细胞
    anchor_cells = []
    for c in np.unique(adata_embedding.obs[pseudo_label]):
        cell_type = adata_embedding.obs[adata_embedding.obs[pseudo_label] == c]  # 从嵌入数据中选择该类别的细胞。
        num_cells = math.ceil(percent * cell_type.shape[0])  # 根据 percent 计算要选择的锚定细胞数量。
        threshold = np.sort(cell_type[dis_col_name].values)[num_cells]  # 阈值：根据平均距离排序，选择距离最小的细胞作为锚定细胞。
        cells = cell_type.index[
            np.where(cell_type[dis_col_name] <= threshold)[0]].to_list()  # 如果选择的细胞数量超过了 num_cells，则随机选择一定数量的细胞。

        if len(cells) > num_cells:
            random.seed(seed)
            selected_cells = random.sample(cells, num_cells)
        else:
            selected_cells = cells

        anchor_cells.extend(selected_cells)

    # 标记锚定细胞和非锚定细胞
    non_anchor_cells = list(set(adata.obs_names) - set(anchor_cells))

    adata_embedding.obs.loc[anchor_cells, f"{pseudo_label}_density_status"] = "low"
    adata_embedding.obs.loc[non_anchor_cells, f"{pseudo_label}_density_status"] = "high"

    adata.obs.loc[anchor_cells, f"{pseudo_label}_density_status"] = "low"
    adata.obs.loc[non_anchor_cells, f"{pseudo_label}_density_status"] = "high"

    return adata, adata_embedding


def oversample_cells(adata, pseudo_label='leiden', seed=42):
    sampled_cells = []
    avg_cellnums = math.ceil(adata.shape[0] / len(np.unique(adata.obs[pseudo_label])))

    for c in np.unique(adata.obs[pseudo_label]):
        cell_type = adata.obs[adata.obs[pseudo_label] == c]
        random.seed(seed)

        if cell_type.shape[0] < avg_cellnums:
            selected_cells = random.choices(list(cell_type.index), k=avg_cellnums)
        else:
            selected_cells = list(cell_type.index)

        sampled_cells.extend(selected_cells)

    sampled_adata = adata[sampled_cells]

    return sampled_adata.copy()



class CellDatasetPseudoLabel(Dataset):
    def __init__(self, adata, pseudo_label="kmeans", oversample_flag=True, seed=42):
        super().__init__()
        self.adata = adata
        self.pseudo_label = pseudo_label
        self.oversample_flag = oversample_flag

        # 从 adata 或 oversample_adata 中提取特征数据、伪标签和真实标签，并将它们转换为 PyTorch 张量（FloatTensor 和 LongTensor）。
        if self.oversample_flag:
            self.oversample_adata = oversample_cells(adata=self.adata,
                                                     pseudo_label=self.pseudo_label,
                                                     seed=seed)
            x = self.oversample_adata.X
            pseudo_label = self.oversample_adata.obs[self.pseudo_label]
            pseudo_label = list(map(int, pseudo_label))
            label = self.oversample_adata.obs['label']
        else:
            x = self.adata.X
            pseudo_label = self.adata.obs[self.pseudo_label]
            pseudo_label = list(map(int, pseudo_label))
            label = self.adata.obs['label']

        self.data = torch.FloatTensor(x)
        self.pseu_label = torch.LongTensor(pseudo_label)
        self.true_label = torch.LongTensor(label)

    # 获取单个样本
    def __getitem__(self, index):
        data = self.data[index]  # 从 self.data 中提取特征数据，
        pseu_label = self.pseu_label[index]  # 从 self.pseu_label 中提取伪标签
        true_label = self.true_label[index]  # 从 self.true_label 中提取真实标签。

        return data, pseu_label, true_label

    def __len__(self):
        return len(self.data)

