from time import time
import math, os
from sklearn import metrics
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from scMDC_distiller import Cluster
from tools import *
import warnings
import pandas as pd
import numpy as np
import collections
import h5py
import scanpy as sc
from preprocess import read_dataset, normalize, clr_normalize_each_cell
from utils import *
from pretrain import pretrain, set_seed

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=-1, type=int)
    parser.add_argument('--cutoff', default=0.4, type=float,
                        help='Start to train combined layer after what ratio of epoch')
    parser.add_argument('--batch_size', default=256, type=int)
    # parser.add_argument('--data_file', default='Normalized_filtered_BMNC_GSE128639_Seurat.h5')
    parser.add_argument('--data_file1',
                        default="Plasschaert/data.h5")

    parser.add_argument('--maxiter', default=100, type=int)
    parser.add_argument('--pretrain_epochs', default=80, type=int)
    parser.add_argument('--gamma', default=.1, type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--tau', default=1., type=float,
                        help='fuzziness of clustering loss')
    parser.add_argument('--phi1', default=0.001, type=float,
                        help='coefficient of KL loss in pretraining stage')
    # parser.add_argument('--phi2', default=0.001, type=float,
    # help='coefficient of KL loss in clustering stage')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--lr', default=0.9, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results')
    # parser.add_argument('--ae_weight_file', default='AE_weights_1.pth.tar')
    parser.add_argument('--ae_weight_file', default='./model_state_dict/AE_weights_GSE128639.pth.tar')
    # parser.add_argument('--resolution', default=0.2, type=float)
    parser.add_argument('--n_neighbors', default=20, type=int)
    parser.add_argument('--embedding_file', action='store_true', default=True)
    parser.add_argument('--prediction_file', action='store_true', default=True)
    parser.add_argument('-el', '--encodeLayer', nargs='+', default=[256, 64, 32, 16])
    parser.add_argument('-dl1', '--decodeLayer1', nargs='+', default=[16, 64, 256])
    # parser.add_argument('-dl2', '--decodeLayer2', nargs='+', default=[16, 20])
    parser.add_argument('--sigma1', default=2.5, type=float)
    # parser.add_argument('--sigma2', default=1.5, type=float)
    parser.add_argument('--f1', default=2000, type=float, help='Number of mRNA after feature selection')
    # parser.add_argument('--f2', default=2000, type=float, help='Number of ADT/ATAC after feature selection')
    parser.add_argument('--filter1', action='store_true', default=True, help='Do mRNA selection')
    # parser.add_argument('--filter2', action='store_true', default=True, help='Do ADT/ATAC selection')
    parser.add_argument('--run', default=1, type=int)
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--no_labels', action='store_true', default=False)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--workers', default=20, type=int)
    parser.add_argument('--K', default=4096, type=int)
    parser.add_argument('--max_element', default=95536, type=int)
    parser.add_argument('--m', default=0.999, type=float)
    parser.add_argument('--n', default=6, type=int)
    parser.add_argument('--c', default=1, type=int)
    parser.add_argument('--T', default=0.3, type=float)
    parser.add_argument('--p', default=0.2, type=float)
    parser.add_argument('--lam', default=0.4, type=float)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--kd_alpha', default=0.05, type=float)
    parser.add_argument('--kd_temperature', default=3, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--num_genes', default=2000, type=int)
    parser.add_argument('--classnum', default=8, type=int)
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--resolution', default=0.3, type=float)
    parser.add_argument('--reload', action='store_true', default=False)
    parser.add_argument('--flag', default='aug_nn', type=str)
    parser.add_argument('--model_path', default='./states/Muraro', type=str)
    parser.add_argument('--dataset_name', default='Muraro', type=str)

    args = parser.parse_args()
    # print_log(str(args), args.log_path)
    set_seed(args.seed)

    print(args)



    # data_mat = h5py.File(args.data_file)
    # x1 = np.array(data_mat['X1'])
    prefix = "//"

    filename = ['Plasschaert/data.h5']

    for file in filename:
        # args.data_file = prefix + file
        args.data_file = args.data_file1
        train_ami, train_nmi, train_ari = [], [], []
        test_ami, test_nmi, test_ari = [], [], []

        for i in range(1):
            print("---------- Step 1: Pretrain MoCo_SC ----------")
            model, start_time1,data,neighbors = pretrain(args=args, device=args.device)
            

            print('---------- Step 2: Get Pseudo Labels ----------')
            # adata_embedding, adata, Y, leiden_pred, _, val_loader, val_datasets = get_pseudo_label(args,
            #                                                                          model,
            #                                                                          device=args.device)
            adata_embedding1, adata1, Y, leiden_pred1, _, val_datasets1 = get_pseudo_label(args,
                                                                                       model,
                                                                                       device=args.device, type=1)

            print(f'y {len(np.unique(Y))},  pred {len(np.unique(leiden_pred1))}')

            # 可视化数据
            plot(adata_embedding1,
                 Y,
                 args.dataset_name,
                 epoch=args.epochs,
                 seed=args.seed,
                 dir_path_name="pictures")

            print('---------- Step 3: get train/val dataload ----------')

            adata1, adata_embedding1 = get_anchor(adata1,
                                                adata_embedding1,
                                                pseudo_label='leiden',
                                                k=30,
                                                percent=0.5)  # 用于确定锚定的百分比

            train_adata = adata1[adata1.obs.leiden_density_status == 'low', :].copy()
            # 从数据集中选择标记为 'high' 密度的细胞作为测试集。
            test_adata = adata1[adata1.obs.leiden_density_status == 'high', :].copy()
            # test_adata = adata2[adata2.obs.leiden_density_status == 'low', :].copy()

            print(
                f"train_adata: {np.unique(train_adata.obs['label'])}, test_adata: {np.unique(test_adata.obs['label'])}")

            # train_dataset = CellDatasetPseudoLabel(train_adata,
            #                                        pseudo_label='leiden',
            #                                        oversample_flag=True)
            # test_dataset = CellDatasetPseudoLabel(test_adata,
            #                                       pseudo_label='leiden',
            #                                       oversample_flag=False)

            print(f"teacher train dataset: {len(train_adata)}")
            print(f"teacher test dataset: {len(test_adata)}")
            
            _, y, _ = prepare_nested_h5(args.data_file)
            # _, y, _ = prepare_nested_h5(args.data_file)
            # _, y2, _ = prepro(args.data_file2)
            y_val = y[adata1.obs.leiden_density_status == 'high'].copy()
            y_train = y[adata1.obs.leiden_density_status == 'low'].copy()

            input_size1 = adata1.n_vars

            # print_log(args, args.log_path)

            encodeLayer = list(map(int, args.encodeLayer))
            decodeLayer1 = list(map(int, args.decodeLayer1))

            # 模型初始化
            # input_dim1 : 输入数据的特征维度。
            # tau、sigma1、gamma、cutoff、phi1: 用于调整模型行为的各种超参数。
            # encodeLayer、decodeLayer1: 编码器和解码器的层结构。
            # activation: 激活函数类型。
            teacher_model = Cluster(input_dim1=input_size1,  tau=args.tau,
                                    encodeLayer=encodeLayer, decodeLayer1=decodeLayer1,
                                    activation='elu', sigma1=args.sigma1, gamma=args.gamma,
                                    cutoff=args.cutoff, phi1=args.phi1, device=args.device).to(args.device)

            # print_log(str(teacher_model), args.log_path)
            print(str(teacher_model))

            if not os.path.exists(args.save_dir):  # 检查是否存在用于保存结果的目录。如果不存在，则创建这个目录。
                os.makedirs(args.save_dir)

            t0 = time()
            a = adata1.X
            b = adata1.raw.X
            c = adata1.obs.size_factors
            # print(adata1.X.shape)
            # print(adata1.raw.X.shape)
            # print(adata1.obs.size_factors.shape)
            # print(adata2.X.shape)
            # print(adata2.raw.X.shape)
            # print(adata2.obs.size_factors.shape)
            # 处理预训练的自动编码器模型（autoencoder）
            # 根据是否存在预训练权重来决定是否从头开始训练或加载已有的模型权重
            if args.ae_weights is None:  # 如果没有指定预训练权重文件（即 args.ae_weights 为 None），则调用 teacher_model.pretrain_autoencoder 方法来执行模型的预训练。传递给这个方法的数据包括两个数据集的输入特征和规模因子、批量大小、预训练的轮数以及权重文件的路径。
                # print_log("==========teacher model pretrain==========", args.log_path)
                print("==========teacher model pretrain==========")
                # X1 和 X_raw1：两个数据集的输入特征.
                teacher_model.pretrain_autoencoder(X1=train_adata.X, X_raw1=train_adata.raw.X,
                                                   sf1=train_adata.obs.size_factors,
                                                   batch_size=args.batch_size,
                                                   epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)

                # model.pretrain_autoencoder(X1=adata1.X, X_raw1=adata1.raw.X, sf1=adata1.obs.size_factors,
                #                            X2=adata2.X, X_raw2=adata2.raw.X, sf2=adata2.obs.size_factors,
                #                            batch_size=args.batch_size,
                #                            epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
            else:
                if os.path.isfile(args.ae_weights):
                    # print_log("==> loading checkpoint '{}'".format(args.ae_weights), args.log_path)
                    print("==> loading checkpoint '{}'".format(args.ae_weights))
                    checkpoint = torch.load(args.ae_weights)
                    teacher_model.load_state_dict(checkpoint['ae_state_dict'])
                else:
                    # print_log("==> no checkpoint found at '{}'".format(args.ae_weights), args.log_path)
                    print("==> no checkpoint found at '{}'".format(args.ae_weights))
                    raise ValueError

            # print_log('Pretraining time: %d seconds.' % int(time() - t0), args.log_path)
            print('Pretraining time: %d seconds.' % int(time() - t0))

            # get k 确定数据集中的最佳聚类数量 k
            # 如果 args.n_clusters 的值是 -1，则表示用户没有事先指定聚类数量，需要通过数据分析自动确定。
            if args.n_clusters == -1:
                # teacher_model.encodeBatch 方法数据集 adata1.X 进行编码。编码过程将原始数据转换为潜在空间的表示 latent。
                latent = teacher_model.encodeBatch(torch.tensor(train_adata.X).to(args.device))
                latent = latent.cpu().numpy()
                # 使用 GetCluster 函数基于潜在表示 latent 自动确定最佳的聚类数量。
                # 函数参数包括分辨率（res）和邻居数量（n），这些参数会影响聚类的结果。
                n_clusters = GetCluster(latent, res=args.resolution,
                                        n=args.n_neighbors)  # 调用 GetCluster 函数，基于编码后的潜在表示 latent 估计合适的聚类数量。
            else:
                # 如果 args.n_clusters 已定义，则直接使用指定的 n_clusters。
                # print_log("n_cluster is defined as " + str(args.n_clusters), args.log_path)
                print("n_cluster is defined as " + str(args.n_clusters))
                n_clusters = args.n_clusters

            # 定义学生模型并进行预训练(model=学生模型)
            model = Cluster(input_dim1=input_size1, tau=args.tau,
                            encodeLayer=encodeLayer, decodeLayer1=decodeLayer1,
                            activation='elu', sigma1=args.sigma1, gamma=args.gamma,
                            cutoff=args.cutoff, phi1=args.phi1, device=args.device).to(args.device)
            # print_log(str(model), args.log_path)
            print(str(model))

            # 调用 model.pretrain_autoencoder 对学生模型进行预训练。
            # print_log("==========distiller model pretrain==========", args.log_path)
            print("==========distiller model pretrain==========")
            model.pretrain_autoencoder(X1=train_adata.X, X_raw1=train_adata.raw.X, sf1=train_adata.obs.size_factors,

                                       batch_size=args.batch_size,
                                       epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)

            if not args.no_labels:  # 如果 args.no_labels 为假（即存在标签），则调用模型的 fit 方法进行聚类分析，传入数据集、大小因子、聚类数、批处理大小、迭代次数、更新间隔、容忍度、学习率和保存目录等参数。
                # 调用 teacher_model.fit 方法训练教师模型,传入标签y
                y_pred_teacher, _ = teacher_model.fit(X1=train_adata.X, X_raw1=train_adata.raw.X,
                                                      sf1=train_adata.obs.size_factors,
                                                      y=y_train,
                                                      n_clusters=n_clusters, batch_size=args.batch_size,
                                                      num_epochs=args.maxiter,
                                                      update_interval=args.update_interval, tol=args.tol,
                                                      lr=args.lr)
                # 冻结教师模型的参数，通过将 teacher_model 的所有参数的 requires_grad 属性设置为 False，确保在后续训练过程中，教师模型的参数不会被更新，确保学生模型在训练时使用的是固定的教师模型。
                for name, param in teacher_model.named_parameters():
                    param.requires_grad = False

                # 调用 model.fit 方法训练学生模型，并将训练好的教师模型传递给学生模型进行知识蒸馏，学生模型利用教师模型的知识进行学习。
                y_pred_student, _ = model.fit(X1=train_adata.X, X_raw1=train_adata.raw.X,
                                              sf1=train_adata.obs.size_factors,
                                              y=y_train,
                                              n_clusters=n_clusters, batch_size=args.batch_size,
                                              num_epochs=args.maxiter,
                                              update_interval=args.update_interval, tol=args.tol,
                                              lr=args.lr,  teacher_model=teacher_model)
            else:  # 如果 args.no_labels 为真（即不存在标签），则在调用 fit 方法时不传入标签 y。
                y_pred_teacher, _ = teacher_model.fit(X1=train_adata.X, X_raw1=train_adata.raw.X,
                                                      sf1=train_adata.obs.size_factors,
                                                      y=y_train,
                                                      n_clusters=n_clusters, batch_size=args.batch_size,
                                                      num_epochs=args.maxiter,
                                                      update_interval=args.update_interval, tol=args.tol,
                                                      lr=args.lr)
                y_pred_student, _ = model.fit(X1=train_adata.X, X_raw1=train_adata.raw.X,
                                              sf1=train_adata.obs.size_factors,
                                              y=y_train,
                                              n_clusters=n_clusters, batch_size=args.batch_size,
                                              num_epochs=args.maxiter,
                                              update_interval=args.update_interval, tol=args.tol, lr=args.lr,
                                               teacher_model=teacher_model)

            # print_log('Total time: %d seconds.' % int(time() - t0), args.log_path)
            print('Total time: %d seconds.' % int(time() - t0))
            torch.save(model.state_dict(), './model_state_dict/final_model.pth')
            print(f'y: {np.unique(y)}, y_train: {np.unique(y_train)}, y_pred_student: {np.unique(y_pred_student)}')
            # if args.prediction_file:  # 如果指定了 args.prediction_file，则根据是否存在标签，保存聚类预测结果 y_pred 到CSV文件。
            #     if not args.no_labels:  # 如果存在标签，使用 best_map 函数将预测结果映射到真实标签的编号上。
            #         y_pred_ = best_map(y_train,
            #                            y_pred_student) - 1  # 如果指定了 args.prediction_file，将预测结果 y_pred_student 映射到真实标签 y 上，并保存到 CSV 文件中。
            #         np.savetxt(args.save_dir + "/" + str(args.run) + "_pred.csv", y_pred_, delimiter=",")
            #     else:
            #         np.savetxt(args.save_dir + "/" + str(args.run) + "_pred.csv", y_pred_student, delimiter=",")
            #
            # if args.embedding_file:  # 如果指定了 args.embedding_file，将学生模型的潜在表示（即编码结果）保存到 CSV 文件中。
            #     final_latent = model.encodeBatch(torch.tensor(train_adata.X).to(args.device))
            #     final_latent = final_latent.cpu().numpy()
            #     np.savetxt(args.save_dir + "/" + str(args.run) + "_embedding.csv", final_latent, delimiter=",")

            if not args.no_labels:

                val_x = []
                val_y = []
                # for i, (x, y) in enumerate(val_loader):
                #     for j in range(len(x)):
                #         for k in range(len(x[0])):
                #             val_x.append(x[j][k])
                #         val_y.append(y[j])
                # index_random = np.random.choice(len(val_x), size=2000)

                for d, l in val_datasets1:
                    val_x.append(d.cpu().numpy())
                    val_y.append(l)

                # index_random = np.random.choice(len(val_x), size=1500)
                # test_data = []
                # test_y = []
                # for index in index_random:
                #     test_data.append(val_x[index])
                #     test_y.append(val_y[index])

                test_data = val_x
                test_y = val_y

                test_y = np.array(test_y)

                model.eval()
                with torch.no_grad():
                    X1 = torch.tensor(test_data).to(args.device)
                    Zdata = model.encodeBatch(X1, batch_size=256)
                    dist, _ = model.kmeans_loss(Zdata)
                    y_pred_student_val = torch.argmin(dist, dim=1).data.cpu().numpy()
                    
                # 评估学生模型
                print("---------- Step 5: Evaluate Student Model ----------")
                y_pred_student_val = np.array(y_pred_student_val, dtype=np.int32)

                # 使用 pandas.Series 设置为类别类型
                adata_embedding1.obs['student_prediction'] = pd.Series(y_pred_student_val).astype("category")

                # y_pred_val = best_map(y, y_pred_student_val)
                print(f'y: {np.unique(y)}, y_train: {np.unique(y_train)}, y_pred_student: {np.unique(y_pred_student)}')
                print(
                    f'y: {np.unique(y)}, y_val: {np.unique(test_y)}, y_pred_student_val: {np.unique(y_pred_student_val)}')
                ami_val = np.round(metrics.adjusted_mutual_info_score(Y, y_pred_student_val), 5)
                nmi_val = np.round(metrics.normalized_mutual_info_score(Y, y_pred_student_val), 5)
                ari_val = np.round(metrics.adjusted_rand_score(Y, y_pred_student_val), 5)
                test_ami.append(ami_val)
                test_nmi.append(nmi_val)
                test_ari.append(ari_val)

                print(args.data_file + ' testdata Final: AMI= %.4f, NMI= %.4f, ARI= %.4f' % (ami_val, nmi_val, ari_val))

            else:
                # print_log("No labels for evaluation!", args.log_path)
                print("No labels for evaluation!")
                

                