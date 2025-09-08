import scanpy as sc
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import pickle
from scipy.sparse import csr_matrix, csc_matrix
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

def cal_observed_mean(target_cluster_name, all_dataset): # read no log
    adata_all = all_dataset.copy()

    adata_all.var_names_make_unique()
    sc.pp.filter_cells(adata_all, min_genes=10)
    # sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata_all, target_sum=1e4)
    sc.pp.log1p(adata_all)

    if target_cluster_name=='non-targeting':
        return np.mean(adata_all[adata_all.obs['celltype']=='non-targeting'].copy().X.toarray(), axis=0)

    return np.mean(adata_all[adata_all.obs['celltype']==target_cluster_name].copy().X.toarray(), axis=0) # log

def cal_notPPI_mean_cfDiff(target_cluster_name, cfDiff_sampled_path):
    target_mat_gen = np.load(cfDiff_sampled_path + f'real_cell{str(target_cluster_name)}_cache5_uniform.npy')

    return np.mean(target_mat_gen, axis=0) # log

def cal_notPPI_mean_sclambda(target_cluster_name, sclambda_sampled_path):
    target_mat_gen = np.load(sclambda_sampled_path + f'sample_of_{str(target_cluster_name)}.npy', allow_pickle=True).item()[f'{str(target_cluster_name)}+ctrl']
    target_mat_gen = np.array(target_mat_gen)

    return np.mean(target_mat_gen, axis=0) # log

def cal_notPPI_mean_ours(target_cluster_name, generated_path):
    target_mat_gen = np.load(generated_path + str(target_cluster_name) + '_v3_eta1.5_30000gen(full)_acc_no_impute' + '.npy')

    return np.mean(target_mat_gen, axis=0) # log

def cal_PPI_mean_cfDiff(target_cluster_name, cluster_pert_list, all_dataset, generated_data_path): # read no log
    '''
    :param target_pert_name: K-th perturbation
    :param cluster_pert_list: like [10, 11]
    :param all_dataset: should be in .h5ad, but never gone through log transformation
    :param generated_data_path: path of the generated cell matrix of K-th perturbation and those in the clusters
    :param cluster_dict: a dictionary, like {9: ['A', 'B', ...]}
    :return: mean vector for all genes
    '''
    target_mat_gen = np.load(generated_data_path + f'real_cell{str(target_cluster_name)}_cache5_uniform.npy')

    adata_all = all_dataset.copy() # no log

    adata_all.var_names_make_unique()
    sc.pp.filter_cells(adata_all, min_genes=10)
    # sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata_all, target_sum=1e4)
    sc.pp.log1p(adata_all)

    mean_value = np.zeros(target_mat_gen.shape[1])

    for i in cluster_pert_list:
        cluster_mat_gen = np.load(generated_data_path + f'real_cell{str(i)}_cache5_uniform.npy')
        adata_cluster_ori = adata_all[adata_all.obs['celltype']==i].copy().X.toarray()
        mean_value += np.mean(adata_cluster_ori, axis=0)-np.mean(cluster_mat_gen, axis=0)

    mean_value = mean_value/len(cluster_pert_list)

    mean_value += np.mean(target_mat_gen, axis=0)

    return mean_value # log

def cal_PPI_mean_sclambda(target_cluster_name, cluster_pert_list, all_dataset, generated_data_path): # read no log
    '''
    :param target_pert_name: K-th perturbation
    :param cluster_pert_list: like [10, 11]
    :param all_dataset: should be in .h5ad, but never gone through log transformation
    :param generated_data_path: path of the generated cell matrix of K-th perturbation and those in the clusters
    :param cluster_dict: a dictionary, like {9: ['A', 'B', ...]}
    :return: mean vector for all genes
    '''
    target_mat_gen = np.load(generated_data_path + f'sample_of_{str(target_cluster_name)}.npy', allow_pickle=True).item()[f'{str(target_cluster_name)}+ctrl']
    target_mat_gen = np.array(target_mat_gen)

    adata_all = all_dataset.copy() # no log

    adata_all.var_names_make_unique()
    sc.pp.filter_cells(adata_all, min_genes=10)
    # sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata_all, target_sum=1e4)
    sc.pp.log1p(adata_all)

    mean_value = np.zeros(target_mat_gen.shape[1])

    for i in cluster_pert_list:
        cluster_mat_gen = np.load(generated_data_path + f'sample_of_{str(i)}.npy', allow_pickle=True).item()[f'{str(i)}+ctrl']
        cluster_mat_gen = np.array(cluster_mat_gen)
        adata_cluster_ori = adata_all[adata_all.obs['celltype']==i].copy().X.toarray()
        mean_value += np.mean(adata_cluster_ori, axis=0)-np.mean(cluster_mat_gen, axis=0)

    mean_value = mean_value/len(cluster_pert_list)

    mean_value += np.mean(target_mat_gen, axis=0)

    return mean_value # log

def cal_PPI_mean_scGPT(target_cluster_name, cluster_pert_list, all_dataset, generated_data_path): # read no log
    '''
    :param target_pert_name: K-th perturbation
    :param cluster_pert_list: like [10, 11]
    :param all_dataset: should be in .h5ad, but never gone through log transformation
    :param generated_data_path: path of the generated cell matrix of K-th perturbation and those in the clusters
    :param cluster_dict: a dictionary, like {9: ['A', 'B', ...]}
    :return: mean vector for all genes
    '''
    target_mat_gen = np.load(generated_data_path + f'sample_of_{str(target_cluster_name)}.npy')
    # target_mat_gen = np.array(target_mat_gen)

    adata_all = all_dataset.copy() # no log

    adata_all.var_names_make_unique()
    sc.pp.filter_cells(adata_all, min_genes=10)
    # sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata_all, target_sum=1e4)
    sc.pp.log1p(adata_all)

    mean_value = np.zeros(target_mat_gen.shape[1])

    count = 0
    for i in cluster_pert_list:
        file_path = generated_data_path + f'sample_of_{str(i)}.npy'
        if not os.path.exists(file_path):
            print(f"{file_path} not found, skipping...")
            continue
        count += 1
        cluster_mat_gen = np.load(generated_data_path + f'sample_of_{str(i)}.npy')
        # cluster_mat_gen = np.array(cluster_mat_gen)
        adata_cluster_ori = adata_all[adata_all.obs['celltype']==i].copy().X.toarray()
        mean_value += np.mean(adata_cluster_ori, axis=0)-np.mean(cluster_mat_gen, axis=0)

    mean_value = mean_value/count

    mean_value += np.mean(target_mat_gen, axis=0)

    return mean_value # log

def cal_PPI_mean_ours(target_cluster_name, cluster_pert_list, all_dataset, generated_data_path): # read no log
    '''
    :param target_pert_name: K-th perturbation
    :param cluster_pert_list: like [10, 11]
    :param all_dataset: should be in .h5ad, but never gone through log transformation
    :param generated_data_path: path of the generated cell matrix of K-th perturbation and those in the clusters
    :param cluster_dict: a dictionary, like {9: ['A', 'B', ...]}
    :return: mean vector for all genes
    '''
    target_mat_gen = np.load(generated_data_path + str(target_cluster_name) + '_v3_eta1.5_30000gen(full)_acc_no_impute' + '.npy')

    adata_all = all_dataset.copy() # no log

    adata_all.var_names_make_unique()
    sc.pp.filter_cells(adata_all, min_genes=10)
    # sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata_all, target_sum=1e4)
    sc.pp.log1p(adata_all)

    mean_value = np.zeros(target_mat_gen.shape[1])

    for i in cluster_pert_list:
        cluster_mat_gen = np.load(generated_data_path + str(i) + '_v3_eta1.5_30000gen(full)_acc_no_impute' + '.npy')
        adata_cluster_ori = adata_all[adata_all.obs['celltype']==i].copy().X.toarray()
        mean_value += np.mean(adata_cluster_ori, axis=0)-np.mean(cluster_mat_gen, axis=0)

    mean_value = mean_value/len(cluster_pert_list)

    mean_value += np.mean(target_mat_gen, axis=0)

    return mean_value # log

def cal_observed_sd(target_cluster_name, all_dataset):
    adata_all = all_dataset.copy()

    adata_all.var_names_make_unique()
    sc.pp.filter_cells(adata_all, min_genes=10)
    # sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata_all, target_sum=1e4)
    sc.pp.log1p(adata_all)

    if target_cluster_name=='non-targeting':
        n = len(adata_all[adata_all.obs['celltype']=='non-targeting'])

        return np.var(adata_all[adata_all.obs['celltype']=='non-targeting'].copy().X.toarray(),
                      axis=0, ddof=1) ** 0.5 / (n ** 0.5)

    n = len(adata_all[adata_all.obs['celltype']==target_cluster_name])

    return np.var(adata_all[adata_all.obs['celltype']==target_cluster_name].copy().X.toarray(), axis=0, ddof=1)**0.5 / (n**0.5)

def cal_notPPI_sd_cfDiff(target_cluster_name, cfDiff_sampled_path):
    target_mat_gen = np.load(cfDiff_sampled_path + f'real_cell{str(target_cluster_name)}_cache5_uniform.npy')
    n = len(target_mat_gen)

    return np.var(target_mat_gen, axis=0, ddof=1)**0.5/(n**0.5)  # log

def cal_notPPI_sd_scGPT(target_cluster_name, scGPT_sampled_path):
    target_mat_gen = np.load(scGPT_sampled_path + f'sample_of_{target_cluster_name}.npy')
    n = len(target_mat_gen)

    return np.var(target_mat_gen, axis=0, ddof=1)**0.5/(n**0.5)  # log

def cal_notPPI_sd_sclambda(target_cluster_name, sclambda_sampled_path):
    target_mat_gen = np.load(sclambda_sampled_path + f'sample_of_{str(target_cluster_name)}.npy', allow_pickle=True).item()[f'{str(target_cluster_name)}+ctrl']
    target_mat_gen = np.array(target_mat_gen)
    n = len(target_mat_gen)

    return np.var(target_mat_gen, axis=0, ddof=1)**0.5/(n**0.5)  # log

def cal_notPPI_sd_ours(target_cluster_name, generated_path):
    target_mat_gen = np.load(generated_path + str(target_cluster_name) + '_v3_eta1.5_30000gen(full)_acc_no_impute' + '.npy')
    n = len(target_mat_gen)

    return np.var(target_mat_gen, axis=0, ddof=1)**0.5/(n**0.5)  # log

def cal_PPI_sd_forall(cluster_pert_list, all_dataset):
    '''
    :param cluster_pert_list: like [9, 10]
    :param all_dataset: should be in .h5ad, but never gone through log transformation
    :param cluster_dict: a dictionary, like {9: ['A', 'B', ...]}
    :return:
    '''
    K_c = len(cluster_pert_list)

    adata_all = all_dataset.copy()
    adata_all.var_names_make_unique()
    sc.pp.filter_cells(adata_all, min_genes=10)
    # sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata_all, target_sum=1e4)
    sc.pp.log1p(adata_all)

    normalized_const = 0
    for i in cluster_pert_list:
        normalized_const += 1 / adata_all[adata_all.obs['celltype']==i].copy().X.shape[0]
    normalized_const = (normalized_const / (K_c ** 2)) ** (0.5)

    variance = np.zeros(adata_all.shape[1])

    count = 0

    for i in cluster_pert_list:
        adata_cluster_ori = adata_all[adata_all.obs['celltype']==i].copy().X.toarray()
        variance += np.var(adata_cluster_ori, axis=0) * adata_cluster_ori.shape[0]
        # print(f'Sample size of {i} is {adata_cluster_ori.shape[0]}')
        count += adata_cluster_ori.shape[0]
    variance /= (count-len(cluster_pert_list))
    return variance**0.5*normalized_const

def cal_PPI_sd_forall_conserve(cluster_pert_list, all_dataset):
    K_c = len(cluster_pert_list)

    adata_all = all_dataset.copy()
    adata_all.var_names_make_unique()
    sc.pp.filter_cells(adata_all, min_genes=10)
    # sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata_all, target_sum=1e4)
    sc.pp.log1p(adata_all)

    mean_of_each_pert = np.zeros((K_c, adata_all.shape[1]))

    s_square = np.zeros(adata_all.shape[1])
    count = 0

    sample_size_record = []

    for i, item in enumerate(cluster_pert_list):
        adata_cluster_ori = adata_all[adata_all.obs['celltype']==item].copy().X.toarray()
        mean_of_each_pert[i] = np.mean(adata_cluster_ori, axis=0)
        s_square += np.var(adata_cluster_ori, axis=0) * adata_cluster_ori.shape[0]
        # print(f'Sample size of {i} is {adata_cluster_ori.shape[0]}')
        count += adata_cluster_ori.shape[0]
        sample_size_record.append(adata_cluster_ori.shape[0])
    s_square /= (count-K_c)

    t_square = np.var(mean_of_each_pert, axis=0, ddof=1)

    variance = np.zeros(adata_all.shape[1])

    for i, sample_size in enumerate(sample_size_record):
        variance += s_square/sample_size + t_square

    return (variance**0.5)/K_c, s_square, t_square, np.mean(sample_size_record) # (conservative_sd, s^2, t^2, \bar n)


def cal_PPI_sd_scGPT(cluster_pert_list, all_dataset, generated_data_path):
    '''
    :param cluster_pert_list: like [9, 10]
    :param all_dataset: should be in .h5ad, but never gone through log transformation
    :param cluster_dict: a dictionary, like {9: ['A', 'B', ...]}
    :return:
    '''
    K_c = 0

    adata_all = all_dataset.copy()
    adata_all.var_names_make_unique()
    sc.pp.filter_cells(adata_all, min_genes=10)
    # sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata_all, target_sum=1e4)
    sc.pp.log1p(adata_all)

    normalized_const = 0
    for i in cluster_pert_list:
        file_path = generated_data_path + f'sample_of_{str(i)}.npy'
        if not os.path.exists(file_path):
            print(f"{file_path} not found, skipping...")
            continue
        K_c += 1
        normalized_const += 1 / adata_all[adata_all.obs['celltype']==i].copy().X.shape[0]
    normalized_const = (normalized_const / (K_c ** 2)) ** (0.5)

    variance = np.zeros(adata_all.shape[1])

    count = 0

    for i in cluster_pert_list:
        file_path = generated_data_path + f'sample_of_{str(i)}.npy'
        if not os.path.exists(file_path):
            print(f"{file_path} not found, skipping...")
            continue
        adata_cluster_ori = adata_all[adata_all.obs['celltype']==i].copy().X.toarray()
        variance += np.var(adata_cluster_ori, axis=0) * adata_cluster_ori.shape[0]
        # print(f'Sample size of {i} is {adata_cluster_ori.shape[0]}')
        count += adata_cluster_ori.shape[0]
    variance /= (count-K_c)
    return variance**0.5*normalized_const

def cal_PPI_sd_scGPT_conserve(cluster_pert_list, all_dataset, generated_data_path):
    K_c = 0

    adata_all = all_dataset.copy()
    adata_all.var_names_make_unique()
    sc.pp.filter_cells(adata_all, min_genes=10)
    # sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata_all, target_sum=1e4)
    sc.pp.log1p(adata_all)

    mean_of_each_pert = []

    s_square = np.zeros(adata_all.shape[1])
    count = 0

    sample_size_record = []

    for i, item in enumerate(cluster_pert_list):
        file_path = generated_data_path + f'sample_of_{str(item)}.npy'
        if not os.path.exists(file_path):
            print(f"{file_path} not found, skipping...")
            continue
        K_c += 1
        adata_cluster_ori = adata_all[adata_all.obs['celltype']==item].copy().X.toarray()
        mean_of_each_pert.append(np.mean(adata_cluster_ori, axis=0))
        s_square += np.var(adata_cluster_ori, axis=0) * adata_cluster_ori.shape[0]
        # print(f'Sample size of {i} is {adata_cluster_ori.shape[0]}')
        count += adata_cluster_ori.shape[0]
        sample_size_record.append(adata_cluster_ori.shape[0])
    s_square /= (count-K_c)

    mean_of_each_pert = np.vstack(mean_of_each_pert)
    t_square = np.var(mean_of_each_pert, axis=0, ddof=1)

    variance = np.zeros(adata_all.shape[1])

    for i, sample_size in enumerate(sample_size_record):
        variance += s_square/sample_size + t_square

    return (variance**0.5)/K_c, s_square, t_square, np.mean(sample_size_record) # (conservative_sd, s^2, t^2, \bar n)
