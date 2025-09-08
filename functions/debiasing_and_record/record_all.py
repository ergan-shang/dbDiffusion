import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import sys
import scanpy as sc
sys.path.append('/home/eshang/diffusion_and_protein/repogle/functions/summary_no_impute/')
from summary_function import *

KO_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc/data/embedding_related/'
KO_whole = np.load(KO_path + 'KO_good_for_PCA.npy')
KO_pert_name = np.load(KO_path + 'good_pert_name.npy').tolist()

sample_cluster_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/emb_cluster_wo_one_pert/'

cluster_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/embedding_related/'
cluster_dic = np.load(cluster_path + 'cluster_dict_leiden_full.npy', allow_pickle=True).item()
KO_pert_sum = np.sum(np.abs(KO_whole), axis=1)
top_num = 14
top_indices = np.argsort(KO_pert_sum)[-top_num:][::-1]
top_pert = set([KO_pert_name[index] for index in top_indices])
pert_to_sample = top_pert # choose what perturbations you want to analyze(PCC or CI etc.)
pert_to_sample = sorted(pert_to_sample)

adata_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/embedding_related/'
adata_all = sc.read_h5ad(adata_path+'adata_no_imputed.h5ad')

eff_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc/data/embedding_related/'
eff_mat = np.load(eff_path+'KO_good_for_PCA.npy')
eff_pert = np.load(eff_path+'good_pert_name.npy').tolist()
sig_gene_num = 300

all_gene_num = adata_all.shape[1]

CI_percent_notPPI_cfDiff = []
CI_percent_notPPI_sclambda = []
CI_percent_notPPI_ours = []
CI_percent_PPI_cfDiff = []
CI_percent_PPI_sclambda = []
CI_percent_PPI_ours = []

CI_percent_notPPI_scGPT = []
CI_percent_PPI_scGPT = []

record_data_all = {}

cfDiff_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/sampled_data/cfDiff_ori_cf_30000gen/'
gears_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/bm/GEARS/mean/'

for pert in tqdm(pert_to_sample):
    target = pert
    found_cluster = [key for key, value in cluster_dic.items() if
                     target in value]  # only one cluster should be found
    assert len(found_cluster) == 1
    cluster_num = found_cluster[0]

    pert_index = eff_pert.index(target)
    gene_vec = abs(eff_mat[pert_index])
    top_indices = sorted(np.argsort(gene_vec)[-sig_gene_num:])

    sclambda_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/bm/scLAMBDA/' + f'wo_{pert}/'
    scGPT_path = f'/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/bm/scGPT/sampled_data/wo_{pert}/'

    record_data_all[target] = {}
    record_data_all[target]['ctrl_mean'] = cal_observed_mean('non-targeting', adata_all)
    record_data_all[target]['ctrl_sd'] = cal_observed_sd('non-targeting', adata_all)
    record_data_all[target]['observed_mean'] = cal_observed_mean(target, adata_all)
    record_data_all[target]['observed_sd'] = cal_observed_sd(target, adata_all)

    try:
        our_sampled_path = f'/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/sampled_data/wo_{target}_location/'
        sample_cluster = np.load(sample_cluster_path + f'cluster_dict_leiden_wo_{target}.npy', allow_pickle=True).item()
        found_cluster = [key for key, value in sample_cluster.items() if
                         target in value]  # only one cluster should be found
        assert len(found_cluster) == 1
        cluster_num = found_cluster[0]
        reference = [neighbor for neighbor in sample_cluster[cluster_num] if neighbor != target]

        record_data_all[target]['notPPI_scGPT_mean'] = np.load(scGPT_path + f'mean_of_{pert}.npy')
        record_data_all[target]['notPPI_scGPT_sd'] = cal_notPPI_sd_scGPT(target, scGPT_path)

        PPI_scGPT_sd, s_square_scGPT, t_square_scGPT, n_bar = cal_PPI_sd_scGPT_conserve(reference, adata_all, scGPT_path)

        record_data_all[target]['PPI_scGPT_mean'] = cal_PPI_mean_scGPT(target, reference, adata_all, scGPT_path)
        record_data_all[target]['PPI_scGPT_sd'] = PPI_scGPT_sd



    except (FileNotFoundError, KeyError) as e:
        print(f'{target} is not measured in genes. We do not sample that!')
        record_data_all[target]['notPPI_scGPT_mean'] = None
        record_data_all[target]['notPPI_scGPT_sd'] = None
        record_data_all[target]['PPI_scGPT_mean'] = None
        record_data_all[target]['PPI_scGPT_sd'] = None

    try:
        our_sampled_path = f'/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/sampled_data/wo_{target}_location/'
        sample_cluster = np.load(sample_cluster_path + f'cluster_dict_leiden_wo_{target}.npy', allow_pickle=True).item()
        found_cluster = [key for key, value in sample_cluster.items() if target in value]  # only one cluster should be found
        assert len(found_cluster) == 1
        cluster_num = found_cluster[0]
        reference = [neighbor for neighbor in sample_cluster[cluster_num] if neighbor != target]

        ######################################################################################
        record_data_all[target]['notPPI_cfDiff_mean'] = cal_notPPI_mean_cfDiff(target, cfDiff_path)
        record_data_all[target]['notPPI_cfDiff_sd'] = cal_notPPI_sd_cfDiff(target, cfDiff_path)

        record_data_all[target]['notPPI_sclambda_mean'] = cal_notPPI_mean_sclambda(target, sclambda_path)
        record_data_all[target]['notPPI_sclambda_sd'] = cal_notPPI_sd_sclambda(target, sclambda_path)

        ######################################################################################
        PPI_sd, _, _, _ = cal_PPI_sd_forall_conserve(reference, adata_all)  # cal_PPI_sd_forall(reference, adata_all)

        record_data_all[target]['PPI_cfDiff_mean'] = cal_PPI_mean_cfDiff(target, reference, adata_all, cfDiff_path)
        record_data_all[target]['PPI_cfDiff_sd'] = PPI_sd

        record_data_all[target]['PPI_sclambda_mean'] = cal_PPI_mean_sclambda(target, reference, adata_all, sclambda_path)
        record_data_all[target]['PPI_sclambda_sd'] = PPI_sd
    except (FileNotFoundError, KeyError) as e:
        print(f'{target} is not measured in genes. We do not sample that!')
        record_data_all[target]['notPPI_cfDiff_mean'] = None
        record_data_all[target]['notPPI_cfDiff_sd'] = None

        record_data_all[target]['notPPI_sclambda_mean'] = None
        record_data_all[target]['notPPI_sclambda_sd'] = None

        record_data_all[target]['PPI_cfDiff_mean'] = None
        record_data_all[target]['PPI_cfDiff_sd'] = None

        record_data_all[target]['PPI_sclambda_mean'] = None
        record_data_all[target]['PPI_sclambda_sd'] = None


    try:
        our_sampled_path = f'/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/sampled_data/wo_{target}_location/'
        sample_cluster = np.load(sample_cluster_path + f'cluster_dict_leiden_wo_{target}.npy', allow_pickle=True).item()
        found_cluster = [key for key, value in sample_cluster.items() if target in value]  # only one cluster should be found
        assert len(found_cluster) == 1
        cluster_num = found_cluster[0]
        my_reference = [neighbor for neighbor in sample_cluster[cluster_num] if neighbor != target]

        PPI_ours_sd, s_square, t_square, n_bar = cal_PPI_sd_forall_conserve(my_reference, adata_all)  # cal_PPI_sd_forall(my_reference, adata_all)


        record_data_all[target]['notPPI_ours_mean'] = cal_notPPI_mean_ours(target, our_sampled_path)
        record_data_all[target]['notPPI_ours_sd'] = cal_notPPI_sd_ours(target, our_sampled_path)
        record_data_all[target]['PPI_ours_mean'] = cal_PPI_mean_ours(target, my_reference, adata_all, our_sampled_path)
        record_data_all[target]['PPI_ours_sd'] = PPI_ours_sd

        # centered_ours = record_data_all[target]['PPI_ours_mean']-record_data_all[target]['ctrl_mean']
        # centered_observed = record_data_all[target]['observed_mean']-record_data_all[target]['ctrl_mean']
        #
        # PCC = pearsonr(centered_ours[top_indices], centered_observed[top_indices])[0]
        #
        # print(f'pert is {pert}, PCC is {PCC}')


    except (FileNotFoundError, KeyError) as e:
        print(f'{target} is not measured in genes. We do not sample that!')
        record_data_all[target]['notPPI_ours_mean'] = None
        record_data_all[target]['notPPI_ours_sd'] = None
        record_data_all[target]['PPI_ours_mean'] = None
        record_data_all[target]['PPI_ours_sd'] = None

    try:
        gears_mean = np.load(gears_path+f'mean_of_{target}.npy', allow_pickle=True).item()[target]
        record_data_all[target]['gears_mean'] = gears_mean

    except (FileNotFoundError, KeyError) as e:
        print(f'Skipped {target}(not exist in perturbation graph!). Append None!')
        record_data_all[target]['gears_mean'] = None





save_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/summary_no_impute/'

np.save(save_path+'record_data_all.npy', record_data_all)



