import numpy as np
import pandas as pd
import scanpy as sc
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
import pandas as pd
from tqdm import tqdm
import pickle
from scipy.sparse import csr_matrix, csc_matrix
import json
from adjustText import adjust_text
from collections import defaultdict

KO_path = '/Users/erganshang/Desktop/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc/data/embedding_related/'
adata_path = '/Users/erganshang/Desktop/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/embedding_related/'
KO_whole = np.load(KO_path+'KO_good_for_PCA.npy')

KO_pert_name = np.load(KO_path+'good_pert_name.npy').tolist()

adata_all = sc.read_h5ad(adata_path+'adata_no_imputed.h5ad')

# target_name = 'ZC3H13'
def get_cluster_dict(target_name, KO_mat, KO_pert, adata, plot_path, save_path, full_cluster):
    idx = KO_pert.index(target_name)
    KO_mat_wo_tar = np.delete(KO_mat, idx, axis=0)
    all_pert_name = KO_pert[:idx] + KO_pert[idx+1:]

    adata_before_PCA = sc.AnnData(KO_mat_wo_tar)
    n_comps = 64
    sc.pp.scale(adata_before_PCA)  # scale to mean 0 variance 1
    sc.tl.pca(adata_before_PCA, n_comps=n_comps, svd_solver='arpack')

    mat_for_emb = adata_before_PCA.obsm['X_pca']

    ################### get embedding dict ##################
    emb_dict = {}
    for i in range(len(all_pert_name)):
        emb_dict[all_pert_name[i]] = mat_for_emb[i]
    emb_dict['non-targeting'] = np.mean(list(emb_dict.values()), axis=0)
    ###########################################################

    pert_adata = sc.AnnData(mat_for_emb)

    pert_adata.obs['celltype'] = pd.Series(all_pert_name, index=pert_adata.obs_names)
    n_neighbors = 5
    resolution = 2.
    sc.pp.neighbors(pert_adata, n_neighbors=n_neighbors, use_rep='X')  # small neighbors more clusters
    sc.tl.leiden(pert_adata, resolution=resolution)  # big resolution more clusters
    sc.tl.umap(pert_adata)

    ####### plot the cluster ##########
    print(f'Print cluster without {target_name}')
    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.umap(pert_adata, color='leiden', legend_loc='on data', size=100, show=False, ax=ax)
    texts = []
    for i, (x, y) in enumerate(pert_adata.obsm['X_umap']):
        texts.append(plt.text(x, y, str(pert_adata.obs['celltype'].iloc[i]), fontsize=6, alpha=0.6))

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.title(f"Leiden Clustering with Labels without {target_name}")
    plt.tight_layout()
    # plt.savefig(plot_path + f'Leiden_cluster_neighbor_{n_neighbors}_reso_{resolution}_wo_{target_name}.pdf')
    # plt.show()
    #####################################
    ####### record the cluster ##########
    print(f'Record cluster without {target_name}')
    cluster_dict_leiden_wo_tar = defaultdict(list)

    for celltype, cluster in zip(pert_adata.obs['celltype'], pert_adata.obs['leiden']):
        cluster_dict_leiden_wo_tar[cluster].append(celltype)

    cluster_dict_leiden_wo_tar = dict(cluster_dict_leiden_wo_tar)

    # for k, v in cluster_dict_leiden_wo_tar.items():
    #     print(f'{k}: {v}')
    #####################################

    ######################### locate the target #########################
    print(f'Locate for {target_name}')
    # you cannot use cells from the target
    adata_all = adata[adata.obs['celltype']!=target_name].copy()
    all_gene = adata_all.var_names.tolist()

    n_comps = 64
    sc.pp.scale(adata_all)  # scale to mean 0 variance 1
    sc.tl.pca(adata_all, n_comps=n_comps, svd_solver='arpack')

    load_mat = adata_all.varm['PCs']  # (1500, 64) each column is the loading coefficients for each gene

    k = 50
    top_pc = load_mat[:, :k]  # (1500, k)
    cor_mat = top_pc @ top_pc.T  # (1500, 1500)

    adata_gene_cov = sc.AnnData(cor_mat)
    sc.tl.pca(adata_gene_cov, n_comps=50, svd_solver='arpack')
    final_cor_mat = adata_gene_cov.obsm['X_pca']

    adata_gene = sc.AnnData(final_cor_mat)
    neighbor = 20  # 20
    resolution = 0.4  # 0.4
    sc.pp.neighbors(adata_gene, n_neighbors=neighbor)
    sc.tl.leiden(adata_gene, resolution=resolution)

    measure_pert_idx = [i for i, g in enumerate(all_gene) if g in all_pert_name]  # len=95

    gene_clusters = adata_gene.obs['leiden']

    adata_gene.obs_names = all_gene
    adata_gene.obs['cluster'] = adata_gene.obs['leiden']
    sc.tl.umap(adata_gene)
    adata_sub = adata_gene[measure_pert_idx, :].copy()
    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.umap(adata_sub, color='leiden', legend_loc='on data', size=100, show=False, ax=ax, title='')
    texts = []
    for i, (x, y) in enumerate(adata_sub.obsm['X_umap']):
        texts.append(plt.text(x, y, str(adata_sub.obs_names[i]), fontsize=8, alpha=0.6))

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    # plt.title("Yao", fontsize=20)
    ax.set_xlabel("UMAP 1", fontsize=16)
    ax.set_ylabel("UMAP 2", fontsize=16)
    plt.tight_layout()
    plt.savefig(plot_path + f'gene_Leiden_cluster_neighbor_{neighbor}_reso_{resolution}.pdf')
    # plt.show()

    print(gene_clusters)

    if target_name in all_gene:
        target_index = all_gene.index(target_name)
        target_cluster = gene_clusters.iloc[target_index]
        same_cluster_indices = gene_clusters[gene_clusters == target_cluster].index.tolist()
        same_cluster_indices = [int(item) for item in same_cluster_indices]
        genes_found = [all_gene[i] for i in same_cluster_indices if i != target_index]  # we should not include itself

        cluster_count = {}
        for key, val in cluster_dict_leiden_wo_tar.items():
            res = []
            for item in val:
                if item in genes_found:
                    res.append(item)
            cluster_count[key] = res

        score_dict = {}
        for key in cluster_dict_leiden_wo_tar:
            total = len(cluster_dict_leiden_wo_tar[key])
            matched_genes = cluster_count[key]
            matched = len(matched_genes)
            proportion = matched / total if total > 0 else 0
            score = (matched) * proportion  # 👈 score function!
            score_dict[key] = {
                'score': score,
                'proportion': proportion,
                'matched_count': matched,
                'matched_genes': matched_genes
            }

        max_score = max(v['score'] for v in score_dict.values())
        best_clusters = {k: v for k, v in score_dict.items() if v['score'] == max_score}

        first_key = next(iter(best_clusters))  # the location
        print(f'locate {target_name} at {first_key}!')
        print(f'The neighbor is {cluster_dict_leiden_wo_tar[first_key]}')
        # for key, value in best_clusters.items():
        #     print(f'key is {key} and value is {value}')
        #     print(f'The matched cluster is {cluster_dict_leiden_wo_tar[first_key]}')

        pert_for_ave = cluster_dict_leiden_wo_tar[first_key]
        embeddings = [emb_dict[pert] for pert in pert_for_ave if pert in emb_dict]
        emb_dict[target_name] = np.mean(embeddings, axis=0)

        cluster_dict_leiden_wo_tar[first_key].append(target_name)

        ################################### determine distance of location ###################################
        old_found_cluster = [key for key, value in full_cluster.items() if target_name in value]
        assert len(old_found_cluster) == 1
        old_cluster_num = old_found_cluster[0]
        old_reference = [item for item in full_cluster[old_cluster_num] if item!=target_name]

        new_cluster_num = first_key
        new_reference = [item for item in cluster_dict_leiden_wo_tar[first_key] if item!=target_name]

        max_len = -float('inf')
        max_label = None # the label of cluster_dict_leiden_wo_tar
        for key, val in cluster_dict_leiden_wo_tar.items():
            len_intersect = len(set(old_reference) & set(val)) # old_reference not include target
            if len_intersect>max_len:
                max_len = len_intersect
                max_label = key
        cluster_distance = None
        if max_label==new_cluster_num:
            print('Locate in the same cluster!')
            cluster_distance = 0
        else:
            print(f'The location is {first_key}; The ground-truth is {max_label}.')
            old_emb = [pert_adata.obsm['X_umap'][all_pert_name.index(pert)] for pert in cluster_dict_leiden_wo_tar[max_label] if pert != target_name]
            old_emb_ave = np.mean(old_emb, axis=0)
            new_emb = [pert_adata.obsm['X_umap'][all_pert_name.index(pert)] for pert in cluster_dict_leiden_wo_tar[first_key] if pert != target_name]
            new_emb_ave = np.mean(new_emb, axis=0)
            cluster_distance = np.sqrt(np.sum((old_emb_ave - new_emb_ave)**2))
            print(f'The distance is calculated as {cluster_distance}')

        np.save(save_path + f'emb_dict_wo_{target_name}.npy', emb_dict)
        np.save(save_path + f'cluster_dict_leiden_wo_{target_name}.npy', cluster_dict_leiden_wo_tar)
        np.save(save_path + f'cluster_distance_wo_{target_name}.npy', cluster_distance)
    else:
        print(f'{target_name} is not measured in genes!')
        return

save_path = '/Users/erganshang/Desktop/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/emb_cluster_wo_one_pert/'
plot_path = '/Users/erganshang/Desktop/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/plot/emb_cluster_wo_one_pert/'
all_cluster_path = '/Users/erganshang/Desktop/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/embedding_related/'

cluster = np.load(all_cluster_path+'cluster_dict_leiden_full.npy', allow_pickle=True).item()

KO_pert_sum = np.sum(np.abs(KO_whole), axis=1)
top_num = 20 # If you want to analyze perturbations with top 20 effect sizes one by one...
top_indices = np.argsort(KO_pert_sum)[-top_num:][::-1]
top_pert = set([KO_pert_name[index] for index in top_indices])
pert_to_sample = top_pert | set(cluster['1']) # you can also add perturbations from one specific clusters

pert_to_sample = sorted(pert_to_sample)
# pert_to_sample = ['ZC3H13']
for pert in tqdm(pert_to_sample):
    get_cluster_dict(pert, KO_whole, KO_pert_name, adata_all, plot_path, save_path, cluster)










