import scanpy as sc
import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
from adjustText import adjust_text
from collections import defaultdict
from scipy.cluster.hierarchy import fcluster
import pandas as pd
import pickle
from scipy.sparse import csr_matrix, csc_matrix
import json

data_path = '/Users/erganshang/Desktop/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc/data/embedding_related/'
adata_path = '/Users/erganshang/Desktop/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/embedding_related/'
plot_path = '/Users/erganshang/Desktop/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/plot/cluster/'
save_path = '/Users/erganshang/Desktop/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/embedding_related/'

emb_dict = np.load(data_path+'emb_dict.npy', allow_pickle=True).item()

n_comps = 64

# KO_good_after_PCA = np.load(data_path+f'KO_good_after_PCA_pc_{n_comps}.npy')
KO_good_after_PCA = np.load(data_path+f'KO_good_after_PCA.npy')

adata_all = sc.read_h5ad(adata_path + 'adata_no_imputed.h5ad')

all_pert = np.load(data_path+'good_pert_name.npy').tolist()
######################################################################################################################
pert_adata = sc.AnnData(KO_good_after_PCA)
pert_adata.obs['celltype'] = pd.Series(all_pert, index=pert_adata.obs_names)
n_neighbors = # Choose your own parameters based on your dataset
resolution = # Choose your own parameters based on your dataset
sc.pp.neighbors(pert_adata, n_neighbors=n_neighbors, use_rep='X') # small neighbors more clusters
sc.tl.leiden(pert_adata, resolution=resolution) # big resolution more clusters
sc.tl.umap(pert_adata)
fig, ax = plt.subplots(figsize=(10, 8))
sc.pl.umap(pert_adata, color='leiden', legend_loc='on data', size=100, show=False, ax=ax, title='')
texts = []
for i, (x, y) in enumerate(pert_adata.obsm['X_umap']):
    texts.append(plt.text(x, y, str(pert_adata.obs['celltype'].iloc[i]), fontsize=8, alpha=0.6))

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

plt.title("Yao", fontsize=20)
ax.set_xlabel("UMAP 1", fontsize=16)
ax.set_ylabel("UMAP 2", fontsize=16)
plt.tight_layout()
# plt.savefig(plot_path+f'Leiden_cluster_neighbor_{n_neighbors}_reso_{resolution}_pc_{n_comps}.pdf')
plt.savefig(plot_path+f'Leiden_cluster_neighbor_{n_neighbors}_reso_{resolution}.pdf')
plt.show()
######################################################################################################################
cluster_dict_leiden = defaultdict(list)

for celltype, cluster in zip(pert_adata.obs['celltype'], pert_adata.obs['leiden']):
    cluster_dict_leiden[cluster].append(celltype)

cluster_dict_leiden = dict(cluster_dict_leiden)

for k, v in cluster_dict_leiden.items():
    print(f'{k}: {v}')

np.save(save_path+f'cluster_dict_leiden_full.npy', cluster_dict_leiden)
