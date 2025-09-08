import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score
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

all_cluster_path = '/Users/erganshang/Desktop/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/embedding_related/'

cluster = np.load(all_cluster_path+'cluster_dict_leiden_full.npy', allow_pickle=True).item()

plot_path = '/Users/erganshang/Desktop/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/plot/cluster/'
KO_path = '/Users/erganshang/Desktop/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc/data/embedding_related/'
adata_path = '/Users/erganshang/Desktop/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/embedding_related/'
KO_whole = np.load(KO_path+'KO_good_for_PCA.npy')

KO_pert_name = np.load(KO_path+'good_pert_name.npy').tolist()

adata_all = sc.read_h5ad(adata_path+'adata_no_imputed.h5ad')

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
neighbor = 19  # 20 # small for more
resolution = 0.7  # 0.4 # big for more
sc.pp.neighbors(adata_gene, n_neighbors=neighbor)
sc.tl.leiden(adata_gene, resolution=resolution)

measure_pert_idx = [i for i, g in enumerate(all_gene) if g in KO_pert_name]  # len=95

gene_clusters = adata_gene.obs['leiden']

adata_gene.obs_names = all_gene
adata_gene.obs['cluster'] = adata_gene.obs['leiden']
sc.tl.umap(adata_gene)
adata_sub = adata_gene[measure_pert_idx, :].copy()
fig, ax = plt.subplots(figsize=(10, 8))
sc.pl.umap(adata_sub, color='leiden', legend_loc=None, size=100, show=False, ax=ax, title='')
texts = []
for i, (x, y) in enumerate(adata_sub.obsm['X_umap']):
    texts.append(plt.text(x, y, str(adata_sub.obs_names[i]), fontsize=8, alpha=0.6))

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
# plt.title("Yao", fontsize=20)
ax.set_xlabel("UMAP 1", fontsize=16)
ax.set_ylabel("UMAP 2", fontsize=16)
plt.tight_layout()
# plt.savefig(plot_path + f'all_gene_Leiden_cluster_neighbor_{neighbor}_reso_{resolution}.pdf')
# plt.show()
############################################################ Get AIR ##########################################################################################
all_pert = set(x for v in cluster.values() for x in v)
all_gene = set(adata_all.var_names.tolist())

all_pert_measured = all_pert & all_gene
all_pert_measured = sorted(list(all_pert_measured)) # 95

# X is one bunch, then ARI could only 1(Y is also one bunch) or 0(Y is not one bunch)

candidates = set(cluster['4']) & set(all_pert_measured)
candidates = sorted(candidates)
candidate_indices_in_all_gene = [adata_all.var_names.tolist().index(g) for g in candidates]
label_X = [0]*len(candidates)
label_Y = gene_clusters.iloc[candidate_indices_in_all_gene].tolist()
ari = adjusted_rand_score(label_X, label_Y)
print(f'Cluster 4 in leiden is ari: {ari}')

candidates = set(cluster['1']) & set(all_pert_measured)
candidates = sorted(candidates)
candidate_indices_in_all_gene = [adata_all.var_names.tolist().index(g) for g in candidates]
label_X = [0]*len(candidates)
label_Y = gene_clusters.iloc[candidate_indices_in_all_gene].tolist()
ari = adjusted_rand_score(label_X, label_Y)
print(f'Cluster 4 in leiden is ari: {ari}')