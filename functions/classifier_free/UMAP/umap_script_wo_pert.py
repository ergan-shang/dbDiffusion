import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
from itertools import islice
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
from scipy import stats
import sys

sys.path.append('/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/functions/diffusion/classifier_free_v3_no_saver/VAE') # GPU server
from VAE_model import VAE
from torch.autograd import Variable
import celltypist
from tqdm import tqdm
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler
import umap
import random
import torch

def load_VAE(VAE_path):
    autoencoder = VAE(
        num_genes=1500,
        device='cuda',
        seed=0,
        loss_ae='mse',
        hidden_dim=128,
        decoder_activation='ReLU',
    )
    autoencoder.load_state_dict(torch.load(VAE_path))
    return autoencoder
KO_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc/data/embedding_related/'
KO_whole = np.load(KO_path + 'KO_good_for_PCA.npy')
KO_pert_name = np.load(KO_path + 'good_pert_name.npy').tolist()


cluster_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/embedding_related/'
cluster_dict = np.load(cluster_path+'cluster_dict_leiden_full.npy', allow_pickle=True).item()

KO_pert_sum = np.sum(np.abs(KO_whole), axis=1)
top_num = 20
top_indices = np.argsort(KO_pert_sum)[-top_num:][::-1]
top_pert = set([KO_pert_name[index] for index in top_indices])
pert_to_sample = top_pert | set(cluster_dict['1'])

pert_to_sample = sorted(pert_to_sample) # choose what perturbation you want to analysis and change the cells from latent space to real cells

adata_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/embedding_related/'
adata = sc.read_h5ad(adata_path+'adata_no_imputed.h5ad')
adata.var_names_make_unique()
# sc.pp.filter_cells(adata, min_genes=10)
# sc.pp.filter_genes(adata, min_cells=3)
gene_names = adata.var_names

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
cell_data = adata.X.toarray()
celltype = adata.obs['celltype'].tolist()

for pert in tqdm(pert_to_sample):
    plot_path = f'/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/plot/UMAP/wo_{pert}_location/'
    if os.path.isdir(plot_path):
        print(f"Skip {pert} — UMAP already finished.")
        continue
    try:
        sample_cluster_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/emb_cluster_wo_one_pert/'
        sample_cluster = np.load(sample_cluster_path + f'cluster_dict_leiden_wo_{pert}.npy', allow_pickle=True).item()
        found_cluster = [key for key, value in sample_cluster.items() if
                         pert in value]  # only one cluster should be found
        assert len(found_cluster) == 1
        cluster_num = found_cluster[0]
        neighbor_pert = sample_cluster[cluster_num]

        print(f'UMAP for {pert}...')
        plot_path = f'/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/plot/UMAP/wo_{pert}_location/'
        os.makedirs(plot_path, exist_ok=True)
        VAE_path = f'/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/model/VAE_wo_{pert}_no_impute_locate/model_seed=0_step=199999.pt'  # path to VAE not Unet!
        autoencoder = load_VAE(VAE_path)
        generated_data_path = f'/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/sampled_data/wo_{pert}_location/'
        for neighbor in neighbor_pert:
            print(f'loading {neighbor}...')
            npzfile = np.load(generated_data_path + f'{neighbor}_v3_eta1.5_30000gen_acc_no_impute.npz', allow_pickle=True)
            cell_gen_all = []
            gen_class = []
            numpy_file = []
            length = len(adata[adata.obs['celltype'] == str(neighbor)])
            length = min(length, 30000)
            numpy_file.append(npzfile['cell_gen']) # cells in latent space
            numpy_file = np.concatenate(numpy_file, axis=0)
            numpy_file = autoencoder(torch.tensor(numpy_file).cuda(), return_decoded=True).detach().cpu().numpy() # change to real cells
            np.save(generated_data_path + f'{neighbor}_v3_eta1.5_30000gen(full)_acc_no_impute.npy', numpy_file)
            cell_gen_all.append(numpy_file[:int(length)])

            gen_class += [f'gen {neighbor}'] * int(length)
            cell_gen_all = np.concatenate(cell_gen_all, axis=0)
            cell_gen = cell_gen_all

            adata = np.concatenate((cell_data, cell_gen), axis=0)
            adata = ad.AnnData(adata, dtype=np.float32)

            print(f'The shape is {adata.X.shape}')
            adata.obs['celltype'] = np.concatenate((celltype, gen_class))
            adata.obs['cell_name'] = [f"true_Cell" for i in range(cell_data.shape[0])] + [f"gen_Cell" for i in range(cell_gen.shape[0])]
            ################################################# UMAP ##############################################
            sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3,
                                        min_disp=0.5)  # This will store a data at adata.var['highly_variable'] for further filtering
            adata.raw = adata
            adata = adata[:, adata.var.highly_variable]

            print('Start PCA...')

            sc.pp.scale(adata)
            print('check 1')
            sc.tl.pca(adata,
                      svd_solver='arpack')  # data will be stored at adata.obsm['X_pca'], 'arpack' is for SVD on sparse data
            print('check 2')
            sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
            print('check 3')
            sc.tl.umap(adata)

            # gen_cato = ['gen ' + pert]
            cato_and_gen = list(set(adata.obs['celltype'].tolist()))

            print('Start UMAP!')
            print(f'category is {neighbor}')
            color_dict = {}
            for cat in cato_and_gen:
                print(f'cat is {cat}')
                if cat == neighbor:
                    color_dict[cat] = 'tab:orange'
                elif cat == f'gen {neighbor}':
                    color_dict[cat] = 'tab:blue'
                else:
                    color_dict[cat] = 'black'
            sc.pl.umap(adata=adata, color="celltype", groups=[neighbor, f'gen {neighbor}'], size=8, palette=color_dict,
                       show=False, save=False)  # groups will make specific categories stand out
            plt.legend(loc='upper right')
            plt.title(f'{neighbor}_wo_{pert}')
            umap_file_path = os.path.join(plot_path,
                                          f'UMAP_of_cleary_data_{neighbor}_and_diffusion_sampled_data_classifier_free_fullLoop_v3_acc(PCA_emb)_eta1.5_wo_{pert}_30000gen_no_impute.png')
            plt.savefig(umap_file_path, dpi=300, bbox_inches='tight')
            plt.close()
    except (FileNotFoundError, KeyError) as e:
        print(f'{pert} is not measured in genes. We do not sample that!')














