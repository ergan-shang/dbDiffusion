"""
Train a diffusion model on images.
"""

import argparse
import sys
import scanpy as sc

sys.path.append('/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/functions/diffusion/classifier_free_v3_no_saver') # GPU server
from guided_diffusion import dist_util, logger
from guided_diffusion.cell_datasets_loader_classifier_free import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util_classifier_free_v3 import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util_classifier_free_fullLoop import TrainLoop

import torch
import numpy as np
import random

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main(pert):
    setup_seed(1234)
    args, unknown = create_argparser().parse_known_args()

    args.tol = 0.01
    args.reltol = 0.0001
    args.order = 0.01
    args.log_interval = 100

    args.all_mask = False # this is training code, we set all_mask false
    args.mask_prob = 0.2

    class_emb_dir = f"/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/embedding_related/emb_mask_{pert}.npy"

    args.data_dir = f'/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/embedding_related/adata_wo_{pert}_no_impute.h5ad' # GPU server

    args.vae_path = f'/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/model/VAE_wo_{pert}_no_impute_locate/model_seed=0_step=199999.pt'

    args.model_name = f'trained_Unet_fullLoop800000_v3_wo_{pert}_no_impute_locate'

    args.save_dir = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/model/'

    loss_store_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/loss_related/'

    dist_util.setup_dist()
    # logger.configure(dir='../output/logs/'+args.model_name)  # log file
    logger.configure(dir=args.save_dir + '/' + args.model_name)  # log file

    logger.log("loading the class embedding for all cell data...")
    all_class_emb = np.load(class_emb_dir)  # a matrix for n cells, the second dimension is the embedding dimension for class
    args.class_input_dim = all_class_emb.shape[1]

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    print(f"Now we are going to use {dist_util.dev()}")
    model.to(dist_util.dev())
    # There is no "model.train()" in the original code.....
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        all_class_emb=all_class_emb,
        vae_path=args.vae_path,
        train_vae=False,
    ) # Here we use VAE to transform data

    logger.log("training...")
    print(f"Total steps for training is {args.lr_anneal_steps} and save model every {args.save_interval} steps!")
    all_loss_rec_scDiffusion = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        model_name=args.model_name,
        save_dir=args.save_dir
    ).run_loop()
    print("The trajectory of losses is saved!")
    np.save(loss_store_path + f'all_loss_rec_classifier_free_fullLoop80_v3_wo_{pert}_no_impute.npy', all_loss_rec_scDiffusion)


def create_argparser():
    defaults = dict(
        data_dir="/data1/lep/Workspace/guided-diffusion/data/tabula_muris/all.h5ad",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0001,
        lr_anneal_steps=800000, # 80，0000
        batch_size=128,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=200000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        vae_path = 'output/Autoencoder_checkpoint/muris_AE/model_seed=0_step=0.pt',
        model_name="muris_diffusion",
        save_dir='output/diffusion_checkpoint',
        branch=0,
        cache_interval=5,
        non_uniform=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    KO_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc/data/embedding_related/'
    KO_whole = np.load(KO_path + 'KO_good_for_PCA.npy')
    KO_pert_name = np.load(KO_path + 'good_pert_name.npy').tolist()

    dict_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/embedding_related/'
    cluster_dict = np.load(dict_path + 'cluster_dict_leiden_full.npy', allow_pickle=True).item()
    KO_pert_sum = np.sum(np.abs(KO_whole), axis=1)
    top_num = 20 # If you want to analyze perturbations with top 20 effect sizes one by one...
    top_indices = np.argsort(KO_pert_sum)[-top_num:][::-1]
    top_pert = set([KO_pert_name[index] for index in top_indices])
    pert_to_sample = top_pert | set(cluster_dict['1']) # you can also add perturbations from one specific clusters

    pert_to_sample = sorted(pert_to_sample)

    adata_mask_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/embedding_related/'

    emb_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/emb_cluster_wo_one_pert/'

    save_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/embedding_related/'
    print(f'All perturbations are {pert_to_sample}!')
    for pert in pert_to_sample:
        print(f'Evaluate {pert}...')
        folder_path = f'/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/model/trained_Unet_fullLoop800000_v3_wo_{pert}_no_impute_locate/'
        if os.path.isdir(folder_path):
            print(f"Skip {pert} — Diffusion model already trained.")
            continue
        try:
            emb_dict = np.load(emb_path + f'emb_dict_wo_{pert}.npy', allow_pickle=True).item()
            adata_use = sc.read_h5ad(adata_mask_path + f'adata_wo_{pert}_no_impute.h5ad')
            dim = len(list(emb_dict.values())[0])
            emb_pert = np.zeros((len(adata_use), dim))
            ############################ Create the Embedding for one pert ############################################
            for i in range(len(emb_pert)):
                pert_name = adata_use.obs['celltype'].values[i]
                assert pert_name in emb_dict.keys()
                emb_pert[i] = emb_dict[pert_name]
            np.save(save_path + f'emb_mask_{pert}.npy', emb_pert)
            ############################################################################################################
            print(f'Begin training diffusion for {pert}...')
            main(pert)
        except (FileNotFoundError, KeyError) as e:
            print(f'{pert} is not measured in genes. We do not train the diffusion model for {pert}!')


