import argparse
import os
import time
from tqdm import tqdm
import numpy as np
import torch
import sys
import scanpy as sc

sys.path.append('/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/functions/diffusion/classifier_free_v3_no_saver/VAE') # GPU server
from VAE_model import VAE
# sys.path.append("..")
# from guided_diffusion.cell_datasets import load_data
# from guided_diffusion.cell_datasets_sapiens import load_data
# from guided_diffusion.cell_datasets_WOT import load_data
# from guided_diffusion.cell_datasets_muris import load_data


sys.path.append('/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/functions/diffusion/classifier_free_v3_no_saver/guided_diffusion') # GPU server
from cell_datasets_loader import load_data # no need to change into cell_datasets_loader_classifier_free since VAE only uses the information from genes, not classes
# from guided_diffusion.cell_datasets_loader import load_data

torch.autograd.set_detect_anomaly(True)
import random

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def prepare_vae(args, state_dict=None):
    """
    Instantiates autoencoder and dataset to run an experiment.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets = load_data(
        data_dir=args["data_dir"],
        batch_size=args["batch_size"],
        train_vae=True,
    )

    autoencoder = VAE(
        num_genes=args["num_genes"],
        device=device,
        seed=args["seed"],
        loss_ae=args["loss_ae"],
        hidden_dim=128,
        decoder_activation=args["decoder_activation"],
    )
    if state_dict is not None:
        print('loading pretrained model from: \n',state_dict)
        use_gpu = device == "cuda"
        autoencoder.encoder.load_state(state_dict["encoder"], use_gpu)
        autoencoder.decoder.load_state(state_dict["decoder"], use_gpu)

    return autoencoder, datasets


def train_vae(args, return_model=False):
    """
    Trains a autoencoder
    """
    if args["state_dict"] is not None:
        filenames = {}
        checkpoint_path = {
            "encoder": os.path.join(
                args["state_dict"], filenames.get("model", "encoder.ckpt")
            ),
            "decoder": os.path.join(
                args["state_dict"], filenames.get("model", "decoder.ckpt")
            ),
            "gene_order": os.path.join(
                args["state_dict"], filenames.get("gene_order", "gene_order.tsv")
            ),
        }
        autoencoder, datasets = prepare_vae(args, checkpoint_path)
    else:
        autoencoder, datasets = prepare_vae(args)
   
    args["hparams"] = autoencoder.hparams

    start_time = time.time()
    print(f'Training VAE for at most {args["max_steps"]} and at most {args["max_minutes"]} minutes')
    print(f'We record model parameters every {args["checkpoint_freq"]}!')
    vae_loss = []
    for step in tqdm(range(args["max_steps"])):

        genes, _ = next(datasets) # only use genes, nothing to do with the classes or conditions

        genes = genes.float()

        minibatch_training_stats = autoencoder.train(genes)

        # print(f'The VAE loss is {minibatch_training_stats.values()} in Step {step}')

        vae_loss.append(minibatch_training_stats['loss_reconstruction'])

        if step % 1000 == 0:
            for key, val in minibatch_training_stats.items():
                print('step ', step, 'loss ', val)

        ellapsed_minutes = (time.time() - start_time) / 60

        stop = ellapsed_minutes > args["max_minutes"] or (
            step == args["max_steps"] - 1
        )

        if ((step % args["checkpoint_freq"]) == 0 or stop):

            os.makedirs(args["save_dir"],exist_ok=True)
            print(f'We save the VAE model record at Step {step}! The Check Frequency is {args["checkpoint_freq"]}.')
            torch.save(
                autoencoder.state_dict(),
                os.path.join(
                    args["save_dir"],
                    "model_seed={}_step={}.pt".format(args["seed"], step),
                ),
            )

            if stop:
                break

    if return_model: # return_model is set to be False by default
        return autoencoder, datasets
    return vae_loss


def parse_arguments():
    """
    Read arguments if this script is called from a terminal.
    """

    parser = argparse.ArgumentParser(description="Finetune Scimilarity")
    # dataset arguments
    parser.add_argument("--data_dir", type=str, default='/data1/lep/Workspace/guided-diffusion/data/tabula_muris/all.h5ad')
    parser.add_argument("--loss_ae", type=str, default="mse")
    parser.add_argument("--decoder_activation", type=str, default="ReLU")

    # AE arguments                                             
    parser.add_argument("--local_rank", type=int, default=0)  
    parser.add_argument("--split_seed", type=int, default=1234)
    parser.add_argument("--num_genes", type=int, default=18996)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hparams", type=str, default="")

    # training arguments
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--max_minutes", type=int, default=3000)
    parser.add_argument("--checkpoint_freq", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--state_dict", type=str, default="/data1/lep/Workspace/guided-diffusion/scimilarity-main/models/annotation_model_v1")  # if pretrain
    # parser.add_argument("--state_dict", type=str, default=None)   # if not pretrain

    parser.add_argument("--save_dir", type=str, default='../output/ae_checkpoint/muris_AE')
    parser.add_argument("--sweep_seeds", type=int, default=200)
    args, unknown = parser.parse_known_args() # my code
    return dict(vars(args))

    # return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    KO_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc/data/embedding_related/'
    KO_whole = np.load(KO_path + 'KO_good_for_PCA.npy')
    KO_pert_name = np.load(KO_path + 'good_pert_name.npy').tolist()

    dict_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/embedding_related/'
    adata_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/embedding_related/'
    adata_all = sc.read_h5ad(adata_path + 'adata_no_imputed.h5ad')

    cluster_dict = np.load(dict_path+'cluster_dict_leiden_full.npy', allow_pickle=True).item()
    KO_pert_sum = np.sum(np.abs(KO_whole), axis=1)
    top_num = 20 # If you want to analyze perturbations with top 20 effect sizes one by one...
    top_indices = np.argsort(KO_pert_sum)[-top_num:][::-1]
    top_pert = set([KO_pert_name[index] for index in top_indices])
    pert_to_sample = top_pert | set(cluster_dict['1']) # you can also add perturbations from one specific clusters

    pert_to_sample = sorted(pert_to_sample)
    save_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/embedding_related/'
    print(f'All perturbations are {pert_to_sample}!')
    for pert in pert_to_sample:
        print(f'Evaluate {pert}...')
        folder_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/model/VAE_wo_' + f'{pert}_no_impute_locate/'
        if os.path.isdir(folder_path):
            print(f"Skip {pert} — VAE model already trained.")
            continue
        try:
            sample_cluster_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/emb_cluster_wo_one_pert/'
            sample_cluster = np.load(sample_cluster_path + f'cluster_dict_leiden_wo_{pert}.npy', allow_pickle=True).item()
            print(f'Train VAE when holding out {pert}!')
            seed_everything(1234)
            args = parse_arguments()
            args['state_dict'] = '/home/eshang/diffusion_and_protein/scDiffusion-main/guided_diffusion/annotation_model_v1'  # path to prestrained auto VAE
            args['num_genes'] = 1500  # number after the filter, not the most most original one(36601) or (13714), 18996 for data in paper of scDiffusion, 4495 for norman

            args['save_dir'] = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/model/VAE_wo_' + f'{pert}_no_impute_locate/'

            adata_wo_pert = adata_all[adata_all.obs['celltype'] != pert].copy()
            adata_wo_pert.write_h5ad(save_path + f'adata_wo_{pert}_no_impute.h5ad')

            args['data_dir'] = f'/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/embedding_related/adata_wo_{pert}_no_impute.h5ad'

            loss_rec_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/loss_related/'

            vae_loss = train_vae(args)
            np.save(loss_rec_path + f'vae_loss_wo_{pert}_no_impute.npy', vae_loss)
            print("The trajectory of VAE loss is saved!")
            print(f'vae_loss is {vae_loss[len(vae_loss) - 1]} for {pert}!')
        except (FileNotFoundError, KeyError) as e:
            print(f'{pert} is not measured in genes. We do not train VAE for that!')





