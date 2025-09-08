"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import argparse
from tqdm import tqdm
import os
import numpy as np
import torch as th
import torch.distributed as dist
import random
import sys
import scanpy as sc

sys.path.append('/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/functions/diffusion/classifier_free_v3_no_saver') # GPU server
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util_classifier_free_v3_acc import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"Using device: {device}")
def save_data(all_cells, traj, data_dir):
    cell_gen = all_cells
    np.savez(data_dir, cell_gen=cell_gen) # a save function in Numpy, saving by key name 'cell_gene'
    return

def main(class_emb_for_one_class, pert_name_for_one_class, folder_path, neighbor):
    """
    :param class_emb_for_one_class: a vector of shape (class_emb, ), needed to be changed into (batch_size, class_emb)
    """
    setup_seed(1234)
    args, unknown = create_argparser().parse_known_args()  # unknown will absorb --host and --port
    # args = create_argparser().parse_args()

    args.num_samples = 30000
    args.batch_size = 10000
    args.eta = 1.5 # you can change the learning rate for sampling here!
    args.all_mask = False # This not sanity checking


    class_emb_for_one_class = th.from_numpy(class_emb_for_one_class)
    class_emb_to_use = class_emb_for_one_class.unsqueeze(0).repeat(args.batch_size, 1).to(th.float32).to(dist_util.dev())

    args.class_input_dim = class_emb_to_use.shape[1]

    args.model_path = f'/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/model/trained_Unet_fullLoop800000_v3_wo_{pert_name_for_one_class}_no_impute_locate/model800000.pt'

    args.sample_dir = folder_path + '/' + neighbor + f'_v3_eta{args.eta}_{args.num_samples}gen_acc_no_impute' + '.npz'

    accel_flag = True

    dist_util.setup_dist()
    print(f'Now we are going to use {dist_util.dev()}')
    logger.configure(dir='checkpoint/sample_logs')

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    print(f'The learning rate of sampling in classifier-free diffusion model is {diffusion.eta}!')
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    print(f'Sampling {neighbor} as the refrence of {pert_name_for_one_class}...')
    all_cells = []
    while len(all_cells) * args.batch_size < args.num_samples:
        model_kwargs = {}
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        ) # please create the same embeddings of class_emb for each specific class before sampling!
        sample, traj = sample_fn(
            model,
            (args.batch_size, args.input_dim), # This input dim is the dimension of latent variables(128 in VAE)
            class_emb_to_use, # of shape (args.batch_size, class_emb_dim)
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            start_time=diffusion.betas.shape[0],
            accel_flag=accel_flag
        ) # acceleration used

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_cells.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(all_cells) * args.batch_size} samples")

    arr = np.concatenate(all_cells, axis=0)
    save_data(arr, traj, args.sample_dir)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=3000,
        batch_size=3000,
        use_ddim=False,
        model_path="output/diffusion_checkpoint/muris_diffusion/model600000.pt",
        sample_dir="output/simulated_samples/muris",
        branch=0,
        cache_interval=5,
        non_uniform=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def setup_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True # 设置随机数种子


if __name__ == "__main__":
    KO_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc/data/embedding_related/'
    KO_whole = np.load(KO_path + 'KO_good_for_PCA.npy')
    KO_pert_name = np.load(KO_path + 'good_pert_name.npy').tolist()

    cluster_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/embedding_related/'
    cluster_dict = np.load(cluster_path + 'cluster_dict_leiden_full.npy', allow_pickle=True).item()
    KO_pert_sum = np.sum(np.abs(KO_whole), axis=1)
    top_num = 14 # If you want to analyze perturbations with top 20 effect sizes one by one...
    top_indices = np.argsort(KO_pert_sum)[-top_num:][::-1]
    top_pert = set([KO_pert_name[index] for index in top_indices])
    pert_to_sample = top_pert | set(cluster_dict['1']) # you can also add perturbations from one specific clusters

    pert_to_sample = sorted(pert_to_sample)

    for pert_name in tqdm(pert_to_sample):
        print(f'Sampling {pert_name}...')
        folder_path = f'/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/sampled_data/wo_{pert_name}_location/'
        if os.path.isdir(folder_path):
            print(f"Skip {pert_name} — Sampling already finished.")
            continue
        try:
            sample_cluster_path = '/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/emb_cluster_wo_one_pert/'
            sample_cluster = np.load(sample_cluster_path + f'cluster_dict_leiden_wo_{pert_name}.npy',
                                 allow_pickle=True).item()
            sample_emb_dict = np.load(sample_cluster_path + f'emb_dict_wo_{pert_name}.npy', allow_pickle=True).item()
            folder_path = f'/home/eshang/diffusion_and_protein/classifier_free/eff_emb_cleary_sub_cluster_saver_top_eff_gene_acc_leiden/data/sampled_data/wo_{pert_name}_location'
            os.makedirs(folder_path, exist_ok=True)

            found_cluster = [key for key, value in sample_cluster.items() if pert_name in value]
            assert len(found_cluster) == 1
            cluster_num = found_cluster[0]
            neighbor_pert = [neighbor for neighbor in sample_cluster[cluster_num] if neighbor != pert_name]
            for neighbor in neighbor_pert:
                print(f'The neighbor of {pert_name} is {neighbor}...')
                # main(class_emb_for_one_class, pert_name_for_one_class, folder_path, neighbor)
                main(sample_emb_dict[neighbor], str(pert_name), folder_path, str(neighbor))
            main(sample_emb_dict[pert_name], str(pert_name), folder_path, str(pert_name))
        except (FileNotFoundError, KeyError) as e:
            print(f'{pert_name} is not measured in genes. We do not sample that!')













