import argparse
import inspect

from . import gaussian_diffusion_classifier_free_v3_acc as gd
from .respace_classifier_free_v3_acc import SpacedDiffusion, space_timesteps
from .cell_model_classifier_free_v3 import Cell_Unet_classifier_free

NUM_CLASSES = 3


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        eta=1.5,
        all_mask=False, # only during sampling and only during sanity checking turned to be True
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        class_cond=False,
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        input_dim = 128,
        class_input_dim = 64,
        hidden_num = [512,512,256,128],
        dropout = 0.0,
        mask_prob=0.2,
        branch=0,
        cache_interval=5,
        non_uniform=False
    )
    res.update(diffusion_defaults())
    return res



def create_model_and_diffusion(
    input_dim,
    class_input_dim,
    hidden_num,
    eta,
    all_mask,
    learn_sigma,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    dropout,
    cache_interval=None,
    non_uniform=None,
    branch=None,
    **kwargs,
):
    model = create_model(
        input_dim,
        class_input_dim,
        hidden_num,
        dropout=dropout,
        cache_interval=cache_interval,
        non_uniform=non_uniform,
        branch=branch
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        eta=eta,
        all_mask=all_mask,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing
    )
    return model, diffusion


def create_model(
    input_dim,
    class_input_dim,
    hidden_num,
    dropout,
    cache_interval,
    non_uniform,
    branch
):

    return Cell_Unet_classifier_free(
        input_dim=input_dim,
        class_input_dim=class_input_dim,
        hidden_num=hidden_num,
        dropout=dropout,
        cache_interval=cache_interval,
        non_uniform=non_uniform,
        branch=branch
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    eta=1.5,
    all_mask=False,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        eta=eta,
        all_mask=all_mask,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
