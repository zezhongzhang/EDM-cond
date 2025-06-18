# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop_cond
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.


@hydra.main(config_path="configs", config_name="main", version_base="1.3")
def main(c: DictConfig):
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    

    # Validate dataset options.
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        # c.dataset_kwargs.resolution = dataset_obj.img_res_w # be explicit about dataset resolution
        c.dataset_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')


    # Training options.
    c.total_kimg = max(int(c.duration * 1000), 1)
    c.ema_halflife_kimg = int(c.ema * 1000)


    # Random seed.
    if c.seed is None:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer learning and resume.
    if c.transfer is not None:
        if c.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = c.transfer
        c.ema_rampup_ratio = None
    elif c.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(c.resume))
        if not match or not os.path.isfile(c.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(c.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = c.resume

    # Description string.
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name:s}-{c.network_kwargs.model_name:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
    if c.desc is not None:
        desc += f'-{c.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif c.nosubdir:
        c.run_dir = c.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(c.outdir):
            prev_run_dirs = [x for x in os.listdir(c.outdir) if os.path.isdir(os.path.join(c.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(c.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(OmegaConf.to_yaml(c, resolve=True))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Network architecture:    {c.network_kwargs.model_name}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()




    # Dry run?
    if c.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)

        # save all the config 
        OmegaConf.save(c, os.path.join(c.run_dir, "training_config.yaml"))

        # Save CLI overrides (e.g., "dataset_kwargs=testdata network_kwargs=ddpmpp")
        overrides = HydraConfig.get().overrides.task
        with open(os.path.join(c.run_dir, 'CLI_overrides.txt'), 'wt') as f:
            for o in overrides:
                f.write(f"{o} ")
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    training_loop_cond.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
