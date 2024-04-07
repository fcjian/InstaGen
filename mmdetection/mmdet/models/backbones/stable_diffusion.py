# mmdet
import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.registry import MODELS

# diffusion
import argparse, os
import cv2
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules.openaimodel import clear_feature_dic,get_feature_dic


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model

@MODELS.register_module()
class StableDiffusion(BaseModule):
    def __init__(self, 
                 config, 
                 sd_ckpt,
                 last_sample=False,
                 scale=5.0,
                 ddim_steps=50,
                 ddim_eta=0.0,
                 precision="autocast",
                 cumprod_file='sd_resources/alphas_cumprod.npy',
                 return_img=False,
                 init_cfg=None):
        super(StableDiffusion, self).__init__(init_cfg)

        config = OmegaConf.load(f"{config}")
        self.model = load_model_from_config(config, f"{sd_ckpt}")
        self.sampler = DDIMSampler(self.model)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.C = 4
        self.H = 512
        self.W = 512
        self.f = 8
        self.scale = scale
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.precision = precision
        self.last_sample = last_sample
        self.return_img = return_img

        self.alphas_cumprod = torch.Tensor(np.load(cumprod_file))

    def forward(self, x=None, data_samples=None):
        batch_size = len(data_samples)

        model = self.model
        sampler = self.sampler
        device = self.device
        precision_scope = autocast if self.precision=="autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    if self.last_sample:
                        start_code = torch.stack([data_samples[i].diffusion_feats for i in range(batch_size)]).to(device)
                    else:
                        start_code = torch.randn([batch_size, self.C, self.H // self.f, self.W // self.f], device=device)

                    prompts = [data_samples[i].prompts for i in range(batch_size)]

                    clear_feature_dic()
                    if self.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    c = model.get_learned_conditioning(prompts)
                    shape = [self.C, self.H // self.f, self.W // self.f]
                    samples_ddim, _ = sampler.sample(S=self.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=batch_size,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=self.scale,
                                                    unconditional_conditioning=uc,
                                                    eta=self.ddim_eta,
                                                    x_T=start_code,
                                                    last_sample=self.last_sample,
                                                    print_flag=False)

                    diffusion_features = get_feature_dic()
                    diffusion_features = [torch.cat(diffusion_features[key], 1).float() for key in ['highest', 'high', 'mid', 'low']]

                    if self.return_img:
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        for data_sample, x_sample in zip(data_samples, x_samples_ddim):
                            data_sample.x_samples_ddim = x_sample

                    return diffusion_features
