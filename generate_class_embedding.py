from mmdet.apis import init_detector, inference_detector
import warnings
from pytorch_lightning import seed_everything
warnings.filterwarnings("ignore")
import argparse, os
import PIL
import torch
from datetime import datetime

import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import torchvision
import random
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import torch.optim as optim
import pickle
from concurrent import futures
from torch import autocast


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    
    for name, parameter in model.named_parameters():
        parameter.requires_grad = False
        
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    return model

def main(args):
    seed_everything(args.seed)

    lines = open(args.classes, "r").readlines()
    cls_names = [line.strip() for line in lines]

    config = OmegaConf.load(args.config)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = load_model_from_config(config, args.ckpt)
    model = model.to(device)

    text_embeddding_list = []
    for cls_name in cls_names:
        query_text="a photograph of a " + cls_name
        c_split = model.cond_stage_model.tokenizer.tokenize(query_text)
        sen_text_embedding = model.get_learned_conditioning(query_text)
        class_embedding = sen_text_embedding[:, 5:len(c_split) + 1, :]
        if class_embedding.size()[1] > 1:
            class_embedding = torch.unsqueeze(class_embedding.mean(1), 1)
        text_embeddding_list.append(class_embedding[0])
    text_embedding = torch.cat(text_embeddding_list)

    torch.save(text_embedding.detach().cpu(), args.save_path)
    print(f"Save text embedding with {text_embedding.shape}: {args.save_path}!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/stable-diffusion-v-1-4-original/sd-v1-4.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="mmdetection/instagen_resources/coco_classes.txt",
        help="path to class names",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="mmdetection/instagen_resources/coco_classes_debug.pt",
        help="path to save the class embeddings",
    )
    
    args = parser.parse_args()
    main(args)
