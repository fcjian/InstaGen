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
from ldm.modules.diffusionmodules.openaimodel import clear_feature_dic, get_feature_dic
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

def main(first_classes, second_classes, detecotr_classes, base_classes, cuda_id, args):
    print(f"cuda_id={cuda_id}, first_classes={first_classes}")

    seed_everything(args.seed)
    device = torch.device(f"cuda:{cuda_id}") if torch.cuda.is_available() else torch.device("cpu")

    print('***********************   loading diffusion model   **********************************')
    config = OmegaConf.load(args.config)
    config.model['params']['cond_stage_config']['params'] = {"device": f"cuda:{cuda_id}"}
    model = load_model_from_config(config, args.ckpt)
    model = model.to(device)

    sampler = DDIMSampler(model)

    print('***********************   loading detector   **********************************')
    pretrain_detector = init_detector(args.detector_config, args.detector_ckpt, device=device)

    print('***********************   begin   **********************************')    
    batch_size = args.n_samples
    num_images = args.num_images
    save_path = args.save_path
    print(f'batch size: {batch_size}')
    print(f'num images: {num_images}')
    print(f'save path: {save_path}')

    start_code = None
    if args.fixed_code:
        print('start_code')
        start_code = torch.randn([args.n_samples, args.C, args.H // args.f, args.W // args.f], device=device)  

    # importance for ema
    for name, param in model.model.named_parameters():
        param.requires_grad = True

    with torch.no_grad():
        with model.ema_scope():
            for select_class_index1 in range(len(first_classes)):
                select_class1 = first_classes[select_class_index1]
                print(f"now generating images of the class: {select_class1}")
                save_img_path = os.path.join(save_path, "images", select_class1)
                save_label_path = os.path.join(save_path, "labels", select_class1)
                save_feat_path = os.path.join(save_path, "features", select_class1)
                save_prompt_path = os.path.join(save_path, "prompts", select_class1)
                if os.path.exists(save_img_path):
                    continue
                os.makedirs(save_img_path)
                os.makedirs(save_label_path)
                os.makedirs(save_feat_path)
                os.makedirs(save_prompt_path)

                for batch_index in tqdm(range((num_images + batch_size - 1) // batch_size), \
                        desc=f"cuda: {cuda_id}, classes: {select_class_index1 + 1}/{len(first_classes)}"):
                    prompts = []
                    batch_class_list = []
                    for img_index in range(batch_size):
                        class_list = []

                        single_or_two = random.randint(0, 1)
                        if single_or_two == 0:
                            class_list.append(select_class1)
                            prompt = f"a photograph of {select_class1}"
                        elif single_or_two == 1:
                            select_class_index2 = random.randint(0, len(second_classes)-1)
                            select_class2 = second_classes[select_class_index2]
                            while select_class1 == select_class2:
                                select_class_index2 = random.randint(0, len(second_classes)-1)
                                select_class2=second_classes[select_class_index2]
                            class_list.append(select_class1)
                            class_list.append(select_class2)
                            prompt = f"a photograph of {select_class1} and {select_class2}"

                        prompts.append(prompt)
                        batch_class_list.append(class_list)

                    clear_feature_dic()

                    uc = None
                    if args.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    
                    c = model.get_learned_conditioning(prompts)
                    shape = [args.C, args.H // args.f, args.W // args.f]
                    latent_feat = [None]
                    samples_ddim, _ = sampler.sample(S=args.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=args.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=args.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=args.ddim_eta,
                                                        x_T=start_code,
                                                        latent_feat=latent_feat,
                                                        print_flag=False)
                    
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    
                    x_sample_list=[]
                    for img_index in range(x_samples_ddim.size()[0]):
                        x_sample = torch.clamp((x_samples_ddim[img_index] + 1.0) / 2.0, min=0.0, max=1.0)
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        x_sample_list.append(x_sample)
    
                    det_result_list=[]
                    result = inference_detector(pretrain_detector, x_sample_list)
                    for img_index in range(len(result)):
                        result_img = result[img_index]
                        bboxes = result_img.pred_instances.bboxes
                        labels = result_img.pred_instances.labels
                        scores = result_img.pred_instances.scores
                        det_result = [[] for _ in range(len(detecotr_classes))]
                        for label, bbox, score in zip(labels, bboxes, scores):
                            det_result[label].append(
                                np.concatenate([bbox.detach().cpu().numpy(), score.detach().cpu().numpy()[None]])
                            )
                        det_result_list.append(det_result)

                    for img_index in range(batch_size):
                        anns = []
                        class_list = batch_class_list[img_index]
                        for class_name in class_list:
                            if class_name in base_classes:
                                class_index = detecotr_classes[class_name]
                                                        
                                if len(det_result_list[img_index][class_index]) == 0:
                                    print("detector fail to detect the object in the class:", class_name)
                                else:
                                    score_thr = 0.
                                    for bbox in det_result_list[img_index][class_index]:
                                        if score > score_thr:
                                            anns.append(f"{class_name} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")

                        # get image index
                        index = batch_index * batch_size + img_index
                        if index >= num_images:
                            continue

                        # save image
                        img_path = os.path.join(save_img_path, f'{str(index).zfill(6)}.jpg')
                        Image.fromarray(x_sample_list[img_index].astype(np.uint8)).save(img_path)

                        # save label
                        ann_path = os.path.join(save_label_path, f'{str(index).zfill(6)}.txt')
                        if len(anns) > 0:
                            anns[-1] = anns[-1].strip()
                            fw = open(ann_path, 'w')
                            fw.writelines(anns)
                            fw.close()
                        else:
                            fw = open(ann_path, 'w')
                            fw.close()

                        # save feat
                        feat_path = os.path.join(save_feat_path, f'{str(index).zfill(6)}.pt')
                        torch.save(latent_feat[0][img_index].detach().cpu(), feat_path)

                        # save prompt
                        prompt_path = os.path.join(save_prompt_path, f'{str(index).zfill(6)}.txt')
                        fw = open(prompt_path, 'w')
                        fw.writelines([prompts[img_index]])
                        fw.close()

    return f"cuda {cuda_id} finished!!!"

                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="the save path",
        default=""
    )
    parser.add_argument(
        "--detector_classes",
        type=str,
        help="detection categories of the pre-trained detector.",
        default="mmdetection/demo/coco_classes.txt"
    )
    parser.add_argument(
        "--base_classes",
        type=str,
        help="the base classes",
        default=""
    )
    parser.add_argument(
        "--first_classes",
        type=str,
        help="generate images of part classes.",
        default=""
    )
    parser.add_argument(
        "--second_classes",
        type=str,
        help="generate images of part classes2.",
        default=""
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1250,
        help="the number of images per class",
    )
    parser.add_argument(
        "--detector_config",
        type=str,
        default="mmdetection/configs/swin/mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco.py",
        help="path to config of the pre-trained detecotr",
    )
    parser.add_argument(
        "--detector_ckpt",
        type=str,
        default="mmdetection/checkpoints/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth",
        help="path to checkpoint of the pre-trained detecotr",
    )
    parser.add_argument(
        "--num_thread",
        type=int,
        default=8,
        help="the number of thread/gpus for generating images",
    )
    
    args = parser.parse_args()

    detecotr_classes = {} 
    f = open(args.detector_classes, "r")
    class_indes = 0
    for line in f.readlines():
        class_name = line.split("\n")[0]
        detecotr_classes[class_name] = class_indes
        class_indes += 1
    print(f"detecotr classes: {len(detecotr_classes)} \n", detecotr_classes)

    if len(args.base_classes) > 0:
        f = open(args.base_classes, "r")
        lines = f.readlines()
        base_classes = [line.strip() for line in lines]
    else:
        base_classes = list(detecotr_classes.keys())
    print(f"base classes: {len(base_classes)} \n", base_classes)

    f = open(args.first_classes, "r")
    lines = f.readlines()
    first_classes = [line.strip() for line in lines]
    print(f"first classes: {len(first_classes)} \n", first_classes)

    f = open(args.second_classes, "r")
    lines = f.readlines()
    second_classes = [line.strip() for line in lines]
    print(f"second_classes: {len(second_classes)} \n", second_classes)

    num_thread = args.num_thread
    if num_thread == 1:
        cuda_id = 0
        main(first_classes=first_classes, second_classes=second_classes, detecotr_classes=detecotr_classes, \
             base_classes=base_classes, cuda_id=cuda_id, args=args)
    else:
        cuda_ids = [i for i in range(args.num_thread)]
        num_classes_thread = (len(first_classes) + num_thread - 1) // num_thread
        first_classes_list = [first_classes[i * num_classes_thread:(i + 1) * num_classes_thread] for i in range(num_thread)]
        print(f'all first classes: {len(first_classes)}, num thread: {num_thread}, number of images per class: {args.num_images}')

        with futures.ThreadPoolExecutor(max_workers=num_thread) as executor:
            threads = [executor.submit(main, first_classes, second_classes, detecotr_classes, base_classes, cuda_id, args) \
                    for cuda_id, first_classes in zip(cuda_ids, first_classes_list)]
            for future in futures.as_completed(threads):
                print(future.result())
