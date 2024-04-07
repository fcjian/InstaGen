import argparse, os
import torch
from mmengine.model import bias_init_with_prob
from mmdet.apis import init_detector, inference_detector
from einops import rearrange
from PIL import Image
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from mmdet.structures import DetDataSample


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(args):
    seed_everything(args.seed)
    cls_names = [name.strip() for name in args.cls_names.split(',')]
    device = torch.device(f"cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    pretrain_detector = init_detector(args.detector_config, args.detector_ckpt, device=device, palette='random')

    text_embeddding_list = []
    for cls_name in cls_names:
        query_text = "a photograph of a " + cls_name
        c_split = pretrain_detector.backbone.model.cond_stage_model.tokenizer.tokenize(query_text)
        sen_text_embedding = pretrain_detector.backbone.model.get_learned_conditioning(query_text)
        class_embedding = sen_text_embedding[:, 5:len(c_split) + 1, :]
        if class_embedding.size()[1] > 1:
            class_embedding = torch.unsqueeze(class_embedding.mean(1), 1)
        text_embeddding_list.append(class_embedding[0])
    text_embeddings = torch.cat(text_embeddding_list)

    c1, c2 = text_embeddings.shape
    if c1 < 2:
        zero_embeddings = text_embeddings.new_zeros(2 - c1, c2)
        text_embeddings = torch.cat([text_embeddings, zero_embeddings], 0)

    prompt = 'a photograph of ' + ' and '.join(cls_names)
    print(f"Prompt: {prompt}")
    batch_data_samples = DetDataSample(
        metainfo=dict(
            prompts = prompt,
            text_embeddings = text_embeddings,
            scale_factor = (1.0, 1.0),
            pad_shape = (512, 512),
            ori_shape = (512, 512),
            batch_input_shape = (512, 512),
            img_shape = (512, 512),
            label_mappings = {0:0, 1:1}
        )
    )

    results = pretrain_detector.predict(None, [batch_data_samples])

    x_sample = torch.clamp((results[0].x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
    img = np.ascontiguousarray(x_sample.astype(np.uint8))

    bboxes = results[0].pred_instances.bboxes
    scores = results[0].pred_instances.scores
    labels = results[0].pred_instances.labels
    for j, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
        if score > args.score_thr:
            img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            img = cv2.putText(img, cls_names[label.item()], (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 0), 2)

    plt.imsave(args.out, img)
    print(f"Save the synthetic images with bounding-boxes: {os.path.join('mmdetection', args.out)}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detector_config",
        type=str,
        default="configs/instagen/instagen-4scale_fd_8xb2-12e_coco_demo.py",
        help="path to config of InstaGen",
    )
    parser.add_argument(
        "--detector_ckpt",
        type=str,
        default="work_dirs/instagen-4scale_fd_8xb2-12e_coco/epoch_12.pth",
        help="path to checkpoint of InstaGen",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=44,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--cls_names",
        type=str,
        default="motorcycle, umbrella",
        help="the class names",
    )
    parser.add_argument(
        "--score_thr",
        type=float,
        default=0.4,
        help="score threshold of the prediction",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="demo/instagen_pred.jpg",
        help="the path to save the demo image",
    )

    args = parser.parse_args()
    main(args)
