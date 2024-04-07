import argparse, os
import cv2
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import mmcv
import random
from tqdm import tqdm
import pickle


def main(args):
    data_root = args.data_root
    base_ann_file = args.base_ann_file
    pred_file = args.pred_file
    score_thr = args.score_thr

    img_root = os.path.join(data_root, "images")

    out_file = os.path.join(data_root, f"anns_base_and_novel_thr{score_thr}.json")
    print(f"out file: {out_file}")

    with open(base_ann_file, 'r') as f:
        base_dict = json.load(f)  # dict_keys(['images', 'annotations', 'categories'])
    with open(pred_file, 'rb') as f:
        instagen_preds = pickle.load(f)
    assert len(base_dict['images']) == len(instagen_preds)

    img_id2anns = defaultdict(list)
    for ann in base_dict['annotations']:
        img_id2anns[ann['image_id']].append([ann['category_id']] + ann['bbox'])

    cat_ind2id = defaultdict()
    for ind, cat in enumerate(base_dict['categories']):
        cat_ind2id[ind] = cat['id']

    line = open(args.novel_ind_file, 'r').readlines()[0]
    novel_inds = [int(char) for char in line.split(', ')]

    annotations = []
    images = []
    ann_id = 0 
    img_id = 0
    for i, image in tqdm(enumerate(base_dict['images']), desc="images", total=len(base_dict['images'])):
        base_anns = img_id2anns[image['id']]

        novel_anns = []
        bboxes = instagen_preds[i]['pred_instances']['bboxes']
        scores = instagen_preds[i]['pred_instances']['scores']
        labels = instagen_preds[i]['pred_instances']['labels']
        for j, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
            cat_ind = label.item()
            cat_id = cat_ind2id[cat_ind]
            if (score > score_thr) and (cat_ind in novel_inds):
                novel_anns.append([cat_id, bbox[0].item(), bbox[1].item(), \
                    (bbox[2] - bbox[0]).item(), (bbox[3] - bbox[1]).item()])
        
        if len(novel_anns) > 0:
            images.append(dict(
                id=img_id,
                file_name=image['file_name'],
                height=image['height'],
                width=image['width'],
                neg_category_ids=[],
                not_exhaustive_category_ids=[]
            ))

            for cat_id, x1, y1, w, h in base_anns + novel_anns:
                data_anno = dict(
                    image_id=img_id,
                    id=ann_id,
                    category_id=cat_id,
                    bbox=[x1, y1, w, h], # attention
                    area=w * h,
                    segmentation=[[0.0, 233.17, 8.77, 216.13, 15.1, 210.42, 15.78, 203.45, 50.51, 139.62, 73.32, 94.43, 79.34, 90.25, 81.05, 83.41, 119.71, 13.39, 128.08, 9.75, 132.53, 7.77, 133.78, 0.0, 127.49, 0.0, 116.23, 2.83, 112.86, 10.2, 74.89, 75.45, 71.21, 79.73, 69.27, 84.76, 65.72, 95.15, 50.98, 121.55, 26.3, 167.02, 8.18, 197.33, 4.81, 201.06, 0.0, 213.25, 0.0, 233.17]],
                    iscrowd=0)
                annotations.append(data_anno)

                ann_id += 1
                
            img_id += 1

    ann_dict = dict(
        images=images,
        annotations=annotations,
        categories=base_dict['categories'])
    with open(out_file, 'w') as f:
        json.dump(ann_dict, f)

    print(f"Save ann with #images: {img_id} and #objects: {ann_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default="outputs/coco_images/coco_novel_ft6_3000",
        help="path to data",
    )
    parser.add_argument(
        "--base_ann_file",
        type=str,
        default="outputs/coco_images/coco_novel_ft6_3000/anns_thr0.8_all_images.json",
        help="path to ann file of base classes",
    )
    parser.add_argument(
        "--pred_file",
        type=str,
        default="",
        help="path to predicted file (pkl)",
    )
    parser.add_argument(
        "--novel_ind_file",
        type=str,
        default="instagen_resources/coco_novel_inds.txt",
        help="path to class index of novel classes",
    )
    parser.add_argument(
        "--score_thr",
        type=float,
        default=0.4,
        help="score threshold of the prediction",
    )

    args = parser.parse_args()
    main(args)
