import argparse, os
import cv2
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import mmcv
import random
from tqdm import tqdm


def main(args):
    data_root = args.data_root
    ori_ann_file = args.ori_ann_file
    score_thr = args.score_thr
    allow_empty_gt = args.allow_empty_gt
    top_images = args.top_images

    img_root = os.path.join(data_root, "images")
    ann_root = os.path.join(data_root, "labels")

    if allow_empty_gt:
        out_file = os.path.join(data_root, f"anns_thr{score_thr}_all_images.json")
    else:
        out_file = os.path.join(data_root, f"anns_thr{score_thr}_pos_images.json")

    if top_images > 0:
        out_file = out_file.replace('.json', f'_top{top_images}.json')

    print(f"out file: {out_file}")

    with open(ori_ann_file, 'r') as f:
        load_dict = json.load(f) # dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
    categories = load_dict['categories']
    cat_name2ids = defaultdict()
    for cat in categories:
        cat_name2ids[cat['name']] = cat['id']

    annotations = []
    images = []
    ann_id = 0
    img_id = 0
    class_names = os.listdir(img_root)
    for class_name in tqdm(class_names, desc="classes", total=len(class_names)):
        img_dir = os.path.join(img_root, class_name)
        ann_dir = os.path.join(ann_root, class_name)
        img_names = os.listdir(img_dir)
        
        img_names.sort()
        if top_images > 0:
            img_names = img_names[:top_images]
            
        for img_name in img_names:
            # img_path = os.path.join(img_dir, img_name)
            ann_path = os.path.join(ann_dir, img_name.replace('.jpg', '.txt'))
            
            lines = open(ann_path, 'r').readlines()
            lines = [line.strip() for line in lines]
            
            class_labels = []
            for line in lines:
                label_name = ' '.join(line.split(' ')[:-5])
                x1, y1, x2, y2, score = line.split(' ')[-5:]
                x1, y1, x2, y2, score = float(x1), float(y1), float(x2), float(y2), float(score)
                if score >= score_thr:
                    class_labels.append([label_name, x1, y1, x2, y2, score])
                    
            if not allow_empty_gt and len(class_labels) == 0:
                continue
        
            height, width = 512, 512  # mmcv.imread(img_path).shape[:2]
            images.append(dict(
                id=img_id,
                file_name=os.path.join(class_name, img_name),
                height=height,
                width=width,
                neg_category_ids=[],
                not_exhaustive_category_ids=[]
            ))
            
            for label_name, x1, y1, x2, y2, score in class_labels:
                w = x2 - x1
                h = y2 - y1
                cat_id = cat_name2ids[label_name]
            
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
        categories=categories)
    with open(out_file, 'w') as f:
        json.dump(ann_dict, f)

    print(f"Save ann with #images: {img_id} and #objects: {ann_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default="outputs/coco_images/coco_base_ft6_1248",
        help="path to data",
    )
    parser.add_argument(
        "--ori_ann_file",
        type=str,
        default="mmdetection/data/ori/annotations/instances_val2017_seen.json",
        help="path to ori ann file",
    )
    parser.add_argument(
        "--score_thr",
        type=float,
        default=0.8,
        help="score threshold of the prediction",
    )
    parser.add_argument(
        "--allow_empty_gt",
        action='store_true',
        help="whether to save image without GT",
    )
    parser.add_argument(
        "--top_images",
        type=int,
        default=-1,
        help="only save top images",
    )

    args = parser.parse_args()
    main(args)
