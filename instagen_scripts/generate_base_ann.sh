# base
python generate_base_ann.py \
    --data_root outputs/coco_ovd_images/coco_base_ft6_1250 \
    --ori_ann_file mmdetection/data/coco/annotations/instances_val2017_seen.json \
    --score_thr 0.8

# novel, for training InstaGen (grounding head)
python generate_base_ann.py \
    --data_root outputs/coco_ovd_images/coco_novel_ft6_3000 \
    --ori_ann_file mmdetection/data/coco/annotations/instances_val2017_seen.json \
    --score_thr 0.8 \
    --top_images 1250 \
    --allow_empty_gt

# only for val
python generate_base_ann.py \
    --data_root outputs/coco_ovd_images/coco_val_ft6_200 \
    --ori_ann_file mmdetection/data/coco/annotations/instances_val2017_seen.json \
    --score_thr 0.8

# novel, for predicting pseudo-labels
python generate_base_ann.py \
    --data_root outputs/coco_ovd_images/coco_novel_ft6_3000 \
    --ori_ann_file mmdetection/data/coco/annotations/instances_val2017_seen.json \
    --score_thr 0.8 \
    --allow_empty_gt
