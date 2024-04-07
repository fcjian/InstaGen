python instagen_scripts/generate_novel_ann.py \
    --data_root outputs/coco_ovd_images/coco_novel_ft6_3000 \
    --base_ann_file outputs/coco_ovd_images/coco_novel_ft6_3000/anns_thr0.8_all_images.json \
    --pred_file mmdetection/outputs/coco_novel_ft6_3000.pkl \
    --novel_ind_file mmdetection/instagen_resources/coco_novel_inds.txt \
    --score_thr 0.4

