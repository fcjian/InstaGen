export TORCH_HOME=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/fengchengjian/Init-models/torch
export TRANSFORMERS_CACHE=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/fengchengjian/cache/huggingface/hub

# base
python generate_image.py \
    --config configs/stable-diffusion/v1-inference_ema.yaml \
    --save_path outputs/coco_ovd_images/coco_base_ft6_1250 \
    --n_samples 4 \
    --detector_classes mmdetection/instagen_resources/coco_classes.txt \
    --first_classes mmdetection/instagen_resources/coco_base_classes.txt \
    --second_classes mmdetection/instagen_resources/coco_base_classes.txt \
    --base_classes mmdetection/instagen_resources/coco_base_classes.txt \
    --num_images 1250 \
    --num_thread 8 \
    --ckpt checkpoints/sd-finetune_coco_base_epoch=000005.ckpt 

# novel
python generate_image.py \
    --config configs/stable-diffusion/v1-inference_ema.yaml \
    --save_path outputs/coco_ovd_images/coco_novel_ft6_3000 \
    --n_samples 4 \
    --detector_classes mmdetection/instagen_resources/coco_classes.txt \
    --first_classes mmdetection/instagen_resources/coco_novel_classes.txt \
    --second_classes mmdetection/instagen_resources/coco_classes.txt \
    --base_classes mmdetection/instagen_resources/coco_base_classes.txt \
    --num_images 3000 \
    --num_thread 8 \
    --ckpt checkpoints/sd-finetune_coco_base_epoch=000005.ckpt 

# only for val
python generate_image.py \
    --config configs/stable-diffusion/v1-inference_ema.yaml \
    --save_path outputs/coco_ovd_images/coco_val_ft6_200 \
    --n_samples 4 \
    --detector_classes mmdetection/instagen_resources/coco_classes.txt \
    --first_classes mmdetection/instagen_resources/coco_novel_classes.txt \
    --second_classes mmdetection/instagen_resources/coco_classes.txt \
    --base_classes mmdetection/instagen_resources/coco_classes.txt \
    --num_images 200 \
    --num_thread 8 \
    --ckpt checkpoints/sd-finetune_coco_base_epoch=000005.ckpt 
