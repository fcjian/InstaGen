python generate_class_embedding.py \
    --config configs/stable-diffusion/v1-inference.yaml \
    --ckpt checkpoints/stable-diffusion-v-1-4-original/sd-v1-4.ckpt \
    --classes mmdetection/instagen_resources/coco_classes.txt \
    --save_path mmdetection/instagen_resources/coco_classes.pt
