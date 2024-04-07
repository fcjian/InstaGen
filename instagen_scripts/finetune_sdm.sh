python main.py \
    -t \
    --base configs/sd-finetune/coco_base.yaml \
    --gpus 0,1,2,3,4,5,6,7 \
    --scale_lr False \
    --num_nodes 1 \
    --check_val_every_n_epoch 1 \
    --finetune_from checkpoints/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt 
