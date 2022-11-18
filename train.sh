python main.py \
    -t \
    --base configs/stable-diffusion/material.yaml \
    --gpus 0,1 \
    --scale_lr False \
    --num_nodes 1 \
    --check_val_every_n_epoch 10 \
    --finetune_from models/ldm/sd-v1-4.ckpt