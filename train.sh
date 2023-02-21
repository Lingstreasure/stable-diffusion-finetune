python main.py \
    -t \
    --base configs/stable-diffusion/material.yaml \
    --gpus 0, \
    --scale_lr False \
    --num_nodes 1 \
    --check_val_every_n_epoch 1 \
    --finetune_from models/ldm/sd-v1-4-full-ema.ckpt \
    --max_epochs 100