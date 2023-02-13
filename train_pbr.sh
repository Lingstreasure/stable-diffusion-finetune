python train_pbr.py \
    --train \
    --name all \
    --postfix _v2 \
    --base configs/stable-diffusion/pbr_improve.yaml \
    --gpus 1, \
    --scale_lr False \
    --num_nodes 1 \
    --check_val_every_n_epoch 3 \
    --finetune_from models/ldm/512-base-ema.ckpt \
    # --auto_scale_batch_size True 