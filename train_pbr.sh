python train_pbr.py \
    --train \
    --name all \
    --postfix _v2 \
    --base configs/stable-diffusion/pbr_improve.yaml \
    --gpus 0,1 \
    --scale_lr False \
    --num_nodes 1 \
    --check_val_every_n_epoch 1 \
    --finetune_from models/ldm/512-base-ema.ckpt \
    --max_epochs 200
    # --auto_scale_batch_size True 