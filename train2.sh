python main.py \
    -t \
    --base configs/stable-diffusion/v2-inference.yaml \
    --gpus 0, \
    --scale_lr False \
    --num_nodes 1 \
    --check_val_every_n_epoch 3 \
    --finetune_from models/ldm/512-base-ema.ckpt \
    --max_epochs 100 \
    # --precision 16