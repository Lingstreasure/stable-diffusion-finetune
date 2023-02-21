python train_ae.py \
    --train \
    --name ae \
    --base configs/autoencoder/autoencoder_kl_32x32x4.yaml \
    --gpus 0,1 \
    --scale_lr False \
    --num_nodes 1 \
    --check_val_every_n_epoch 1 \
    --finetune_from models/ldm/512-base-ema.ckpt \
    --max_epochs 100
    # --auto_scale_batch_size True 