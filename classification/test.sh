python3 test.py --batch_size 1 \
    --num_worker 12 \
    --seed 23 \
    --data_dir "/root/hz/DataSet/mat/train" \
    --load_dir 'logs/resnet_csra/version_23/checkpoints/best-epoch=35-val_mAP=80.267.ckpt' \
    --config 'logs/resnet_csra/version_23/hparams.yaml' \
    --devices 3