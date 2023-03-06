python3 test.py --batch_size 1 \
    --num_worker 12 \
    --seed 42 \
    --data_dir "/media/d5/7D1922F98D178B12/hz/DataSet/mat/data/polyhaven" \
    --load_dir "logs/resnet_csra/version_32/checkpoints/best-epoch=27-val_mAP=85.168.ckpt" \
    --config "logs/resnet_csra/version_32/hparams.yaml" \
    --devices 1