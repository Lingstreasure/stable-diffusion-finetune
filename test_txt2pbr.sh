python scripts/txt2pbr.py \
    --prompt "parquet wood floor" \
    --outdir "outputs/txt2pbr" \
    --diff_ckpt "logs/old/2023-02-03T05-24-23_v2-inference/checkpoints/last.ckpt" \
    --albedo_ckpt "logs/old/2022-11-17T17-01-51_albedo_v1/checkpoints/last.ckpt" \
    --normal_ckpt "logs/old/2022-11-17T22-40-12_normal_v1/checkpoints/last.ckpt" \
    --H 512 --W 512 \
    --n_samples 4 \
    --ddim_steps 25 \
    --config "configs/stable-diffusion/txt2pbr.yaml" \
    --dpm_solver \
    --scale 3.0 \
    --device_num 1 \
    --seed 42 \