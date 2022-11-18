python scripts/txt2img.py \
    --prompt "A texture map of black mosaic tile with white squares" \
    --outdir "outputs/generated" \
    --ckpt "logs/2022-10-19T14-00-44_material_v1/checkpoints/last.ckpt" \
    --H 512 --W 512 \
    --n_samples 4 \
    --ddim_steps 50 \
    --config "configs/stable-diffusion/material.yaml"