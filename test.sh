python scripts/txt2img.py \
    --prompt "A texture map of red wood" \
    --outdir "outputs/generated" \
    --ckpt "logs/2022-10-19T14-00-44_material/checkpoints/last.ckpt" \
    --H 512 --W 512 \
    --n_samples 4 \
    --ddim_steps 100 \
    --config "configs/stable-diffusion/material.yaml"