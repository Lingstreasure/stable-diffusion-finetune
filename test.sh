python scripts/txt2img.py \
    --prompt "A texture map of red bricks" \
    --outdir "outputs/generated" \
    --ckpt "logs/2022-10-19T14-00-44_material/checkpoints/epoch=000005.ckpt" \
    --H 256 --W 256 \
    --n_samples 4 \
    --config "configs/stable-diffusion/material.yaml"