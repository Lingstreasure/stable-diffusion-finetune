python scripts/txt2img.py --prompt "A big elephant" --plms --outdir ./outputs/generated \
    --ckpt ./models/ldm/sd-v1-4.ckpt \
    --ddim_steps 100 \
    --H 512 \
    --W 512 \
    --seed 8