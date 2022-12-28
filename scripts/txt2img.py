import argparse, os, sys, glob
sys.path.append(os.getcwd())
import torch
import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=8,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )

    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--dyn",
        type=float,
        help="dynamic thresholding from Imagen, in latent space (TODO: try in pixel space with intermediate decode)",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--device_num",
        type=int,
        default=0,
        help="the number of the device for sampling",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device(f"cuda:{opt.device_num}") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    
    ## see the SimpleDecoder params number
    # from ldm.modules.diffusionmodules.model import SimpleDecoder
    # test_in = torch.randn((4, 512, 64, 64))
    # simpleDecoder = SimpleDecoder(512, 3)
    # simpleDecoder.eval()
    # model_params = sum(p.numel() for p in simpleDecoder.parameters())
    # print("sampleDecoder params: {}M".format(model_params // 1e6))
    # # test_out = simpleDecoder(test_in)
    # # print("test_out.shape: ", test_out.shape)
    # assert 0
    
    ### visualize AE out
    # dataset = instantiate_from_config(config.data.params.train)  # data[idx]: {'txt': str, 'image': tensor}
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.n_samples, num_workers=12, shuffle=False)
    # save_path = "/root/hz/Code/stable-diffusion-finetune/scripts/AE_out" 
    # for idx, data in enumerate(dataloader):
    #     if idx < 10:
    #         continue
    #     data['image'] = data['image'].to(device)
    #     # print(data['txt'])
    #     print(data['image'].shape)
    #     out1, posterior1 = model.first_stage_model(data['image'])
    #     assert 0
    #     out2, posterior2 = model.first_stage_model(data['image'], False)
    #     out1 = torch.clamp((out1 + 1.0) / 2.0, min=0.0, max=1.0).permute(0, 2, 3, 1)  # b c h w -> b h w c
    #     out2 = torch.clamp((out2 + 1.0) / 2.0, min=0.0, max=1.0).permute(0, 2 ,3, 1)  # b c h w -> b h w c
    #     for i, img1 in enumerate(out1):
    #         img1 = img1.cpu().numpy() * 255.0
    #         Image.fromarray(img1.astype(np.uint8)).save(
    #                                 os.path.join(save_path, f"{i:05}_ae1.png"))
    #     for i, img2 in enumerate(out2):
    #         img2 = img2.cpu().numpy() * 255.0
    #         Image.fromarray(img2.astype(np.uint8)).save(
    #                                 os.path.join(save_path, f"{i:05}_ae2.png"))
    #     assert 0    
    
    
    
    ### visualize model structure 
    # model_params = sum(p.numel() for p in model.parameters())
    # model_trainble_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("model total params: {}M, trainable params: {}M".format(model_params // 1e6, model_trainble_params // 1e6))
    # model_sub1 = model.first_stage_model
    # model_sub1_decoder = model.first_stage_model.decoder
    # model_sub2 = model.cond_stage_model
    # model_sub1_params = sum(p.numel() for p in model_sub1.parameters())
    # model_sub1_decoder_params = sum(p.numel() for p in model_sub1_decoder.parameters())
    # model_sub1_trainable_params = sum(p.numel() for p in model_sub1.parameters() if p.requires_grad)
    # model_sub2_params = sum(p.numel() for p in model_sub2.parameters())
    # model_sub2_trainable_params = sum(p.numel() for p in model_sub2.parameters() if p.requires_grad)
    # print("model sub1 params: {}M, trainalbe params: {}M".format(model_sub1_params // 1e6, model_sub1_trainable_params // 1e6))
    # print("model sub2 params: {}M, trainalbe params: {}M".format(model_sub2_params // 1e6, model_sub2_trainable_params // 1e6))
    # print("model sub1 decoder params; {}M".format(model_sub1_decoder_params // 1e6))
    # assert 0

    ### export the model structure (not yet)
    # torch.onnx.export(
    #     model,
    #     opt.prompt,
    #     'model.onnx',
    #     export_params=True,
    #     opset_version=8,
    # )


    if opt.plms:
        sampler = PLMSSampler(model)
    elif opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope(f"cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, intermediates = sampler.sample(S=opt.ddim_steps,
                                                                    conditioning=c,
                                                                    batch_size=opt.n_samples,
                                                                    shape=shape,
                                                                    verbose=False,
                                                                    log_every_t=5,
                                                                    unconditional_guidance_scale=opt.scale,
                                                                    unconditional_conditioning=uc,
                                                                    eta=opt.ddim_eta,
                                                                    dynamic_threshold=opt.dyn,
                                                                    x_T=start_code)
                        
                        ### visualize the latent space feature map
                        feature_maps = rearrange(samples_ddim, 'b (c n) h w -> (b c) n h w', n=1)  # b c h w -> (b c) 1 h w
                        min_value = (feature_maps.min(dim=-1, keepdim=True).values).min(dim=-2, keepdim=True).values
                        max_value = (feature_maps.max(dim=-1, keepdim=True).values).max(dim=-2, keepdim=True).values
                        feature_maps = (feature_maps - min_value) / (max_value - min_value)
                        feature_maps = make_grid(feature_maps, normalize=False, nrow=samples_ddim.shape[1], padding=1, pad_value=1.0)  # 1 (b h) (c w)
                        feature_maps = feature_maps.cpu().numpy().transpose(1, 2, 0)  # (b h) (c w) 3
                        # feature_maps = np.mean(feature_maps, axis=-1, keepdims=True)  # (b h) (c w) 1
                        feature_maps = (feature_maps * 255.0).astype(np.uint8)
                        # print(feature_maps.shape, feature_maps.dtype)
                        feature_maps = np.repeat(feature_maps, 4, axis=0)  # (b c) -> 4(b c)
                        feature_maps = np.repeat(feature_maps, 4, axis=1)  # w -> 4w
                        feature_maps = cv2.applyColorMap(feature_maps,cv2.COLORMAP_JET)
                        if not opt.skip_save:
                            cv2.imwrite(os.path.join(sample_path, f"{base_count:05}_features.png"), feature_maps)
                            
                            

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        if not opt.skip_save:
                            for x_sample in x_samples_ddim:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                    os.path.join(sample_path, f"{base_count:05}.png"))
                                base_count += 1
                        all_samples.append(x_samples_ddim)
                        
                        ### intermediates x
                        # x_intermediates = intermediates['x_inter']
                        # x_inters = []
                        # for i in range(len(x_intermediates)):
                        #     x_inter = x_intermediates[i]
                        #     x_inter = model.decode_first_stage(x_inter)
                        #     x_inter = torch.clamp((x_inter + 1.0) / 2.0, min=0.0, max=1.0)
                        #     x_inter = x_inter.cpu().permute(0, 2, 3, 1).numpy()
                        #     x_inters.append(x_inter)
                        # x_checked_inters = np.array(x_inters)
                        # x_checked_inters_torch = torch.from_numpy(x_checked_inters).permute(1, 0, 4, 2, 3)  # b n c h w
                        # all_x_inters = rearrange(x_checked_inters_torch, 'b n c h w -> (b n) c h w')
                        # if not opt.skip_save:
                        #     all_x_inters = make_grid(all_x_inters, nrow=len(x_intermediates), padding=4)  # grid - b, c, h, (w, n) c
                        #     all_x_inters = 255. * rearrange(all_x_inters.cpu().numpy(), 'c h w -> h w c')
                        #     img_inters = Image.fromarray(all_x_inters.astype(np.uint8))
                        #     img_inters.save(os.path.join(sample_path, f"{base_count:05}_inters.png"))           
                        # base_count += 1
                        
                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, prompts[0] + f'[seed-{opt.seed}].png'))
                    grid_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f"Sampling took {toc - tic}s, i.e. produced {opt.n_iter * opt.n_samples / (toc - tic):.2f} samples/sec."
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
