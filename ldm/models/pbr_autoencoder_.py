import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from contextlib import contextmanager

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config
# import sys
# sys.path.append("/home/d5/hz/Code/diffmat/diffmat/core")
# from render import Renderer
from einops import rearrange
from torchvision import transforms
import numpy as np

class PBRAutoEncoderImprove(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 scheduler_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 input_key="input",
                 gt_key="gt",
                 colorize_nlabels=None,
                 monitor=None,
                 scale_factor=1.0
                 ):
        super().__init__()
        self.input_key = input_key
        self.gt_key = gt_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.scheduler_config = scheduler_config
        self.use_scheduler = self.scheduler_config is not None
        self.rank_zero_print = self.print if self.trainer else print
        self.scale_factor = scale_factor
        # self.renderer = Renderer(device=self.device, normal_format='dx')
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.frozen_weights()

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        self.rank_zero_print(f"Restored from {path}")
    
    def frozen_weights(self):
        self.encoder = self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.quant_conv.parameters():
            p.requires_grad = False
        for p in self.post_quant_conv.parameters():
            p.requires_grad = False
        self.rank_zero_print("Frozen the weights of encoder, quant_conv, post_quant_conv")
    
    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):  # only train the decodder
        z = self.post_quant_conv(z)
        dec = self.decoder(z.detach())
        return dec
    
    @torch.no_grad()
    def decode_for_inference(self, z):
        z = 1.0 / self.scale_factor * z
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=False):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.input_key)
        gts = self.get_input(batch, self.gt_key)
        reconstructions, posterior = self(inputs)

        decoder_loss, log_dict_decoder = self.loss(gts, reconstructions, inputs, split="train")
        # print(log_dict_decoder)
        # self.log("decoder_loss", decoder_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("lr", self.learning_rate, prog_bar=False, logger=True, on_epoch=True)
        self.log_dict(log_dict_decoder, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return decoder_loss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.input_key)
        gts = self.get_input(batch, self.gt_key)
        reconstructions, posterior = self(inputs)
    
        val_decoder_loss, val_log_dict_decoder = self.loss(gts, reconstructions, inputs, split="val")
        # self.log("val_decoder_loss", val_decoder_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        # self.log("lr", self.learning_rate, prog_bar=False, logger=True, on_epoch=True)
        self.log_dict(val_log_dict_decoder, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return val_decoder_loss

    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.Adam(self.decoder.parameters(), lr=lr, betas=(0.5, 0.9))
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            self.print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }
            ]
            return [opt], scheduler
        return [opt], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        inputs = self.get_input(batch, self.input_key)
        inputs = inputs.to(self.device)
        gts = self.get_input(batch, self.gt_key)
        gts = gts.to(self.device)
        log["inputs"] = inputs
        if self.loss.renderer.device != self.device:
            self.loss.renderer.to(self.device)
        if not only_inputs:
            xrec, posterior = self(inputs)
            B, C, H, W = xrec.shape
            if C > 3:
                log["albedo"] = gts[:, :3, ...]
                log["normal"] = gts[:, 3:6, ...]
                log["rough"] = gts[:, 6:7, ...]#.repeat(1, 3, 1, 1)
                log["metal"] = gts[:, 7:, ...]#.repeat(1, 3, 1, 1)
                log["albedo_rec"] = xrec[:, :3, ...]
                log["normal_rec"] = xrec[:, 3:6, ...]
                log["rough_rec"] = xrec[:, 6:7, ...]#.repeat(1, 3, 1, 1)
                log["metal_rec"] = xrec[:, 7:, ...]#.repeat(1, 3, 1, 1)
                
                render_img = self.loss.renderer.evaluate(*[(xrec[:, :3, ...] + 1.0) / 2.0, 
                                                            (xrec[:, 3:6, ...] + 1.0) / 2.0, 
                                                            (xrec[:, 6:7, ...] + 1.0) / 2.0, 
                                                            (xrec[:, 7:, ...] + 1.0) / 2.0, 
                                                            torch.ones((B, 1, H, W)).float().to(self.device)])
                log["render"] = (render_img * 2.0 - 1.0).clamp_(-1, 1)
            else:
                log["gts"] = gts
                log["reconstructions"] = xrec
                
            # log bottle neck feature
            feature = posterior.sample()
            feature_maps = rearrange(feature, 'b (c n) h w -> b c (n h) w', c=1)
            min_value = (feature_maps.min(dim=-1, keepdim=True).values).min(dim=-2, keepdim=True).values
            max_value = (feature_maps.max(dim=-1, keepdim=True).values).max(dim=-2, keepdim=True).values
            feature_maps = (feature_maps - min_value) / (max_value - min_value)
            feature_maps = (feature_maps * 2.0 - 1).clamp_(-1, 1)
            feature_maps = np.repeat(feature_maps.cpu().numpy(), 4, axis=-2)
            feature_maps = np.repeat(feature_maps, 4, axis=-1)
            log["bottle_neck_feature"] = torch.from_numpy(feature_maps).to(self.device)
        return log

    def to_rgb(self, x):
        assert self.input_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
