import torch
import torch.nn as nn
import sys
sys.path.append("/root/hz/Code/diffmat/diffmat/core")
from render import Renderer
from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        # self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
        #                                          n_layers=disc_num_layers,
        #                                          use_actnorm=use_actnorm
        #                                          ).apply(weights_init)
        # self.discriminator_iter_start = disc_start
        # self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        # self.disc_factor = disc_factor
        # self.discriminator_weight = disc_weight
        # self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        # if optimizer_idx == 0:
        # generator update
        # if cond is None:
        #     assert not self.disc_conditional
        #     logits_fake = self.discriminator(reconstructions.contiguous())
        # else:
        #     assert self.disc_conditional
        #     logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
        # g_loss = -torch.mean(logits_fake)

        # if self.disc_factor > 0.0:
        #     try:
        #         d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
        #     except RuntimeError:
        #         assert not self.training
        #         d_weight = torch.tensor(0.0)
        # else:
        #     d_weight = torch.tensor(0.0)
        
        # disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
        loss = weighted_nll_loss + self.kl_weight * kl_loss# + d_weight * disc_factor * g_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
            #    "{}/d_weight".format(split): d_weight.detach(),
            #    "{}/disc_factor".format(split): torch.tensor(disc_factor),
            #    "{}/g_loss".format(split): g_loss.detach().mean(),
                }
        return loss, log

        # if optimizer_idx == 1:
        #     # second pass for discriminator update
        #     if cond is None:
        #         logits_real = self.discriminator(inputs.contiguous().detach())
        #         logits_fake = self.discriminator(reconstructions.contiguous().detach())
        #     else:
        #         logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
        #         logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

        #     disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
        #     d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

        #     log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
        #            "{}/logits_real".format(split): logits_real.detach().mean(),
        #            "{}/logits_fake".format(split): logits_fake.detach().mean()
        #            }
        #    return d_loss, log


class PBRDecoderLoss(nn.Module):
    def __init__(self, 
                 random_perceptual_weight = 0, 
                 perceptual_weight=0, 
                 render_weight=0, 
                 disc_weight=0,
                 disc_factor=1.0, 
                 use_actnorm=False,
                 disc_num_layers=3, 
                 disc_in_channels=8, 
                 disc_start=2000, 
                 disc_loss="hinge"):
        """ The loss of PBR map decoder.(VAE-decoder)
        """        
        super().__init__()
        if random_perceptual_weight > 0 or perceptual_weight > 0:
            self.perceptual_loss = LPIPS().eval()
        self.renderer = Renderer(normal_format='dx')
        self.random_perceptual_weight = random_perceptual_weight
        self.perceptual_weight = perceptual_weight
        self.render_weight = render_weight
        self.disc_weight = disc_weight
        if disc_weight > 0:
            self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                    n_layers=disc_num_layers,
                                                    use_actnorm=use_actnorm
                                                    ).apply(weights_init)
            self.disc_start = disc_start
            self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
            self.disc_factor = disc_factor
            
    
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.disc_weight
        return d_weight
    
    def forward(self, gts, reconstructions, inputs, optimizer_idx, 
                global_step, last_layer=None, split="train"):
        B, C, H, W = reconstructions.shape
        rec_loss = torch.abs(gts.contiguous() - reconstructions.contiguous())
        nll_loss = rec_loss.mean()
        loss = rec_loss.mean()
        
        ## GAN part
        if self.disc_weight > 0:
            if optimizer_idx == 0:
                ## decoder update
                logits_fake = self.discriminator(reconstructions.contiguous())
                g_loss = -torch.mean(logits_fake)

                if self.disc_factor > 0.0:
                    try:
                        d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                    except RuntimeError:
                        assert not self.training
                        d_weight = torch.tensor(0.0)
                else:
                    d_weight = torch.tensor(0.0)
                
                disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.disc_start)
                loss += g_loss * d_weight * disc_factor
                
                
            if optimizer_idx == 1:
                ## discriminatior update
                logits_real = self.discriminator(gts.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
                disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.disc_start)
                d_loss = disc_factor * self.disc_loss(logits_real, logits_fake).mean()

                log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                        "{}/logits_real".format(split): logits_real.detach().mean(),
                        "{}/logits_fake".format(split): logits_fake.detach().mean()
                      }
                return d_loss, log
        
        # if self.perceptual_weight > 0:
        #     p_loss = self.perceptual_loss(gts.contiguous(), reconstructions.contiguous())
        #     loss += p_loss / p_loss.shape[0] * self.perceptual_weight

        if self.render_weight > 0:
            if self.renderer.device != gts.device:
                self.renderer.to(gts.device)
            render_imgs = self.renderer.evaluate(*[(reconstructions[:, :3, ...] + 1.0) / 2.0, 
                                                    (reconstructions[:, 3:6, ...] + 1.0) / 2.0, 
                                                    (reconstructions[:, 6:7, ...] + 1.0) / 2.0, 
                                                    (reconstructions[:, 7:, ...] + 1.0) / 2.0, 
                                                    torch.ones((B, 1, H, W)).float().to(gts.device)])
            render_loss = torch.abs((inputs * 2.0 - 1).contiguous() - render_imgs.contiguous())
            loss += torch.sum(render_loss) / render_loss.shape[0] * self.render_weight
        
        if self.random_perceptual_weight > 0:
            indexes = torch.randint(0, C, (3,))
            random_p_loss = self.perceptual_loss(gts[:, indexes, ...].contiguous(), 
                                                 reconstructions[:, indexes, ...].contiguous())
            # loss += torch.sum(random_p_loss) / random_p_loss.shape[0] * self.random_perceptual_weight
            loss += random_p_loss.mean() * self.random_perceptual_weight

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(), 
               "{}/rec_loss".format(split): rec_loss.detach().mean()}
        
        if self.render_weight > 0:
            log["{}/render_loss".format(split)] = render_loss.detach().mean()
            
        if self.random_perceptual_weight > 0:
            log["{}/random_per_loss".format(split)] = random_p_loss.detach().mean()
        
        if self.disc_weight > 0:
            log["{}/d_weight".format(split)] = torch.tensor(d_weight).clone().detach()
            log['{}/g_loss'.format(split)] = g_loss.detach().mean()
        # if self.perceptual_weight > 0:
        #     log["{}/per_loss".format(split)] = p_loss.detach().mean()
        return loss, log
