from torch import nn
from model.vit_csra import VIT_CSRA
from functools import partial
import torch.utils.model_zoo as model_zoo

model_url = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth'

class VitL16224Csra(VIT_CSRA):
    def __init__(self, 
                 pretrained=True,  
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=47, 
                 embed_dim=1024, 
                 depth=24,
                 num_heads=16, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 hybrid_backbone=None, 
                 cls_num_heads=1,
                 lam=0.3,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),   
                 **kwargs):
        super(VitL16224Csra, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
                                            num_heads, mlp_ratio, qkv_bias, qk_scale,  drop_rate, 
                                            attn_drop_rate, drop_path_rate, hybrid_backbone, norm_layer, 
                                            cls_num_heads, lam)
        if pretrained:
            state_dict = model_zoo.load_url(model_url)
            self.load_state_dict(state_dict, strict=False)
