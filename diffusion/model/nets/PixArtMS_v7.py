import torch
import torch.nn as nn

from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint
from diffusion.model.nets.PixArtMS_v6 import PixArtMSV6


@MODELS.register_module()
class PixArtMSV7(PixArtMSV6):
    """v7 SR backbone with 8-channel concat input and adapter-dispenser dict input."""

    def __init__(self, input_size=32, patch_size=2, in_channels=8, injection_strategy='front_dense', **kwargs):
        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            injection_strategy=injection_strategy,
            **kwargs,
        )
        self.hidden_size = self.x_embedder.proj.out_channels
        if hasattr(self, 'layer_feat_proj'):
            del self.layer_feat_proj

        self._init_injection_strategy(self.depth, mode=injection_strategy)
        self._rebuild_injection_modules()

    def _rebuild_injection_modules(self):
        n = len(self.injection_layers)
        self.injection_scales = nn.ParameterList([nn.Parameter(torch.ones(1, device=self.t_block[1].weight.device)) for _ in range(n)])
        self.input_adaln = nn.ModuleList([nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=True) for _ in range(n)])
        self.input_res_proj = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=True) for _ in range(n)])
        for lin in self.input_adaln:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        for lin in self.input_res_proj:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _init_injection_strategy(self, depth, mode='front_dense'):
        layers = []
        if mode == 'front_dense':
            layers.extend(list(range(0, 15)))
            layers.extend(list(range(15, 25, 2)))
        else:
            layers.extend(list(range(depth)))
        self.injection_layers = sorted(set([l for l in layers if l < self.injection_cutoff_layer]))
        self.injection_cutoff_layer = min(self.injection_cutoff_layer, 25)

    def load_pretrained_weights_with_zero_init(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if 'x_embedder.proj.weight' in name and param.shape[1] == 4 and own_state[name].shape[1] == 8:
                old_weight = param.data if isinstance(param, nn.Parameter) else param
                new_weight = own_state[name]
                new_weight[:, :4, :, :] = old_weight
                nn.init.zeros_(new_weight[:, 4:, :, :])
                own_state[name].copy_(new_weight)
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            own_state[name].copy_(param)

    def forward(self, x, timestep, y, mask=None, data_info=None, adapter_cond=None, force_drop_ids=None, **kwargs):
        bs = x.shape[0]
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)

        if data_info is None:
            c_size = torch.ones((bs, 2), device=x.device, dtype=self.dtype) * x.shape[-1]
            ar = torch.ones((bs,), device=x.device, dtype=self.dtype)
        else:
            c_size = data_info["img_hw"].to(self.dtype)
            ar = data_info["aspect_ratio"].to(self.dtype)

        self.h = x.shape[-2] // self.patch_size
        self.w = x.shape[-1] // self.patch_size

        pos_embed = self._build_dynamic_pos_embed(x)
        x = self.x_embedder(x) + pos_embed

        adapter_features = {}
        style_vec = None
        if adapter_cond is not None:
            if isinstance(adapter_cond, (tuple, list)) and len(adapter_cond) == 2:
                adapter_features, style_vec = adapter_cond

        t = self.t_embedder(timestep)
        t = t + torch.cat([self.csize_embedder(c_size, bs), self.ar_embedder(ar, bs)], dim=1)
        t0 = self.t_block(t)
        if style_vec is not None:
            t0 = t0 + self.style_fusion_mlp(style_vec)

        y = self.y_embedder(y, self.training, force_drop_ids=force_drop_ids)
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        alpha = self.adapter_alpha_mlp(t).view(-1, 1, 1)
        for i, block in enumerate(self.blocks):
            if i in adapter_features and i in self.injection_layers and i < self.injection_cutoff_layer:
                scale_idx = self.injection_layers.index(i)
                feat = adapter_features[i].flatten(2).transpose(1, 2)
                with torch.cuda.amp.autocast(enabled=False):
                    feat = self.input_adapter_ln(feat.float())
                    sb = self.input_adaln[scale_idx](feat)
                    shift, scale = sb.chunk(2, dim=-1)
                res = self.input_res_proj[scale_idx](feat)
                x = x + (self.injection_scales[scale_idx] * alpha) * res.to(x.dtype)
                x = auto_grad_checkpoint(
                    block,
                    x,
                    y,
                    t0,
                    y_lens,
                    adaln_shift=shift,
                    adaln_scale=scale,
                    adaln_alpha=self.injection_scales[scale_idx] * alpha,
                    **kwargs,
                )
            else:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, **kwargs)

        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x

    def _build_dynamic_pos_embed(self, x):
        from diffusion.model.nets.PixArt import get_2d_sincos_pos_embed
        pos_embed = (
            torch.from_numpy(
                get_2d_sincos_pos_embed(
                    self.pos_embed.shape[-1],
                    (self.h, self.w),
                    lewei_scale=self.lewei_scale,
                    base_size=self.base_size,
                )
            )
            .unsqueeze(0)
            .to(x.device)
            .to(self.dtype)
        )
        return pos_embed


@MODELS.register_module()
def PixArtMSV7_XL_2(**kwargs):
    return PixArtMSV7(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
