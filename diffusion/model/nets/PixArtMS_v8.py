import torch
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint, to_2tuple
from diffusion.model.nets.PixArt_blocks import (
    t2i_modulate,
    CaptionEmbedder,
    WindowAttention,
    MultiHeadCrossAttention,
    T2IFinalLayer,
    SizeEmbedder,
    TimestepEmbedder # Reuse TimestepEmbedder for Augmentation Level
)
from diffusion.model.nets.PixArt import PixArt, get_2d_sincos_pos_embed


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, bias=True):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class PixArtMSBlockV8(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0.0, window_size=0, input_size=None, use_rel_pos=False, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            use_rel_pos=use_rel_pos,
            **block_kwargs,
        )
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.window_size = window_size
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(self, x, y, t, mask=None, adaln_shift=None, adaln_scale=None, adaln_alpha=None, **kwargs):
        b, n, c = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(b, 6, -1)
        ).chunk(6, dim=1)

        # Adapter Modulation Injection
        if adaln_shift is not None and adaln_scale is not None and adaln_alpha is not None:
            with torch.cuda.amp.autocast(enabled=False):
                h = self.norm1(x.float())
                # Apply affine transformation from adapter
                h = h * (1.0 + adaln_alpha.float() * adaln_scale.float()) + adaln_alpha.float() * adaln_shift.float()
                h = h.to(x.dtype)
        else:
            h = self.norm1(x)

        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(h, shift_msa, scale_msa)))
        x = x + self.cross_attn(x, y, mask)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x


@MODELS.register_module()
class PixArtMSV8(PixArt):
    """
    DiT-SR v8 Backbone.
    Features:
    1. Strong Initialization (Copy Init) for LR channels.
    2. Explicit Conditioning Augmentation (Noise Level Embedding).
    3. Non-zero Adapter Initialization.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=8,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=True,
        pred_sigma=True,
        drop_path: float = 0.0,
        window_size=0,
        window_block_indexes=None,
        use_rel_pos=False,
        caption_channels=4096,
        lewei_scale=1.0,
        config=None,
        model_max_length=120,
        sparse_inject_ratio: float = 1.0,
        injection_cutoff_layer: int = 25,
        injection_strategy: str = "front_dense",
        **kwargs,
    ):
        if window_block_indexes is None:
            window_block_indexes = []
            
        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            pred_sigma=pred_sigma,
            drop_path=drop_path,
            window_size=window_size,
            window_block_indexes=window_block_indexes,
            use_rel_pos=use_rel_pos,
            caption_channels=caption_channels,
            lewei_scale=lewei_scale,
            config=config,
            model_max_length=model_max_length,
            **kwargs,
        )

        target_channels = 4
        self.out_channels = target_channels * 2 if pred_sigma else target_channels
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.h = self.w = 0
        self.depth = depth
        self.hidden_size = hidden_size

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size, bias=True)
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )
        self.csize_embedder = SizeEmbedder(hidden_size // 3)
        self.ar_embedder = SizeEmbedder(hidden_size // 3)
        
        # [V8 New Feature] Augmentation Level Embedder (for SR3-style conditioning)
        self.aug_embedder = TimestepEmbedder(hidden_size)

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            PixArtMSBlockV8(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                input_size=(input_size // patch_size, input_size // patch_size),
                window_size=window_size if i in window_block_indexes else 0,
                use_rel_pos=use_rel_pos if i in window_block_indexes else False,
            )
            for i in range(depth)
        ])
        
        self.injection_cutoff_layer = min(depth, int(injection_cutoff_layer))
        self._init_injection_strategy(depth, mode=injection_strategy, sparse_ratio=sparse_inject_ratio)

        n = len(self.injection_layers)
        self.injection_scales = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(n)])
        self.style_fusion_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.adapter_alpha_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )

        self.input_adapter_ln = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.input_adaln = nn.ModuleList([nn.Linear(hidden_size, 2 * hidden_size, bias=True) for _ in range(n)])
        self.input_res_proj = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=True) for _ in range(n)])

        self.initialize()
        for lin in self.input_adaln:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        for lin in self.input_res_proj:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        
        # [V8 Fix] Ensure Adapter is NOT zero-initialized.
        # Giving it a small kick (0.02) to ensure gradient flow.
        nn.init.normal_(self.adapter_alpha_mlp[-2].weight, std=0.02)
        nn.init.zeros_(self.adapter_alpha_mlp[-2].bias)
        
        # Init Augmentation Embedder
        nn.init.normal_(self.aug_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.aug_embedder.mlp[2].weight, std=0.02)

    def _init_injection_strategy(self, depth, mode='front_dense', sparse_ratio=1.0):
        if mode == 'front_dense':
            layers = list(range(0, min(15, self.injection_cutoff_layer)))
            layers.extend(list(range(15, min(25, self.injection_cutoff_layer), 2)))
            self.injection_layers = sorted(set(layers))
            return

        all_layers = list(range(depth))
        if sparse_ratio < 1.0:
            num_keep = max(1, int(len(all_layers) * sparse_ratio))
            self.injection_layers = [l for l in all_layers[:num_keep] if l < self.injection_cutoff_layer]
        else:
            self.injection_layers = [l for l in all_layers if l < self.injection_cutoff_layer]

    def load_pretrained_weights_with_zero_init(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if 'x_embedder.proj.weight' in name and param.shape[1] == 4 and own_state[name].shape[1] == 8:
                old_weight = param.data if isinstance(param, nn.Parameter) else param
                new_weight = own_state[name]
                
                # [V8 CRITICAL FIX: Copy Initialization]
                # Instead of noise or zero, we COPY the pretrained weights to the new channels.
                # Since LR latent has similar statistics to Noisy Latent (spatially),
                # this lets the model "see" structures immediately.
                # Scaling by 0.5 helps balance the variance doubling from concat.
                new_weight[:, :4, :, :] = old_weight
                new_weight[:, 4:, :, :] = old_weight * 0.5
                
                own_state[name].copy_(new_weight)
                continue
            
            if isinstance(param, nn.Parameter):
                param = param.data
            if own_state[name].shape == param.shape:
                own_state[name].copy_(param)

    def _build_dynamic_pos_embed(self, x):
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

    def forward(self, x, timestep, y, aug_level=None, mask=None, data_info=None, adapter_cond=None, force_drop_ids=None, **kwargs):
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

        x = self.x_embedder(x) + self._build_dynamic_pos_embed(x)

        adapter_features = {}
        style_vec = None
        if adapter_cond is not None and isinstance(adapter_cond, (tuple, list)) and len(adapter_cond) == 2:
            adapter_features, style_vec = adapter_cond

        t = self.t_embedder(timestep)
        t = t + torch.cat([self.csize_embedder(c_size, bs), self.ar_embedder(ar, bs)], dim=1)
        
        # [V8 New] Inject Augmentation Level Embedding
        if aug_level is not None:
            aug_emb = self.aug_embedder(aug_level.to(self.dtype))
            t = t + aug_emb

        if style_vec is not None:
            t = t + self.style_fusion_mlp(style_vec)
        t0 = self.t_block(t) 

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
                    adaln_shift, adaln_scale = sb.chunk(2, dim=-1)
                res = self.input_res_proj[scale_idx](feat)
                layer_alpha = self.injection_scales[scale_idx] * alpha
                x = x + layer_alpha * res.to(x.dtype)
                x = auto_grad_checkpoint(
                    block,
                    x,
                    y,
                    t0,
                    y_lens,
                    adaln_shift=adaln_shift,
                    adaln_scale=adaln_scale,
                    adaln_alpha=layer_alpha,
                    **kwargs,
                )
            else:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, **kwargs)

        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x

    def forward_with_dpmsolver(self, x, timestep, y, data_info, aug_level=None, **kwargs):
        # Support V8 aug_level in inference
        model_out = self.forward(x, timestep, y, aug_level=aug_level, data_info=data_info, **kwargs)
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, timestep, y, cfg_scale, data_info, aug_level=None, **kwargs):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        # Aug level needs to be duplicated for batch
        if aug_level is not None:
            aug_level = torch.cat([aug_level, aug_level], dim=0)
            
        model_out = self.forward(combined, timestep, y, aug_level=aug_level, data_info=data_info, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        assert self.h * self.w == x.shape[1]
        x = x.reshape((x.shape[0], self.h, self.w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape((x.shape[0], c, self.h * p, self.w * p))

    def initialize(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        nn.init.normal_(self.csize_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.csize_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.ar_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.ar_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


@MODELS.register_module()
def PixArtMSV8_XL_2(**kwargs):
    return PixArtMSV8(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)