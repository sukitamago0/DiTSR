  
# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.

# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.

# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.

# import torch
# import torch.nn as nn
# from timm.models.layers import DropPath
# from timm.models.vision_transformer import Mlp

# from diffusion.model.builder import MODELS
# from diffusion.model.utils import auto_grad_checkpoint, to_2tuple
# from diffusion.model.nets.PixArt_blocks import (
#     t2i_modulate, CaptionEmbedder, WindowAttention, MultiHeadCrossAttention,
#     T2IFinalLayer, TimestepEmbedder, SizeEmbedder
# )
# from diffusion.model.nets.PixArt import PixArt, get_2d_sincos_pos_embed


# class PatchEmbed(nn.Module):
#     def __init__(self, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, bias=True):
#         super().__init__()
#         patch_size = to_2tuple(patch_size)
#         self.patch_size = patch_size
#         self.flatten = flatten
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

#     def forward(self, x):
#         x = self.proj(x)
#         if self.flatten:
#             x = x.flatten(2).transpose(1, 2)
#         x = self.norm(x)
#         return x


# class PixArtMSBlock(nn.Module):
#     def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0.0, window_size=0, input_size=None, use_rel_pos=False, **block_kwargs):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.attn = WindowAttention(
#             hidden_size, num_heads=num_heads, qkv_bias=True,
#             input_size=input_size if window_size == 0 else (window_size, window_size),
#             use_rel_pos=use_rel_pos, **block_kwargs
#         )
#         self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
#         self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         approx_gelu = lambda: nn.GELU(approximate="tanh")
#         self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
#         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
#         self.window_size = window_size
#         self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

#     def forward(self, x, y, t, mask=None, **kwargs):
#         B, N, C = x.shape
#         shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)

#         x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)))
#         x = x + self.cross_attn(x, y, mask)
#         x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
#         return x


# @MODELS.register_module()
# class PixArtMS(PixArt):
#     def __init__(self, input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, learn_sigma=True, pred_sigma=True, drop_path: float = 0.0, window_size=0, window_block_indexes=None, use_rel_pos=False, caption_channels=4096, lewei_scale=1.0, config=None, model_max_length=120, **kwargs):
#         if window_block_indexes is None: window_block_indexes = []
#         super().__init__(input_size=input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, class_dropout_prob=class_dropout_prob, learn_sigma=learn_sigma, pred_sigma=pred_sigma, drop_path=drop_path, window_size=window_size, window_block_indexes=window_block_indexes, use_rel_pos=use_rel_pos, lewei_scale=lewei_scale, config=config, model_max_length=model_max_length, **kwargs)

#         self.h = self.w = 0
#         approx_gelu = lambda: nn.GELU(approximate="tanh")
#         self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
#         self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size, bias=True)
#         self.y_embedder = CaptionEmbedder(in_channels=caption_channels, hidden_size=hidden_size, uncond_prob=class_dropout_prob, act_layer=approx_gelu, token_num=model_max_length)
#         self.csize_embedder = SizeEmbedder(hidden_size // 3)
#         self.ar_embedder = SizeEmbedder(hidden_size // 3)

#         drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]
#         self.blocks = nn.ModuleList([
#             PixArtMSBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i], input_size=(input_size // patch_size, input_size // patch_size), window_size=window_size if i in window_block_indexes else 0, use_rel_pos=use_rel_pos if i in window_block_indexes else False)
#             for i in range(depth)
#         ])
#         self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

#  # ==========================================================
#         # [修改] Input Injection
#         # ==========================================================
#         self.injection_layers = [0, 7, 14, 21] 
        
#         # [关键修改] 改为初始化为 1.0
#         self.injection_scales = nn.ParameterList([
#             nn.Parameter(torch.ones(1)) for _ in range(len(self.injection_layers))
#         ])

#         # 👇👇👇 [新增代码开始] 👇👇👇
#         # 新增: 专门用于 Input Injection 的 LayerNorm
#         # 作用: 强制归一化 Adapter 的特征，防止方差过大导致生成图过饱和(油画感)
#         self.input_adapter_ln = nn.LayerNorm(hidden_size, elementwise_affine=True)

#         # ==========================================================
#         # [修改] Cross-Attn Injection
#         # ==========================================================
#         self.adapter_proj = nn.Linear(hidden_size, hidden_size)
#         nn.init.xavier_uniform_(self.adapter_proj.weight)
#         nn.init.zeros_(self.adapter_proj.bias)

#         self.adapter_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

#         # [关键修改] 同理，初始化为 1.0，避免死锁
#         self.cross_attn_scale = nn.Parameter(torch.ones(1))

#         self.initialize()

#     def forward(
#         self,
#         x,
#         timestep,
#         y,
#         mask=None,
#         data_info=None,
#         adapter_cond=None,      
#         injection_mode='hybrid',
#         **kwargs,
#     ):
#         bs = x.shape[0]
#         x = x.to(self.dtype)
#         timestep = timestep.to(self.dtype)
#         y = y.to(self.dtype)

#         c_size = data_info["img_hw"].to(self.dtype)
#         ar = data_info["aspect_ratio"].to(self.dtype)

#         self.h = x.shape[-2] // self.patch_size
#         self.w = x.shape[-1] // self.patch_size

#         pos_embed = (
#             torch.from_numpy(
#                 get_2d_sincos_pos_embed(
#                     self.pos_embed.shape[-1],
#                     (self.h, self.w),
#                     lewei_scale=self.lewei_scale,
#                     base_size=self.base_size,
#                 )
#             )
#             .unsqueeze(0)
#             .to(x.device)
#             .to(self.dtype)
#         )

#         x = self.x_embedder(x) + pos_embed
        
#         # [确认] adapter_cond 已经是 [B, 1152, 64, 64] (由新 Adapter 输出)
#         adapter_flat = None
#         if adapter_cond is not None:
#             # Flatten 为 [B, 4096, 1152] (64*64=4096)
#             adapter_flat = adapter_cond.flatten(2).transpose(1, 2)

#             # 👇👇👇 [新增代码开始] 👇👇👇
#             # 对 Adapter 特征进行归一化，使其分布与主干兼容
#             adapter_flat = self.input_adapter_ln(adapter_flat)
#             # 👆👆👆 [新增代码结束] 👆👆👆

#         t = self.t_embedder(timestep)
#         t = t + torch.cat([self.csize_embedder(c_size, bs), self.ar_embedder(ar, bs)], dim=1)
#         t0 = self.t_block(t)
#         y = self.y_embedder(y, self.training)

#         if adapter_cond is not None and injection_mode in ['cross_attn', 'hybrid']:
#             vis_tokens = adapter_flat
#             vis_tokens = self.adapter_proj(vis_tokens)
#             vis_tokens = self.adapter_norm(vis_tokens)
#             vis_tokens = vis_tokens * self.cross_attn_scale # Scale=1.0 * Input=0.0 -> 0.0
#             if y.dim() == 4:
#                 vis_tokens = vis_tokens.unsqueeze(1)
#                 concat_dim = 2
#             else:
#                 concat_dim = 1
#             y = torch.cat([y, vis_tokens], dim=concat_dim)

#         # ... (Mask 处理保持不变) ...
#         if mask is not None:
#              if mask.shape[0] != y.shape[0]: mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
#              mask = mask.squeeze(1).squeeze(1)
#              y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
#              y_lens = mask.sum(dim=1).tolist()
#         else:
#              if 'y_lens' not in locals():
#                  seq_len_dim = 2 if y.dim() == 4 else 1
#                  y_lens = [y.shape[seq_len_dim]] * y.shape[0]
#                  if y.dim() == 4: y = y.squeeze(1).view(1, -1, x.shape[-1])
#                  else: y = y.view(1, -1, x.shape[-1])

#         for i, block in enumerate(self.blocks):
#             if adapter_cond is not None and injection_mode in ['input', 'hybrid']:
#                 if i in self.injection_layers:
#                     scale_idx = self.injection_layers.index(i)
#                     current_scale = self.injection_scales[scale_idx]
#                     # x [B, N, C] + 1.0 * adapter [B, N, C] (初始为0)
#                     x = x + current_scale * adapter_flat
#             x = auto_grad_checkpoint(block, x, y, t0, y_lens, **kwargs)

#         x = self.final_layer(x, t)
#         x = self.unpatchify(x)
#         return x

#     def forward_with_dpmsolver(self, x, timestep, y, data_info, **kwargs):
#         model_out = self.forward(x, timestep, y, data_info=data_info, **kwargs)
#         return model_out.chunk(2, dim=1)[0]

#     def forward_with_cfg(self, x, timestep, y, cfg_scale, data_info, **kwargs):
#         half = x[: len(x) // 2]
#         combined = torch.cat([half, half], dim=0)
#         model_out = self.forward(combined, timestep, y, data_info=data_info)
#         eps, rest = model_out[:, :3], model_out[:, 3:]
#         cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
#         half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
#         eps = torch.cat([half_eps, half_eps], dim=0)
#         return torch.cat([eps, rest], dim=1)

#     def unpatchify(self, x):
#         c = self.out_channels
#         p = self.x_embedder.patch_size[0]
#         assert self.h * self.w == x.shape[1]
#         x = x.reshape((x.shape[0], self.h, self.w, p, p, c))
#         x = torch.einsum("nhwpqc->nchpwq", x)
#         return x.reshape((x.shape[0], c, self.h * p, self.w * p))

#     def initialize(self):
#         def _basic_init(module):
#             if isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     nn.init.constant_(module.bias, 0)
#         self.apply(_basic_init)

#         w = self.x_embedder.proj.weight.data
#         nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
#         nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
#         nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
#         nn.init.normal_(self.t_block[1].weight, std=0.02)
#         nn.init.normal_(self.csize_embedder.mlp[0].weight, std=0.02)
#         nn.init.normal_(self.csize_embedder.mlp[2].weight, std=0.02)
#         nn.init.normal_(self.ar_embedder.mlp[0].weight, std=0.02)
#         nn.init.normal_(self.ar_embedder.mlp[2].weight, std=0.02)
#         nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
#         nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

#         for block in self.blocks:
#             nn.init.constant_(block.cross_attn.proj.weight, 0)
#             nn.init.constant_(block.cross_attn.proj.bias, 0)

#         nn.init.constant_(self.final_layer.linear.weight, 0)
#         nn.init.constant_(self.final_layer.linear.bias, 0)


# @MODELS.register_module()
# def PixArtMS_XL_2(**kwargs):
#     return PixArtMS(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)






#————————————————————————————
# v2****最好版本 pixartMS.py
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint, to_2tuple
from diffusion.model.nets.PixArt_blocks import (
    t2i_modulate, CaptionEmbedder, WindowAttention, MultiHeadCrossAttention,
    T2IFinalLayer, TimestepEmbedder, SizeEmbedder
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


class PixArtMSBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0.0, window_size=0, input_size=None, use_rel_pos=False, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(
            hidden_size, num_heads=num_heads, qkv_bias=True,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            use_rel_pos=use_rel_pos, **block_kwargs
        )
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.window_size = window_size
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(self, x, y, t, mask=None, **kwargs):
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)

        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)))
        x = x + self.cross_attn(x, y, mask)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x


@MODELS.register_module()
class PixArtMS(PixArt):
    def __init__(self, input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, learn_sigma=True, pred_sigma=True, drop_path: float = 0.0, window_size=0, window_block_indexes=None, use_rel_pos=False, caption_channels=4096, lewei_scale=1.0, config=None, model_max_length=120, **kwargs):
        if window_block_indexes is None: window_block_indexes = []
        super().__init__(input_size=input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, class_dropout_prob=class_dropout_prob, learn_sigma=learn_sigma, pred_sigma=pred_sigma, drop_path=drop_path, window_size=window_size, window_block_indexes=window_block_indexes, use_rel_pos=use_rel_pos, lewei_scale=lewei_scale, config=config, model_max_length=model_max_length, **kwargs)

        self.h = self.w = 0
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size, bias=True)
        self.y_embedder = CaptionEmbedder(in_channels=caption_channels, hidden_size=hidden_size, uncond_prob=class_dropout_prob, act_layer=approx_gelu, token_num=model_max_length)
        self.csize_embedder = SizeEmbedder(hidden_size // 3)
        self.ar_embedder = SizeEmbedder(hidden_size // 3)

        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            PixArtMSBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i], input_size=(input_size // patch_size, input_size // patch_size), window_size=window_size if i in window_block_indexes else 0, use_rel_pos=use_rel_pos if i in window_block_indexes else False)
            for i in range(depth)
        ])
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        # ==========================================================
        # Input Injection (Multi-Level Support)
        # ==========================================================
        self.injection_layers = [0, 7, 14, 21] 
        
        self.injection_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(len(self.injection_layers))
        ])

        # [保留成功经验] LayerNorm，用于归一化 Adapter 特征
        self.input_adapter_ln = nn.LayerNorm(hidden_size, elementwise_affine=True)

        # ==========================================================
        # Cross-Attn Injection
        # ==========================================================
        self.adapter_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.adapter_proj.weight)
        nn.init.zeros_(self.adapter_proj.bias)

        self.adapter_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.cross_attn_scale = nn.Parameter(torch.ones(1))

        self.initialize()

    def forward(
        self,
        x,
        timestep,
        y,
        mask=None,
        data_info=None,
        adapter_cond=None,      
        injection_mode='hybrid',
        **kwargs,
    ):
        bs = x.shape[0]
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)

        c_size = data_info["img_hw"].to(self.dtype)
        ar = data_info["aspect_ratio"].to(self.dtype)

        self.h = x.shape[-2] // self.patch_size
        self.w = x.shape[-1] // self.patch_size

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

        x = self.x_embedder(x) + pos_embed
        
        # --------------------------------------------------------
        # [核心修改] Adapter 特征预处理 (支持 List 或 Tensor)
        # --------------------------------------------------------
        adapter_features = [] # 用于存储处理后的特征列表
        
        if adapter_cond is not None:
            # 1. 如果是列表 (Multi-Scale Adapter 输出)
            if isinstance(adapter_cond, list):
                for feat in adapter_cond:
                    # Flatten: [B, C, H, W] -> [B, N, C]
                    feat_flat = feat.flatten(2).transpose(1, 2)
                    # Apply LN (保留成功经验)
                    with torch.cuda.amp.autocast(enabled=False):
                        feat_flat = self.input_adapter_ln(feat_flat.float())
                    adapter_features.append(feat_flat)
            
            # 2. 如果是 Tensor (Single-Scale Adapter 输出，兼容旧代码)
            else:
                feat_flat = adapter_cond.flatten(2).transpose(1, 2)
                with torch.cuda.amp.autocast(enabled=False):
                    feat_flat = self.input_adapter_ln(feat_flat.float())
                # 复制 4 份，填满 injection_layers
                adapter_features = [feat_flat] * len(self.injection_layers)

        t = self.t_embedder(timestep)
        t = t + torch.cat([self.csize_embedder(c_size, bs), self.ar_embedder(ar, bs)], dim=1)
        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)

        # Cross-Attn Injection (取 adapter_features 的最后一个特征作为全局特征)
        if len(adapter_features) > 0 and injection_mode in ['cross_attn', 'hybrid']:
            vis_tokens = adapter_features[-1] # 使用最深层特征
            vis_tokens = self.adapter_proj(vis_tokens)
            vis_tokens = self.adapter_norm(vis_tokens)
            vis_tokens = vis_tokens * self.cross_attn_scale 
            if y.dim() == 4:
                vis_tokens = vis_tokens.unsqueeze(1)
                concat_dim = 2
            else:
                concat_dim = 1
            y = torch.cat([y, vis_tokens], dim=concat_dim)

        if mask is not None:
             if mask.shape[0] != y.shape[0]: mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
             mask = mask.squeeze(1).squeeze(1)
             y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
             y_lens = mask.sum(dim=1).tolist()
        else:
             if 'y_lens' not in locals():
                 seq_len_dim = 2 if y.dim() == 4 else 1
                 y_lens = [y.shape[seq_len_dim]] * y.shape[0]
                 if y.dim() == 4: y = y.squeeze(1).view(1, -1, x.shape[-1])
                 else: y = y.view(1, -1, x.shape[-1])

        # 主干循环
        for i, block in enumerate(self.blocks):
            if len(adapter_features) > 0 and injection_mode in ['input', 'hybrid']:
                if i in self.injection_layers:
                    # 获取注入层索引 (0, 1, 2, 3)
                    scale_idx = self.injection_layers.index(i)
                    current_scale = self.injection_scales[scale_idx]
                    
                    # [关键] 从 adapter_features 列表中取出对应层级的特征
                    # 如果列表长度不够(例如只传了1个)，复用最后一个
                    feat_idx = scale_idx if scale_idx < len(adapter_features) else -1
                    current_feat = adapter_features[feat_idx]
                    
                    # 注入: x = x + scale * LN(adapter_feat)
                    x = x + current_scale * current_feat
            
            x = auto_grad_checkpoint(block, x, y, t0, y_lens, **kwargs)

        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x

    def forward_with_dpmsolver(self, x, timestep, y, data_info, **kwargs):
        model_out = self.forward(x, timestep, y, data_info=data_info, **kwargs)
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, timestep, y, cfg_scale, data_info, **kwargs):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, y, data_info=data_info)
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
def PixArtMS_XL_2(**kwargs):
    return PixArtMS(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

