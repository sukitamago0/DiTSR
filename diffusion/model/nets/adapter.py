# # v1 单尺度
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # -------------------------
# # 基础残差块
# # -------------------------
# class ResBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.norm1 = nn.GroupNorm(32, channels)
#         self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
#         self.norm2 = nn.GroupNorm(32, channels)
#         self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
#         self.act = nn.SiLU()

#     def forward(self, x):
#         h = self.act(self.norm1(x))
#         h = self.conv1(h)
#         h = self.act(self.norm2(h))
#         h = self.conv2(h)
#         return x + h

# # -------------------------
# # 新的 Frequency Adapter (Latent 对齐 + Token 对齐版)
# # -------------------------
# class ProgressiveFrequencyAdapter(nn.Module):
#     def __init__(self, in_channels=4, hidden_size=1152):
#         super().__init__()

#         # 输入: [B, 4, 64, 64] (VAE Latent)
#         self.stem = nn.Sequential(
#             nn.Conv2d(in_channels, 64, 3, padding=1),
#             ResBlock(64),
#             ResBlock(64)
#         )

#         # 特征提取主体 (保持 64x64 分辨率以提取细粒度特征)
#         self.mid_body = nn.Sequential(
#             nn.Conv2d(64, 256, 3, padding=1),
#             nn.SiLU(),
#             ResBlock(256),
#             nn.Conv2d(256, 768, 3, padding=1),
#             nn.SiLU(),
#             ResBlock(768)
#         )

#         # [关键修改] Fusion block 必须包含下采样 (Stride=2)
#         # 原因: PixArt PatchEmbed (PatchSize=2) 会将 64x64 的 Latent 变成 32x32 的 Tokens
#         # 因此 Adapter 最终输出必须也是 32x32，否则会报 1024 vs 4096 的维度错误
#         self.fusion = nn.Sequential(
#             # 64x64 -> 32x32 (匹配 PixArt 的 Token Grid)
#             nn.Conv2d(768, hidden_size, 3, padding=1, stride=2), 
#             nn.SiLU(),
#             # Zero-Init 1x1 卷积 (保持零初始化特性)
#             nn.Conv2d(hidden_size, hidden_size, 1), 
#         )

#         self._zero_init_last()

#     def _zero_init_last(self):
#         nn.init.zeros_(self.fusion[-1].weight)
#         nn.init.zeros_(self.fusion[-1].bias)

#     def forward(self, x):
#         # x: [B, 4, 64, 64]
#         x = self.stem(x)     # -> [B, 64, 64, 64]
#         x = self.mid_body(x) # -> [B, 768, 64, 64]
        
#         # 下采样并映射到 Hidden Size
#         x = self.fusion(x)   # -> [B, 1152, 32, 32] (Token Aligned)
        
#         return x

# adapter.py
# import torch
# import torch.nn as nn

# # ResBlock 保持不变
# class ResBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.norm1 = nn.GroupNorm(32, channels)
#         self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
#         self.norm2 = nn.GroupNorm(32, channels)
#         self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
#         self.act = nn.SiLU()

#     def forward(self, x):
#         h = self.act(self.norm1(x))
#         h = self.conv1(h)
#         h = self.act(self.norm2(h))
#         h = self.conv2(h)
#         return x + h

# class MultiLevelAdapter(nn.Module):
#     def __init__(self, in_channels=4, hidden_size=1152):
#         super().__init__()
        
#         # 1. Stem (对应 Layer 0 注入)
#         self.stem = nn.Sequential(
#             nn.Conv2d(in_channels, 64, 3, padding=1),
#             ResBlock(64),
#             ResBlock(64)
#         )
#         self.proj_0 = self._make_projection(64, hidden_size)

#         # 2. Body 1 (对应 Layer 7 注入)
#         self.body1 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, padding=1),
#             ResBlock(128),
#             ResBlock(128)
#         )
#         self.proj_1 = self._make_projection(128, hidden_size)

#         # 3. Body 2 (对应 Layer 14 注入)
#         self.body2 = nn.Sequential(
#             nn.Conv2d(128, 256, 3, padding=1),
#             ResBlock(256),
#             ResBlock(256)
#         )
#         self.proj_2 = self._make_projection(256, hidden_size)

#         # 4. Body 3 (对应 Layer 21 注入)
#         self.body3 = nn.Sequential(
#             nn.Conv2d(256, 512, 3, padding=1),
#             ResBlock(512),
#             ResBlock(512)
#         )
#         self.proj_3 = self._make_projection(512, hidden_size)

#     def _make_projection(self, in_ch, out_ch):
#         # 必须包含 Stride=2 以匹配 PixArt 的 Token 数量 (1024)
#         # 必须包含 Zero-Init 以保证初始无损
#         proj = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=2),
#             nn.SiLU(),
#             nn.Conv2d(out_ch, out_ch, 1) # Zero-Init 放在这里
#         )
#         # 初始化最后一层为 0
#         nn.init.zeros_(proj[-1].weight)
#         nn.init.zeros_(proj[-1].bias)
#         return proj

#     def forward(self, x):
#         # x: [B, 4, 64, 64]
        
#         # Level 0
#         f0 = self.stem(x)           # [B, 64, 64, 64]
#         out0 = self.proj_0(f0)      # [B, 1152, 32, 32] -> Flatten later
        
#         # Level 1
#         f1 = self.body1(f0)         # [B, 128, 64, 64]
#         out1 = self.proj_1(f1)      # [B, 1152, 32, 32]
        
#         # Level 2
#         f2 = self.body2(f1)         # [B, 256, 64, 64]
#         out2 = self.proj_2(f2)      # [B, 1152, 32, 32]
        
#         # Level 3
#         f3 = self.body3(f2)         # [B, 512, 64, 64]
#         out3 = self.proj_3(f3)      # [B, 1152, 32, 32]
        
#         # 返回列表
#         return [out0, out1, out2, out3]



# # 多尺度adapter
# import torch
# import torch.nn as nn

# # ResBlock 保持不变
# class ResBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.norm1 = nn.GroupNorm(32, channels)
#         self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
#         self.norm2 = nn.GroupNorm(32, channels)
#         self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
#         self.act = nn.SiLU()

#     def forward(self, x):
#         h = self.act(self.norm1(x))
#         h = self.conv1(h)
#         h = self.act(self.norm2(h))
#         h = self.conv2(h)
#         return x + h

# class MultiLevelAdapter(nn.Module):
#     def __init__(self, in_channels=4, hidden_size=1152):
#         super().__init__()
        
#         # 1. Stem (对应 Layer 0 注入)
#         self.stem = nn.Sequential(
#             nn.Conv2d(in_channels, 64, 3, padding=1),
#             ResBlock(64),
#             ResBlock(64)
#         )
#         self.proj_0 = self._make_projection(64, hidden_size)

#         # 2. Body 1 (对应 Layer 7 注入)
#         self.body1 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, padding=1),
#             ResBlock(128),
#             ResBlock(128)
#         )
#         self.proj_1 = self._make_projection(128, hidden_size)

#         # 3. Body 2 (对应 Layer 14 注入)
#         self.body2 = nn.Sequential(
#             nn.Conv2d(128, 256, 3, padding=1),
#             ResBlock(256),
#             ResBlock(256)
#         )
#         self.proj_2 = self._make_projection(256, hidden_size)

#         # 4. Body 3 (对应 Layer 21 注入)
#         self.body3 = nn.Sequential(
#             nn.Conv2d(256, 512, 3, padding=1),
#             ResBlock(512),
#             ResBlock(512)
#         )
#         self.proj_3 = self._make_projection(512, hidden_size)

#     def _make_projection(self, in_ch, out_ch):
#         # 必须包含 Stride=2 以匹配 PixArt 的 Token 数量 (1024)
#         # 必须包含 Zero-Init 以保证初始无损
#         proj = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=2),
#             nn.SiLU(),
#             nn.Conv2d(out_ch, out_ch, 1) # Zero-Init 放在这里
#         )
#         # 初始化最后一层为 0
#         nn.init.zeros_(proj[-1].weight)
#         nn.init.zeros_(proj[-1].bias)
#         return proj

#     def forward(self, x):
#         # x: [B, 4, 64, 64]
        
#         # Level 0
#         f0 = self.stem(x)           # [B, 64, 64, 64]
#         out0 = self.proj_0(f0)      # [B, 1152, 32, 32] -> Flatten later
        
#         # Level 1
#         f1 = self.body1(f0)         # [B, 128, 64, 64]
#         out1 = self.proj_1(f1)      # [B, 1152, 32, 32]
        
#         # Level 2
#         f2 = self.body2(f1)         # [B, 256, 64, 64]
#         out2 = self.proj_2(f2)      # [B, 1152, 32, 32]
        
#         # Level 3
#         f3 = self.body3(f2)         # [B, 512, 64, 64]
#         out3 = self.proj_3(f3)      # [B, 1152, 32, 32]
        
#         # 返回列表
#         return [out0, out1, out2, out3]

# v3
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class ResBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.norm1 = nn.GroupNorm(32, channels)
#         self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
#         self.norm2 = nn.GroupNorm(32, channels)
#         self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
#         self.act = nn.SiLU()

#     def forward(self, x):
#         h = self.act(self.norm1(x))
#         h = self.conv1(h)
#         h = self.act(self.norm2(h))
#         h = self.conv2(h)
#         return x + h


# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=4):
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(channels, channels // reduction, 1),
#             nn.SiLU(),
#             nn.Conv2d(channels // reduction, channels, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         w = self.pool(x)
#         w = self.fc(w)
#         return x * w


# class MultiLevelAdapterFPN(nn.Module):
#     def __init__(self, in_channels=4, hidden_size=1152, base_channels=128):
#         super().__init__()
#         self.stem = nn.Sequential(
#             nn.Conv2d(in_channels, 64, 3, padding=1),
#             ResBlock(64),
#             ResBlock(64),
#         )
#         self.body1 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, padding=1, stride=2),  # 64 -> 32
#             ResBlock(128),
#             ResBlock(128),
#         )
#         self.body2 = nn.Sequential(
#             nn.Conv2d(128, 256, 3, padding=1, stride=2),  # 32 -> 16
#             ResBlock(256),
#             ResBlock(256),
#         )
#         self.body3 = nn.Sequential(
#             nn.Conv2d(256, 512, 3, padding=1, stride=2),  # 16 -> 8
#             ResBlock(512),
#             ResBlock(512),
#         )

#         self.lat0 = nn.Conv2d(64, base_channels, 1)
#         self.lat1 = nn.Conv2d(128, base_channels, 1)
#         self.lat2 = nn.Conv2d(256, base_channels, 1)
#         self.lat3 = nn.Conv2d(512, base_channels, 1)

#         self.refine0 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
#         self.refine1 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
#         self.refine2 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
#         self.refine3 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))

#         self.scale_gates = nn.Parameter(torch.ones(4, dtype=torch.float32))
#         self.proj_0 = self._make_projection(base_channels, hidden_size, stride=2)
#         self.proj_1 = self._make_projection(base_channels, hidden_size, stride=1)
#         self.proj_2 = self._make_projection(base_channels, hidden_size, stride=1)
#         self.proj_3 = self._make_projection(base_channels, hidden_size, stride=1)

#     def _make_projection(self, in_ch, out_ch, stride):
#         proj = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride),
#             nn.SiLU(),
#             nn.Conv2d(out_ch, out_ch, 1),
#         )
#         nn.init.zeros_(proj[-1].weight)
#         nn.init.zeros_(proj[-1].bias)
#         return proj

#     def _fpn(self, f0, f1, f2, f3):
#         p3 = self.lat3(f3)
#         p2 = self.lat2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
#         p1 = self.lat1(f1) + F.interpolate(p2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
#         p0 = self.lat0(f0) + F.interpolate(p1, size=f0.shape[-2:], mode="bilinear", align_corners=False)
#         p0 = self.refine0(p0)
#         p1 = self.refine1(p1)
#         p2 = self.refine2(p2)
#         p3 = self.refine3(p3)
#         return p0, p1, p2, p3

#     def forward(self, x):
#         f0 = self.stem(x)   # 64x64
#         f1 = self.body1(f0) # 32x32
#         f2 = self.body2(f1) # 16x16
#         f3 = self.body3(f2) # 8x8

#         p0, p1, p2, p3 = self._fpn(f0, f1, f2, f3)

#         out0 = self.proj_0(p0) * self.scale_gates[0]
#         out1 = self.proj_1(p1) * self.scale_gates[1]
#         out2 = self.proj_2(F.interpolate(p2, size=(32, 32), mode="bilinear", align_corners=False)) * self.scale_gates[2]
#         out3 = self.proj_3(F.interpolate(p3, size=(32, 32), mode="bilinear", align_corners=False)) * self.scale_gates[3]
#         return [out0, out1, out2, out3]


# class MultiLevelAdapterSE(MultiLevelAdapterFPN):
#     def __init__(self, in_channels=4, hidden_size=1152, base_channels=128):
#         super().__init__(in_channels=in_channels, hidden_size=hidden_size, base_channels=base_channels)
#         self.se0 = SEBlock(base_channels)
#         self.se1 = SEBlock(base_channels)
#         self.se2 = SEBlock(base_channels)
#         self.se3 = SEBlock(base_channels)

#     def _fpn(self, f0, f1, f2, f3):
#         p0, p1, p2, p3 = super()._fpn(f0, f1, f2, f3)
#         p0 = self.se0(p0)
#         p1 = self.se1(p1)
#         p2 = self.se2(p2)
#         p3 = self.se3(p3)
#         return p0, p1, p2, p3


# def build_adapter(adapter_type: str, in_channels: int = 4, hidden_size: int = 1152):
#     if adapter_type == "fpn":
#         return MultiLevelAdapterFPN(in_channels=in_channels, hidden_size=hidden_size)
#     if adapter_type == "fpn_se":
#         return MultiLevelAdapterSE(in_channels=in_channels, hidden_size=hidden_size)
#     raise ValueError(f"Unknown adapter_type={adapter_type}")


# # Backward compatibility: older training scripts import MultiLevelAdapter directly.
# MultiLevelAdapter = MultiLevelAdapterFPN

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class ResBlock(nn.Module):
#     def __init__(self, channels: int):
#         super().__init__()
#         self.norm1 = nn.GroupNorm(32, channels)
#         self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
#         self.norm2 = nn.GroupNorm(32, channels)
#         self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
#         self.act = nn.SiLU()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         h = self.act(self.norm1(x))
#         h = self.conv1(h)
#         h = self.act(self.norm2(h))
#         h = self.conv2(h)
#         return x + h


# class SEBlock(nn.Module):
#     def __init__(self, channels: int, reduction: int = 4):
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(channels, channels // reduction, 1),
#             nn.SiLU(),
#             nn.Conv2d(channels // reduction, channels, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         w = self.pool(x)
#         w = self.fc(w)
#         return x * w


# class MultiLevelAdapterFPN(nn.Module):
#     """
#     多尺度 latent Adapter：
#       - 输入:  [B, in_channels(4), 64, 64]  (VAE latent, bicubic 对齐)
#       - 主干:  stem(64x64) -> body1(32x32) -> body2(16x16) -> body3(8x8)
#       - FPN:   p3 -> p2 -> p1 -> p0 (全都 base_channels 通道)
#       - 输出:  4 个尺度特征, 都映射到 [B, hidden_size, 32, 32]
#                对应 PixArtMS block 0, 7, 14, 21 的注入
#       - 新增:  高频支路 (hf_conv + hf_scale), 在 64x64 上轻量注入高频信息
#     """
#     def __init__(self, in_channels: int = 4, hidden_size: int = 1152, base_channels: int = 128):
#         super().__init__()
#         # ---- stem: 64x64 ----
#         self.stem = nn.Sequential(
#             nn.Conv2d(in_channels, 64, 3, padding=1),
#             ResBlock(64),
#             ResBlock(64),
#         )

#         # 高频支路: 只在 64x64 上提取高频，然后加到 stem 上
#         self.hf_conv = nn.Sequential(
#             nn.Conv2d(in_channels, 64, 3, padding=1),
#             nn.SiLU(),
#         )
#         # 标量 gate，初始为 0，训练过程自己学要不要用这条路
#         self.hf_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

#         # ---- pyramid body ----
#         # 64 -> 32
#         self.body1 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, padding=1, stride=2),
#             ResBlock(128),
#             ResBlock(128),
#         )
#         # 32 -> 16
#         self.body2 = nn.Sequential(
#             nn.Conv2d(128, 256, 3, padding=1, stride=2),
#             ResBlock(256),
#             ResBlock(256),
#         )
#         # 16 -> 8
#         self.body3 = nn.Sequential(
#             nn.Conv2d(256, 512, 3, padding=1, stride=2),
#             ResBlock(512),
#             ResBlock(512),
#         )

#         # lateral convs
#         self.lat0 = nn.Conv2d(64,  base_channels, 1)
#         self.lat1 = nn.Conv2d(128, base_channels, 1)
#         self.lat2 = nn.Conv2d(256, base_channels, 1)
#         self.lat3 = nn.Conv2d(512, base_channels, 1)

#         # refine blocks for each FPN level
#         self.refine0 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
#         self.refine1 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
#         self.refine2 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
#         self.refine3 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))

#         # per-scale gates (learnable)
#         self.scale_gates = nn.Parameter(torch.ones(4, dtype=torch.float32))

#         # 将每个尺度的特征映射到 hidden_size，空间统一为 32x32
#         self.proj_0 = self._make_projection(base_channels, hidden_size, stride=2)  # 64->32
#         self.proj_1 = self._make_projection(base_channels, hidden_size, stride=1)  # 32->32
#         self.proj_2 = self._make_projection(base_channels, hidden_size, stride=1)  # 16->16 再上采样
#         self.proj_3 = self._make_projection(base_channels, hidden_size, stride=1)  #  8-> 8 再上采样

#     def _make_projection(self, in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
#         proj = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride),
#             nn.SiLU(),
#             nn.Conv2d(out_ch, out_ch, 1),
#         )
#         # 最后一层 conv 初始为 0，保证一开始不破坏 PixArt 预训练分布
#         nn.init.zeros_(proj[-1].weight)
#         nn.init.zeros_(proj[-1].bias)
#         return proj

#     def _fpn(self, f0: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor):
#         # top-down: 8 -> 16 -> 32 -> 64
#         p3 = self.lat3(f3)
#         p2 = self.lat2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
#         p1 = self.lat1(f1) + F.interpolate(p2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
#         p0 = self.lat0(f0) + F.interpolate(p1, size=f0.shape[-2:], mode="bilinear", align_corners=False)

#         # local refinement
#         p0 = self.refine0(p0)
#         p1 = self.refine1(p1)
#         p2 = self.refine2(p2)
#         p3 = self.refine3(p3)
#         return p0, p1, p2, p3

#     def forward(self, x: torch.Tensor):
#         """
#         x: VAE latent of LR, shape [B, in_channels(4), 64, 64]
#         return: 4 feature maps, each [B, hidden_size, 32, 32]
#         """
#         # 高频分支: (x - 平滑(x)) 做 3x3 高通
#         with torch.cuda.amp.autocast(enabled=False):
#             hf = x.float() - F.avg_pool2d(x.float(), kernel_size=3, stride=1, padding=1)
#             hf = self.hf_conv(hf)  # [B,64,64,64]

#         f0 = self.stem(x) + self.hf_scale * hf  # 64x64
#         f1 = self.body1(f0)                     # 32x32
#         f2 = self.body2(f1)                     # 16x16
#         f3 = self.body3(f2)                     #  8x8

#         p0, p1, p2, p3 = self._fpn(f0, f1, f2, f3)

#         out0 = self.proj_0(p0) * self.scale_gates[0]
#         out1 = self.proj_1(p1) * self.scale_gates[1]
#         out2 = self.proj_2(F.interpolate(p2, size=(32, 32), mode="bilinear", align_corners=False)) * self.scale_gates[2]
#         out3 = self.proj_3(F.interpolate(p3, size=(32, 32), mode="bilinear", align_corners=False)) * self.scale_gates[3]

#         return [out0, out1, out2, out3]


# class MultiLevelAdapterSE(MultiLevelAdapterFPN):
#     """
#     在 FPN 基础上加 SEBlock，对每个尺度做通道注意力。
#     """
#     def __init__(self, in_channels: int = 4, hidden_size: int = 1152, base_channels: int = 128):
#         super().__init__(in_channels=in_channels, hidden_size=hidden_size, base_channels=base_channels)
#         self.se0 = SEBlock(base_channels)
#         self.se1 = SEBlock(base_channels)
#         self.se2 = SEBlock(base_channels)
#         self.se3 = SEBlock(base_channels)

#     def _fpn(self, f0, f1, f2, f3):
#         p0, p1, p2, p3 = super()._fpn(f0, f1, f2, f3)
#         p0 = self.se0(p0)
#         p1 = self.se1(p1)
#         p2 = self.se2(p2)
#         p3 = self.se3(p3)
#         return p0, p1, p2, p3


# def build_adapter(
#     adapter_type: str,
#     in_channels: int = 4,
#     hidden_size: int = 1152,
#     base_channels: int = 128,
# ):
#     """
#     工厂函数：
#       - adapter_type = "fpn" / "fpn_se"
#       - in_channels   默认 4 (VAE latent)
#       - hidden_size   默认 1152 (PixArtMS 的 hidden size)
#       - base_channels 默认 128，可在训练脚本中改小以适配 12G 显存
#     """
#     if adapter_type == "fpn":
#         return MultiLevelAdapterFPN(in_channels=in_channels, hidden_size=hidden_size, base_channels=base_channels)
#     if adapter_type == "fpn_se":
#         return MultiLevelAdapterSE(in_channels=in_channels, hidden_size=hidden_size, base_channels=base_channels)
#     raise ValueError(f"Unknown adapter_type={adapter_type}")


# # 兼容旧脚本: 旧的 MultiLevelAdapter = 单尺度版本
# MultiLevelAdapter = MultiLevelAdapterFPN

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return x + h


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class MultiLevelAdapterFPN(nn.Module):
    def __init__(self, in_channels=4, hidden_size=1152, base_channels=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            ResBlock(64),
            ResBlock(64),
        )
        self.body1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, stride=2),  # 64 -> 32
            ResBlock(128),
            ResBlock(128),
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, stride=2),  # 32 -> 16
            ResBlock(256),
            ResBlock(256),
        )
        self.body3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, stride=2),  # 16 -> 8
            ResBlock(512),
            ResBlock(512),
        )
        self.body4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, stride=2),  # 8 -> 4
            ResBlock(512),
            ResBlock(512),
        )

        self.lat0 = nn.Conv2d(64, base_channels, 1)
        self.lat1 = nn.Conv2d(128, base_channels, 1)
        self.lat2 = nn.Conv2d(256, base_channels, 1)
        self.lat3 = nn.Conv2d(512, base_channels, 1)
        self.lat4 = nn.Conv2d(512, base_channels, 1)

        self.refine0 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
        self.refine1 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
        self.refine2 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
        self.refine3 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
        self.refine4 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))

        self.scale_gates = nn.Parameter(torch.ones(8, dtype=torch.float32))
        self.proj_0 = self._make_projection(base_channels, hidden_size, stride=2)
        self.proj_1 = self._make_projection(base_channels, hidden_size, stride=1)
        self.proj_2 = self._make_projection(base_channels, hidden_size, stride=1)
        self.proj_3 = self._make_projection(base_channels, hidden_size, stride=1)
        self.proj_4 = self._make_projection(base_channels, hidden_size, stride=1)
        self.proj_5 = self._make_projection(base_channels, hidden_size, stride=1)
        self.proj_6 = self._make_projection(base_channels, hidden_size, stride=1)
        self.proj_7 = self._make_projection(base_channels, hidden_size, stride=1)

    def _make_projection(self, in_ch, out_ch, stride):
        proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 1),
        )
        nn.init.zeros_(proj[-1].weight)
        nn.init.zeros_(proj[-1].bias)
        return proj

    def _fpn(self, f0, f1, f2, f3, f4):
        p4 = self.lat4(f4)
        p3 = self.lat3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lat2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lat1(f1) + F.interpolate(p2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        p0 = self.lat0(f0) + F.interpolate(p1, size=f0.shape[-2:], mode="bilinear", align_corners=False)
        p0 = self.refine0(p0)
        p1 = self.refine1(p1)
        p2 = self.refine2(p2)
        p3 = self.refine3(p3)
        p4 = self.refine4(p4)
        return p0, p1, p2, p3, p4

    def forward(self, x):
        f0 = self.stem(x)   # 64x64
        f1 = self.body1(f0) # 32x32
        f2 = self.body2(f1) # 16x16
        f3 = self.body3(f2) # 8x8
        f4 = self.body4(f3) # 4x4

        p0, p1, p2, p3, p4 = self._fpn(f0, f1, f2, f3, f4)
        p2_up = F.interpolate(p2, size=(32, 32), mode="bilinear", align_corners=False)
        p3_up = F.interpolate(p3, size=(32, 32), mode="bilinear", align_corners=False)
        p4_up = F.interpolate(p4, size=(32, 32), mode="bilinear", align_corners=False)

        out0 = self.proj_0(p0) * self.scale_gates[0]
        out1 = self.proj_1(p1) * self.scale_gates[1]
        out2 = self.proj_2(p2_up) * self.scale_gates[2]
        out3 = self.proj_3(p3_up) * self.scale_gates[3]
        out4 = self.proj_4(p4_up) * self.scale_gates[4]
        out5 = self.proj_5(p3_up) * self.scale_gates[5]
        out6 = self.proj_6(p2_up) * self.scale_gates[6]
        out7 = self.proj_7(p4_up) * self.scale_gates[7]
        return [out0, out1, out2, out3, out4, out5, out6, out7]


class MultiLevelAdapterSE(MultiLevelAdapterFPN):
    def __init__(self, in_channels=4, hidden_size=1152, base_channels=128):
        super().__init__(in_channels=in_channels, hidden_size=hidden_size, base_channels=base_channels)
        self.se0 = SEBlock(base_channels)
        self.se1 = SEBlock(base_channels)
        self.se2 = SEBlock(base_channels)
        self.se3 = SEBlock(base_channels)
        self.se4 = SEBlock(base_channels)

    def _fpn(self, f0, f1, f2, f3, f4):
        p0, p1, p2, p3, p4 = super()._fpn(f0, f1, f2, f3, f4)
        p0 = self.se0(p0)
        p1 = self.se1(p1)
        p2 = self.se2(p2)
        p3 = self.se3(p3)
        p4 = self.se4(p4)
        return p0, p1, p2, p3, p4


class MultiLevelAdapterFPNHF(MultiLevelAdapterFPN):
    def __init__(self, in_channels=4, hidden_size=1152, base_channels=128):
        super().__init__(in_channels=in_channels, hidden_size=hidden_size, base_channels=base_channels)
        kernel = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=torch.float32)
        self.register_buffer("hf_kernel", kernel.view(1, 1, 3, 3))
        self.hf_body = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, stride=2),
            nn.SiLU(),
            ResBlock(base_channels),
        )
        self.hf_proj = self._make_projection(base_channels, hidden_size, stride=1)

    def _laplacian(self, x):
        c = x.shape[1]
        weight = self.hf_kernel.repeat(c, 1, 1, 1)
        return F.conv2d(x, weight, padding=1, groups=c)

    def forward(self, x):
        features = super().forward(x)
        hf = self._laplacian(x)
        hf = self.hf_body(hf)
        hf = self.hf_proj(hf)
        if len(features) >= 8:
            features = list(features)
            features[-1] = hf
            return features
        return features + [hf]


class MultiLevelAdapterV6(nn.Module):
    """V6 adapter: 4 spatial features + 1 global style vector."""
    def __init__(self, in_channels=4, hidden_size=1152, base_channels=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            ResBlock(64),
            ResBlock(64),
        )
        self.body1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, stride=2),  # 64 -> 32
            ResBlock(128),
            ResBlock(128),
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, stride=2),  # 32 -> 16
            ResBlock(256),
            ResBlock(256),
        )
        self.body3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, stride=2),  # 16 -> 8
            ResBlock(512),
            ResBlock(512),
        )

        self.lat0 = nn.Conv2d(64, base_channels, 1)
        self.lat1 = nn.Conv2d(128, base_channels, 1)
        self.lat2 = nn.Conv2d(256, base_channels, 1)
        self.lat3 = nn.Conv2d(512, base_channels, 1)

        self.refine0 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
        self.refine1 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
        self.refine2 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
        self.refine3 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))

        self.scale_gates = nn.Parameter(torch.ones(4, dtype=torch.float32))
        self.proj_0 = self._make_projection(base_channels, hidden_size, stride=2)
        self.proj_1 = self._make_projection(base_channels, hidden_size, stride=1)
        self.proj_2 = self._make_projection(base_channels, hidden_size, stride=1)
        self.proj_3 = self._make_projection(base_channels, hidden_size, stride=1)

        # Style head: global texture/statistics vector (no spatial constraint).
        self.style_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _make_projection(self, in_ch, out_ch, stride):
        proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 1),
        )
        nn.init.zeros_(proj[-1].weight)
        nn.init.zeros_(proj[-1].bias)
        return proj

    def _fpn(self, f0, f1, f2, f3):
        p3 = self.lat3(f3)
        p2 = self.lat2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lat1(f1) + F.interpolate(p2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        p0 = self.lat0(f0) + F.interpolate(p1, size=f0.shape[-2:], mode="bilinear", align_corners=False)
        p0 = self.refine0(p0)
        p1 = self.refine1(p1)
        p2 = self.refine2(p2)
        p3 = self.refine3(p3)
        return p0, p1, p2, p3

    def forward(self, x):
        f0 = self.stem(x)   # 64x64
        f1 = self.body1(f0) # 32x32
        f2 = self.body2(f1) # 16x16
        f3 = self.body3(f2) # 8x8

        p0, p1, p2, p3 = self._fpn(f0, f1, f2, f3)
        p2_up = F.interpolate(p2, size=(32, 32), mode="bilinear", align_corners=False)
        p3_up = F.interpolate(p3, size=(32, 32), mode="bilinear", align_corners=False)

        out0 = self.proj_0(p0) * self.scale_gates[0]
        out1 = self.proj_1(p1) * self.scale_gates[1]
        out2 = self.proj_2(p2_up) * self.scale_gates[2]
        out3 = self.proj_3(p3_up) * self.scale_gates[3]
        style_vec = self.style_head(f2)
        return [out0, out1, out2, out3, style_vec]

def build_adapter(adapter_type: str, in_channels: int = 4, hidden_size: int = 1152):
    if adapter_type == "fpn":
        return MultiLevelAdapterFPN(in_channels=in_channels, hidden_size=hidden_size)
    if adapter_type == "fpn_se":
        return MultiLevelAdapterSE(in_channels=in_channels, hidden_size=hidden_size)
    if adapter_type == "fpn_hf":
        return MultiLevelAdapterFPNHF(in_channels=in_channels, hidden_size=hidden_size)
    if adapter_type == "fpn_v6":
        return MultiLevelAdapterV6(in_channels=in_channels, hidden_size=hidden_size)
    raise ValueError(f"Unknown adapter_type={adapter_type}")


# Backward compatibility: older training scripts import MultiLevelAdapter directly.
MultiLevelAdapter = MultiLevelAdapterFPN
