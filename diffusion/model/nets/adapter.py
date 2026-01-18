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
import torch
import torch.nn as nn
import torch.nn.functional as F

# ResBlock 保持不变
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

class MultiLevelAdapter(nn.Module):
    def __init__(self, in_channels=4, hidden_size=1152):
        super().__init__()
        base_channels = 128
        
        # 1. Stem (64x64, 对应 Layer 0 注入)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            ResBlock(64),
            ResBlock(64)
        )

        # 2. Body 1 (32x32, 对应 Layer 7 注入)
        self.body1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            ResBlock(128),
            ResBlock(128)
        )

        # 3. Body 2 (16x16, 对应 Layer 14 注入)
        self.body2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, stride=2),
            ResBlock(256),
            ResBlock(256)
        )

        # 4. Body 3 (8x8, 对应 Layer 21 注入)
        self.body3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, stride=2),
            ResBlock(512),
            ResBlock(512)
        )

        # FPN-style lateral projections (统一到 base_channels)
        self.lat0 = nn.Conv2d(64, base_channels, 1)
        self.lat1 = nn.Conv2d(128, base_channels, 1)
        self.lat2 = nn.Conv2d(256, base_channels, 1)
        self.lat3 = nn.Conv2d(512, base_channels, 1)

        # FPN refinement blocks
        self.refine0 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
        self.refine1 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
        self.refine2 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
        self.refine3 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))

        # 32x32 token-aligned projections (zero-init)
        self.proj_0 = self._make_projection(base_channels, hidden_size, stride=2)
        self.proj_1 = self._make_projection(base_channels, hidden_size, stride=1)
        self.proj_2 = self._make_projection(base_channels, hidden_size, stride=1)
        self.proj_3 = self._make_projection(base_channels, hidden_size, stride=1)

        # per-scale gates (learned)
        self.scale_gates = nn.Parameter(torch.ones(4, dtype=torch.float32))

    def _make_projection(self, in_ch, out_ch, stride: int):
        # stride=2: 64x64 -> 32x32 (token 对齐)
        # stride=1: 已经是 32x32
        proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 1) # Zero-Init 放在这里
        )
        # 初始化最后一层为 0
        nn.init.zeros_(proj[-1].weight)
        nn.init.zeros_(proj[-1].bias)
        return proj

    def forward(self, x):
        # x: [B, 4, 64, 64]
        
        # Bottom-up
        f0 = self.stem(x)           # [B, 64, 64, 64]
        f1 = self.body1(f0)         # [B, 128, 32, 32]
        f2 = self.body2(f1)         # [B, 256, 16, 16]
        f3 = self.body3(f2)         # [B, 512, 8, 8]

        # Lateral (统一通道) + Top-down fusion
        p3 = self.lat3(f3)
        p2 = self.lat2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lat1(f1) + F.interpolate(p2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        p0 = self.lat0(f0) + F.interpolate(p1, size=f0.shape[-2:], mode="bilinear", align_corners=False)

        # Refine each scale
        p0 = self.refine0(p0)
        p1 = self.refine1(p1)
        p2 = self.refine2(p2)
        p3 = self.refine3(p3)

        # Token-aligned outputs (32x32)
        out0 = self.proj_0(p0) * self.scale_gates[0]
        out1 = self.proj_1(p1) * self.scale_gates[1]

        p2_up = F.interpolate(p2, size=(32, 32), mode="bilinear", align_corners=False)
        out2 = self.proj_2(p2_up) * self.scale_gates[2]

        p3_up = F.interpolate(p3, size=(32, 32), mode="bilinear", align_corners=False)
        out3 = self.proj_3(p3_up) * self.scale_gates[3]
        
        # 返回列表
        return [out0, out1, out2, out3]
