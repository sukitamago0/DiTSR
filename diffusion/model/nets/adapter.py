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

import torch
import torch.nn as nn

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
        
        # 1. Stem (对应 Layer 0 注入)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            ResBlock(64),
            ResBlock(64)
        )
        self.proj_0 = self._make_projection(64, hidden_size)

        # 2. Body 1 (对应 Layer 7 注入)
        self.body1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            ResBlock(128),
            ResBlock(128)
        )
        self.proj_1 = self._make_projection(128, hidden_size)

        # 3. Body 2 (对应 Layer 14 注入)
        self.body2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            ResBlock(256),
            ResBlock(256)
        )
        self.proj_2 = self._make_projection(256, hidden_size)

        # 4. Body 3 (对应 Layer 21 注入)
        self.body3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            ResBlock(512),
            ResBlock(512)
        )
        self.proj_3 = self._make_projection(512, hidden_size)

    def _make_projection(self, in_ch, out_ch):
        # 必须包含 Stride=2 以匹配 PixArt 的 Token 数量 (1024)
        # 必须包含 Zero-Init 以保证初始无损
        proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 1) # Zero-Init 放在这里
        )
        # 初始化最后一层为 0
        nn.init.zeros_(proj[-1].weight)
        nn.init.zeros_(proj[-1].bias)
        return proj

    def forward(self, x):
        # x: [B, 4, 64, 64]
        
        # Level 0
        f0 = self.stem(x)           # [B, 64, 64, 64]
        out0 = self.proj_0(f0)      # [B, 1152, 32, 32] -> Flatten later
        
        # Level 1
        f1 = self.body1(f0)         # [B, 128, 64, 64]
        out1 = self.proj_1(f1)      # [B, 1152, 32, 32]
        
        # Level 2
        f2 = self.body2(f1)         # [B, 256, 64, 64]
        out2 = self.proj_2(f2)      # [B, 1152, 32, 32]
        
        # Level 3
        f3 = self.body3(f2)         # [B, 512, 64, 64]
        out3 = self.proj_3(f3)      # [B, 1152, 32, 32]
        
        # 返回列表
        return [out0, out1, out2, out3]