import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.model.nets.adapter import ResBlock


class StyleExtractor(nn.Module):
    """Global style extractor for v6 (captures degradation/style statistics)."""

    def __init__(self, in_channels: int, hidden_dim: int = 1152):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = torch.flatten(feat, 1)
        return self.proj(feat)


class MultiLevelAdapterV6(nn.Module):
    """Dual-stream adapter: spatial pyramid + global style vector."""

    def __init__(self, in_channels: int = 4, hidden_size: int = 1152, base_channels: int = 128):
        super().__init__()
        self.style_extractor = StyleExtractor(in_channels, hidden_size)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            ResBlock(base_channels),
            ResBlock(base_channels),
        )
        self.body1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            ResBlock(base_channels * 2),
            ResBlock(base_channels * 2),
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            ResBlock(base_channels * 4),
            ResBlock(base_channels * 4),
        )
        self.body3 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1),
            ResBlock(base_channels * 8),
            ResBlock(base_channels * 8),
        )

        self.lat0 = nn.Conv2d(base_channels, base_channels, 1)
        self.lat1 = nn.Conv2d(base_channels * 2, base_channels, 1)
        self.lat2 = nn.Conv2d(base_channels * 4, base_channels, 1)
        self.lat3 = nn.Conv2d(base_channels * 8, base_channels, 1)

        self.refine0 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
        self.refine1 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
        self.refine2 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))
        self.refine3 = nn.Sequential(ResBlock(base_channels), ResBlock(base_channels))

        self.proj_0 = self._make_projection(base_channels, hidden_size, stride=2)
        self.proj_1 = self._make_projection(base_channels, hidden_size, stride=1)
        self.proj_2 = self._make_projection(base_channels, hidden_size, stride=1)
        self.proj_3 = self._make_projection(base_channels, hidden_size, stride=1)

    def _make_projection(self, in_c: int, out_c: int, stride: int) -> nn.Sequential:
        proj = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_c, out_c, 1),
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

    def forward(self, x: torch.Tensor):
        style_vec = self.style_extractor(x)

        f0 = self.stem(x)
        f1 = self.body1(f0)
        f2 = self.body2(f1)
        f3 = self.body3(f2)

        p0, p1, p2, p3 = self._fpn(f0, f1, f2, f3)
        p2_up = F.interpolate(p2, size=(32, 32), mode="bilinear", align_corners=False)
        p3_up = F.interpolate(p3, size=(32, 32), mode="bilinear", align_corners=False)

        out0 = self.proj_0(p0)
        out1 = self.proj_1(p1)
        out2 = self.proj_2(p2_up)
        out3 = self.proj_3(p3_up)
        return [out0, out1, out2, out3], style_vec


def build_adapter_v6(in_channels: int = 4, hidden_size: int = 1152, base_channels: int = 128) -> MultiLevelAdapterV6:
    return MultiLevelAdapterV6(
        in_channels=in_channels,
        hidden_size=hidden_size,
        base_channels=base_channels,
    )
