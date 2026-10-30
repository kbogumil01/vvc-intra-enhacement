"""
WCDANN-YUV: Weakly Connected Dense Attention Neural Network
extended with U/V channels and unified QP.

Based on the original WCDANN architecture (Shen et al., 2023):
  Y. Shen et al., "WCDANN: A Lightweight CNN Post-Processing Filter for VVC",
  IEEE Access, 2023.
- DSConv (Depthwise Separable Convolution)
- CAB (Channel Attention Block)
- SAB (Spatial Attention Block)
- CSAB (Channel-Spatial Joint Attention Block)
- RAB (Residual Attention Block)
- WCDAB (Weakly Connected Dense Attention Block)

Modifications vs original:
- Input: 4 channels (YUV444 + QP map) instead of 1 (Y) or 2 (Y+QP)
- Output: 3 channels (YUV444) instead of 1 (Y)
- Loss: per-channel weighted (Y=2/3, U=1/6, V=1/6)
- Data: YUV420p10le from our pipeline (dataset.py)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_msssim import ssim, ms_ssim

from registry import register_model


class DSConv(nn.Module):
    """Depthwise Separable Convolution + ReLU."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.pointwise(self.depthwise(x)))


class CAB(nn.Module):
    """Channel Attention Block."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(self.avg_pool(x))


class SAB(nn.Module):
    """Spatial Attention Block (depthwise variant)."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv3x3_dw = nn.Conv2d(channels, channels, 3, padding=1,
                                     groups=channels, bias=False)
        self.conv1x1_dw_1 = nn.Conv2d(channels, channels, 1,
                                       groups=channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1_dw_2 = nn.Conv2d(channels, channels, 1,
                                       groups=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv3x3_dw(x) + self.conv1x1_dw_1(x))
        mask = self.sigmoid(self.conv1x1_dw_2(out))
        return x * mask


class CSAB(nn.Module):
    """Channel-Spatial Joint Attention Block."""
    def __init__(self, channels: int):
        super().__init__()
        self.cab = CAB(channels)
        self.sab = SAB(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cab(x) + self.sab(x)


class RAB(nn.Module):
    """Residual Attention Block (dual-branch + channel shuffle)."""
    def __init__(self, channels: int):
        super().__init__()
        self.branch1 = DSConv(channels, channels)
        self.branch2 = nn.Sequential(
            DSConv(channels, channels),
            DSConv(channels, channels),
        )
        self.conv1x1 = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.cab = CAB(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = torch.cat([out1, out2], dim=1)
        out = F.channel_shuffle(out, groups=2)
        out = self.cab(self.conv1x1(out))
        return x + out


class WCDAB(nn.Module):
    """Weakly Connected Dense Attention Block (4 RABs + dense concat)."""
    def __init__(self, channels: int):
        super().__init__()
        self.rab1 = RAB(channels)
        self.rab2 = RAB(channels)
        self.rab3 = RAB(channels)
        self.rab4 = RAB(channels)
        self.conv1x1 = nn.Conv2d(channels * 5, channels, 1, bias=False)
        self.csab = CSAB(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.rab1(x)
        out2 = self.rab2(out1)
        out3 = self.rab3(out2)
        out4 = self.rab4(out3)
        cat_out = torch.cat([x, out1, out2, out3, out4], dim=1)
        res = self.csab(self.conv1x1(cat_out))
        return x + res


@dataclass
class WCDANNConfig:
    in_channels: int = 4
    out_channels: int = 3
    channels: int = 64
    num_wcdab: int = 6


class WCDANNEnhancer(nn.Module):
    """WCDANN extended with YUV + QP input."""

    def __init__(self, cfg: WCDANNConfig = WCDANNConfig()):
        super().__init__()
        c = cfg.channels

        self.head = nn.Sequential(
            nn.Conv2d(cfg.in_channels, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.backbone = nn.ModuleList([WCDAB(c) for _ in range(cfg.num_wcdab)])

        self.tail = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, cfg.out_channels, 3, padding=1),
        )

        nn.init.zeros_(self.tail[-1].weight)
        nn.init.zeros_(self.tail[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 4, H, W] (YUV + QP) → [B, 3, H, W] (enhanced YUV)."""
        yuv = x[:, :3, :, :]
        feat = self.head(x)
        for block in self.backbone:
            feat = block(feat)
        res = self.tail(feat)
        return yuv + res


class WCDANNLoss(nn.Module):
    """Per-channel weighted L1 loss (Y=0.8, U=0.1, V=0.1).
    SSIM logged but not in the gradient."""

    def __init__(self, data_range: float = 1.0, ssim_win_size: int = 9):
        super().__init__()
        self.data_range = data_range
        self.ssim_win_size = ssim_win_size
        self.register_buffer(
            "channel_weights",
            torch.tensor([0.8, 0.1, 0.1]),
        )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cw = self.channel_weights

        l1_per_ch = torch.stack([
            F.l1_loss(pred[:, c], target[:, c]) for c in range(3)
        ])
        total = (cw * l1_per_ch).sum()

        with torch.no_grad():
            pred_c = torch.clamp(pred, 0.0, 1.0)
            ssim_per_ch = torch.stack([
                ssim(pred_c[:, c:c+1], target[:, c:c+1],
                     data_range=self.data_range, size_average=True,
                     win_size=self.ssim_win_size)
                for c in range(3)
            ])

        parts = {
            "loss_total": total.detach(),
            "loss_l1_y": l1_per_ch[0].detach(),
            "loss_l1_u": l1_per_ch[1].detach(),
            "loss_l1_v": l1_per_ch[2].detach(),
            "ssim_y": ssim_per_ch[0].detach(),
            "ssim_u": ssim_per_ch[1].detach(),
            "ssim_v": ssim_per_ch[2].detach(),
        }
        return total, parts


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    return 10.0 * torch.log10(1.0 / (mse + 1e-10))


def compute_y_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred[:, 0], target[:, 0])
    return 10.0 * torch.log10(1.0 / (mse + 1e-10))


class WCDANNLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_config: Optional[WCDANNConfig] = None,
        lr: float = 1e-4,
        weight_decay: Optional[float] = None,
        model_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model_config"])
        self.model_config = model_config or WCDANNConfig()
        self.model = WCDANNEnhancer(self.model_config)
        self.criterion = WCDANNLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        x = batch["input"]
        y = batch["target"]
        pred = self(x)
        loss, parts = self.criterion(pred, y)

        pred_clamped = torch.clamp(pred, 0.0, 1.0)
        psnr = compute_psnr(pred_clamped, y)
        y_psnr = compute_y_psnr(pred_clamped, y)

        self.log(f"{stage}/loss", loss, prog_bar=True,
                 on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}/psnr", psnr, prog_bar=True,
                 on_step=False, on_epoch=True)
        self.log(f"{stage}/y_psnr", y_psnr, prog_bar=True,
                 on_step=False, on_epoch=True)

        for k, v in parts.items():
            self.log(f"{stage}/{k}", v, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


register_model(
    name="wcdann_yuv",
    model_cls=WCDANNEnhancer,
    lightning_cls=WCDANNLightningModule,
    config_cls=WCDANNConfig,
    in_channels=4,
    out_channels=3,
    channels=64,
    num_wcdab=6,
)

register_model(
    name="wcdann_yuv_small",
    model_cls=WCDANNEnhancer,
    lightning_cls=WCDANNLightningModule,
    config_cls=WCDANNConfig,
    in_channels=4,
    out_channels=3,
    channels=64,
    num_wcdab=3,
)
