from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn

from model_v2 import compute_psnr, compute_y_psnr
from registry import register_model


@dataclass
class PPFFConfig:
    in_channels: int = 4
    out_channels: int = 3
    num_blocks: int = 16
    channels: int = 128
    use_global_residual: bool = True
    use_tanh: bool = True


class FeatureExtractionBlock(nn.Module):
    """
    A single Feature Extraction Block (FEB) with a 3x3 convolution and PReLU.
    """
    def __init__(self, channels=128):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, bias=True)
        self.prelu = nn.PReLU(num_parameters=channels)

    def forward(self, x):
        return self.prelu(self.conv(x))


class VVCPPFF(nn.Module):
    """
    The complete VVC-PPFF architecture with progressive feature fusion.
    """
    def __init__(self, cfg: PPFFConfig) -> None:
        super().__init__()
        self.cfg = cfg
        channels = cfg.channels

        self.initial_conv = nn.Conv2d(in_channels=cfg.in_channels, out_channels=channels, kernel_size=1, bias=True)
        self.initial_prelu = nn.PReLU(num_parameters=channels)

        self.febs = nn.ModuleList([
            FeatureExtractionBlock(channels) for _ in range(cfg.num_blocks)
        ])

        self.bottlenecks = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=1, bias=True) for _ in range(cfg.num_blocks)
        ])

        self.fusion_smoothing_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, bias=True)

        self.final_conv = nn.Conv2d(in_channels=channels, out_channels=cfg.out_channels, kernel_size=1, bias=True)
        self.use_tanh = cfg.use_tanh

        nn.init.zeros_(self.final_conv.bias)
        nn.init.normal_(self.final_conv.weight, mean=0.0, std=1e-3)

    def forward(self, x):
        original_yuv = x[:, :3, :, :]

        out = self.initial_prelu(self.initial_conv(x))

        fusion_accumulator = None
        current_feature = out

        for i in range(len(self.febs)):
            current_feature = self.febs[i](current_feature)
            bottleneck_feature = self.bottlenecks[i](current_feature)

            if fusion_accumulator is None:
                fusion_accumulator = bottleneck_feature
            else:
                fusion_accumulator = fusion_accumulator + bottleneck_feature

        out = self.fusion_smoothing_conv(fusion_accumulator)

        noise_residual = self.final_conv(out)
        if self.use_tanh:
            noise_residual = torch.tanh(noise_residual)

        if self.cfg.use_global_residual:
            return original_yuv + noise_residual
        return noise_residual


class PPFFLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_config: Optional[PPFFConfig] = None,
        lr: float = 1e-4,
        model_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model_config"])

        self.model_config = model_config or PPFFConfig()
        self.model = VVCPPFF(self.model_config)
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        x = batch["input"]
        y = batch["target"]
        pred = self(x)
        loss = self.criterion(pred, y)

        pred_clamped = torch.clamp(pred, 0.0, 1.0)
        psnr = compute_psnr(pred_clamped, y)
        y_psnr = compute_y_psnr(pred_clamped, y)

        self.log(f"{stage}/loss", loss, prog_bar=True,
                 on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}/psnr", psnr, prog_bar=True,
                 on_step=False, on_epoch=True)
        self.log(f"{stage}/y_psnr", y_psnr, prog_bar=True,
                 on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
        )


register_model(
    "ppff_base",
    config_cls=PPFFConfig,
    model_cls=VVCPPFF,
    lightning_cls=PPFFLightningModule,
    num_blocks=16,
    channels=128,
)


if __name__ == "__main__":
    cfg = PPFFConfig()
    model = VVCPPFF(cfg)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    x = torch.rand(2, 4, 132, 132)
    y = model(x)
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 3, 132, 132), "Zły output shape!"
    print("OK")
