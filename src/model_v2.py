from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim


class ConvLayer(nn.Sequential):
    """
    Convolutional layer:
      [ReflectionPad2d] -> Conv2d(bias=False) -> BN -> PReLU -> [Dropout2d]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        dropout: float = 0.0,
        reflect_padding: bool = True,
    ) -> None:
        layers: List[nn.Module] = []

        if reflect_padding and padding > 0:
            layers.append(nn.ReflectionPad2d(padding))
            conv_padding = 0
        else:
            conv_padding = padding

        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=conv_padding,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.PReLU())

        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))

        super().__init__(*layers)


class DenseLayer(nn.Module):
    """
    Dense layer:
      bottleneck (1x1) -> main conv (kxk) -> concat with input
    """

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        kernel_size: int = 3,
        padding: int = 1,
        dropout: float = 0.0,
        bn_size: float = 2.0,
    ) -> None:
        super().__init__()
        bottleneck_channels = int(bn_size * growth_rate)

        self.bottleneck = ConvLayer(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
            padding=0,
            reflect_padding=False,
        )
        self.conv = ConvLayer(
            in_channels=bottleneck_channels,
            out_channels=growth_rate,
            kernel_size=kernel_size,
            padding=padding,
            dropout=dropout,
            reflect_padding=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(self.bottleneck(x))
        return torch.cat((x, out), dim=1)


class DenseBlock(nn.Sequential):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        growth_rate: int,
        kernel_size: int = 3,
        padding: int = 1,
        dropout: float = 0.0,
        bn_size: float = 2.0,
    ) -> None:
        super().__init__()
        ch = in_channels
        for i in range(num_layers):
            self.add_module(
                f"dense_{i}",
                DenseLayer(
                    in_channels=ch,
                    growth_rate=growth_rate,
                    kernel_size=kernel_size,
                    padding=padding,
                    dropout=dropout,
                    bn_size=bn_size,
                ),
            )
            ch += growth_rate


class Transition(nn.Sequential):
    """
    Transition layer — halves the number of channels.
    Conv(1x1, bias=False) -> BN -> PReLU.
    """

    def __init__(self, in_channels: int, out_channels: Optional[int] = None) -> None:
        super().__init__()
        out_channels = out_channels or (in_channels // 2)
        self.add_module(
            "conv",
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )
        self.add_module("bn", nn.BatchNorm2d(out_channels))
        self.add_module("act", nn.PReLU())


@dataclass
class BlockCfg:
    num_layers: int
    kernel_size: int
    padding: int
    dropout: float = 0.0


@dataclass
class DenseVVCConfig:
    in_channels: int = 4
    out_channels: int = 3
    growth_rate: int = 16
    bn_size: float = 2.0

    stem_features: int = 64
    stem_kernel: int = 9
    stem_padding: int = 4
    stem_dense: bool = True

    blocks: List[BlockCfg] = field(default_factory=lambda: [
        BlockCfg(num_layers=4, kernel_size=7, padding=3),
        BlockCfg(num_layers=6, kernel_size=5, padding=2),
        BlockCfg(num_layers=4, kernel_size=3, padding=1),
        BlockCfg(num_layers=3, kernel_size=3, padding=1),
        BlockCfg(num_layers=2, kernel_size=3, padding=1),
    ])

    use_global_residual: bool = True


class DenseNetBackbone(nn.Module):
    def __init__(self, cfg: DenseVVCConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.stem = ConvLayer(
            in_channels=cfg.in_channels,
            out_channels=cfg.stem_features,
            kernel_size=cfg.stem_kernel,
            padding=cfg.stem_padding,
        )

        if cfg.stem_dense:
            ch = cfg.in_channels + cfg.stem_features
        else:
            ch = cfg.stem_features

        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i, bcfg in enumerate(cfg.blocks):
            block = DenseBlock(
                num_layers=bcfg.num_layers,
                in_channels=ch,
                growth_rate=cfg.growth_rate,
                kernel_size=bcfg.kernel_size,
                padding=bcfg.padding,
                dropout=bcfg.dropout,
                bn_size=cfg.bn_size,
            )
            self.blocks.append(block)
            ch += bcfg.num_layers * cfg.growth_rate

            if i < len(cfg.blocks) - 1:
                self.transitions.append(Transition(ch))
                ch = ch // 2

        self.output = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, cfg.out_channels, kernel_size=3, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem_out = self.stem(x)
        if self.cfg.stem_dense:
            h = torch.cat((x, stem_out), dim=1)
        else:
            h = stem_out
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i < len(self.transitions):
                h = self.transitions[i](h)
        return self.output(h)


class DenseVVCEnhancer(nn.Module):
    """
    input:  [B, 4, H, W]  =  YUV + QP-map
    output: [B, 3, H, W]  =  enhanced YUV
    """

    def __init__(self, cfg: Optional[DenseVVCConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or DenseVVCConfig()
        self.backbone = DenseNetBackbone(self.cfg)
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m: nn.Module) -> None:
        """Weight initialization: Conv normal(0, 0.02), BN normal(1, 0.02)."""
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        correction = self.backbone(x)
        if self.cfg.use_global_residual:
            out = x[:, :3, :, :] + correction
        else:
            out = correction
        return out


class DenseVVCLoss(nn.Module):
    """
    g_loss = 1.0 * mse_loss + 0.5 * l1_loss
    Equal channel weights: Y=1/3, U=1/3, V=1/3
    """

    def __init__(
        self,
        data_range: float = 1.0,
        ssim_win_size: int = 9,
        ms_ssim_win_size: int = 9,
    ) -> None:
        super().__init__()
        self.data_range = data_range
        self.ssim_win_size = ssim_win_size
        self.ms_ssim_win_size = ms_ssim_win_size
        self.register_buffer(
            "channel_weights",
            torch.tensor([0.8, 0.1, 0.1]),
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cw = self.channel_weights

        mse_per_ch = torch.stack([
            F.mse_loss(pred[:, c], target[:, c]) for c in range(3)
        ])
        l1_per_ch = torch.stack([
            F.l1_loss(pred[:, c], target[:, c]) for c in range(3)
        ])
        mse_loss = (cw * mse_per_ch).sum()
        l1_loss = (cw * l1_per_ch).sum()

        pred_c = torch.clamp(pred, 0.0, 1.0)
        ssim_per_ch = torch.stack([
            ssim(pred_c[:, c:c+1], target[:, c:c+1],
                 data_range=self.data_range, size_average=True,
                 win_size=self.ssim_win_size)
            for c in range(3)
        ])
        ms_ssim_per_ch = torch.stack([
            ms_ssim(pred_c[:, c:c+1], target[:, c:c+1],
                    data_range=self.data_range, size_average=True,
                    win_size=self.ms_ssim_win_size)
            for c in range(3)
        ])
        ssim_loss = (cw * (1.0 - ssim_per_ch)).sum()
        ms_ssim_loss = (cw * (1.0 - ms_ssim_per_ch)).sum()

        total = 1.0 * mse_loss + 0.5 * l1_loss

        parts = {
            "loss_total": total.detach(),
            "loss_l1": l1_loss.detach(),
            "loss_mse": mse_loss.detach(),
            "loss_ssim": ssim_loss.detach(),
            "loss_ms_ssim": ms_ssim_loss.detach(),
            "ssim_y": ssim_per_ch[0].detach(),
            "ssim_u": ssim_per_ch[1].detach(),
            "ssim_v": ssim_per_ch[2].detach(),
        }
        return total, parts


def compute_psnr(
    pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0
) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return (10.0 * torch.log10((max_val**2) / mse)).mean()


def compute_y_psnr(
    pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0
) -> torch.Tensor:
    pred_y = pred[:, 0:1, :, :]
    target_y = target[:, 0:1, :, :]
    mse = F.mse_loss(pred_y, target_y, reduction="none")
    mse = mse.mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return (10.0 * torch.log10((max_val**2) / mse)).mean()


class DenseVVCLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_config: Optional[DenseVVCConfig] = None,
        lr: float = 1e-4,
        weight_decay: Optional[float] = None,
        model_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model_config"])

        self.model_config = model_config or DenseVVCConfig()
        self.model = DenseVVCEnhancer(self.model_config)
        self.criterion = DenseVVCLoss()

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
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            threshold=1e-4,
            threshold_mode="rel",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


from registry import register_model

_DENSE_PRESETS = {
    "dense_base": dict(growth_rate=16, bn_size=2.0, stem_features=64, blocks=[
        BlockCfg(4, 7, 3), BlockCfg(6, 5, 2), BlockCfg(4, 3, 1),
        BlockCfg(3, 3, 1), BlockCfg(2, 3, 1),
    ]),
    "dense_medium": dict(growth_rate=24, bn_size=2.0, stem_features=80, blocks=[
        BlockCfg(5, 7, 3), BlockCfg(7, 5, 2), BlockCfg(5, 3, 1),
        BlockCfg(4, 3, 1), BlockCfg(3, 3, 1),
    ]),
    "dense_large": dict(growth_rate=32, bn_size=2.0, stem_features=96, blocks=[
        BlockCfg(6, 7, 3), BlockCfg(8, 5, 2), BlockCfg(6, 3, 1),
        BlockCfg(5, 3, 1), BlockCfg(4, 3, 1),
    ]),
}

for _name, _kw in _DENSE_PRESETS.items():
    register_model(
        _name,
        config_cls=DenseVVCConfig,
        model_cls=DenseVVCEnhancer,
        lightning_cls=DenseVVCLightningModule,
        **_kw,
    )


if __name__ == "__main__":
    cfg = DenseVVCConfig()
    model = DenseVVCEnhancer(cfg)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    x = torch.rand(2, 4, 132, 132)
    y = model(x)
    print("Output shape:", y.shape)

    criterion = DenseVVCLoss()
    target = torch.rand(2, 3, 132, 132)
    loss, parts = criterion(y, target)
    print(f"Loss: {loss.item():.6f}")
    for k, v in parts.items():
        print(f"  {k}: {v.item():.6f}")
