"""
VVC-PPFF GAN: Generator (existing PPFF) + PatchGAN Discriminator.

Stable training via:
  - Spectral normalization on discriminator
  - LSGAN loss (least-squares, no log → no vanishing gradients)
  - Pixel loss dominant (λ_pixel >> λ_adv)
  - Two-timescale update rule (D lr > G lr)
  - Optional generator warmup (pure MSE for N epochs first)
  - Gradient penalty (R1 regularization)

Usage:
    python train_gan.py --model ppff_base --ckpt-path <pretrained.ckpt> ...
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from model_v2 import compute_psnr, compute_y_psnr
from registry import create_model


def yuv_to_rgb(yuv: torch.Tensor) -> torch.Tensor:
    """Convert YCbCr [0,1] to RGB [0,1] (BT.709). Input: [B, 3, H, W]."""
    y = yuv[:, 0:1]
    u = yuv[:, 1:2] - 0.5
    v = yuv[:, 2:3] - 0.5
    r = y + 1.5748 * v
    g = y - 0.1873 * u - 0.4681 * v
    b = y + 1.8556 * u
    return torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features (conv1_2, conv2_2, conv3_4, conv4_4).
    Input: RGB [0,1] tensors.

    VGG19 pretrained on ImageNet (torchvision).
    Ref: Simonyan & Zisserman, "Very Deep Convolutional Networks for
    Large-Scale Image Recognition", ICLR 2015 (arXiv:1409.1556).
    """
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:4]),
            nn.Sequential(*list(vgg.children())[4:9]),
            nn.Sequential(*list(vgg.children())[9:18]),
            nn.Sequential(*list(vgg.children())[18:27]),
        ])
        for p in self.parameters():
            p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: RGB [0,1] [B, 3, H, W]."""
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = torch.tensor(0.0, device=pred.device)
        x, y = pred, target
        for s in self.slices:
            x = s(x)
            y = s(y)
            loss = loss + F.l1_loss(x, y)
        return loss


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator (70x70 receptive field).
    Input: 3ch YUV (original or enhanced).
    Output: NxN feature map (real/fake per patch).

    Spectral norm on all conv layers for Lipschitz constraint.
    """
    def __init__(self, in_channels: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        layers = []

        layers.append(spectral_norm(nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.extend([
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.extend([
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ])

        layers.append(spectral_norm(nn.Conv2d(ndf * nf_mult, 1, 4, stride=1, padding=1)))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PPFFDiscriminatorBlock(nn.Module):
    """FEB block for discriminator — LeakyReLU + optional spectral norm."""
    def __init__(self, channels: int = 128, use_spectral_norm: bool = True):
        super().__init__()
        conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        if use_spectral_norm:
            conv = spectral_norm(conv)
        self.conv = conv
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class PPFFDiscriminator(nn.Module):
    """
    Deep PatchGAN discriminator using PPFF architecture (progressive feature fusion).

    Same structure as the PPFF generator but:
    - No skip connection (classification, not residual learning)
    - LeakyReLU instead of PReLU
    - Optional spectral normalization on all convolutions
    - 1-channel output (patch-level real/fake logits)

    With 16 blocks × 128ch this has ~2.7M params — matches generator capacity.

    Conditional mode: set in_channels=6 and pass cat(decoded, enhanced/original).
    """
    def __init__(self, in_channels: int = 3, num_blocks: int = 16, channels: int = 128,
                 use_spectral_norm: bool = True):
        super().__init__()

        def maybe_sn(layer):
            return spectral_norm(layer) if use_spectral_norm else layer

        self.initial_conv = maybe_sn(nn.Conv2d(in_channels, channels, kernel_size=1, bias=True))
        self.initial_act = nn.LeakyReLU(0.2, inplace=True)

        self.febs = nn.ModuleList([
            PPFFDiscriminatorBlock(channels, use_spectral_norm) for _ in range(num_blocks)
        ])

        self.bottlenecks = nn.ModuleList([
            maybe_sn(nn.Conv2d(channels, channels, kernel_size=1, bias=True))
            for _ in range(num_blocks)
        ])

        self.fusion_smoothing_conv = maybe_sn(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        )

        self.head = maybe_sn(nn.Conv2d(channels, 1, kernel_size=1, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] — enhanced or original YUV patch
        Returns:
            [B, 1, H, W] — patch-level logits (before sigmoid)
        """
        out = self.initial_act(self.initial_conv(x))

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
        logits = self.head(out)
        return logits


def lsgan_loss_d(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
    """LSGAN discriminator loss: (D(real)-1)^2 + D(fake)^2"""
    return 0.5 * (F.mse_loss(real_pred, torch.ones_like(real_pred)) +
                  F.mse_loss(fake_pred, torch.zeros_like(fake_pred)))


def lsgan_loss_g(fake_pred: torch.Tensor) -> torch.Tensor:
    """LSGAN generator loss: (D(fake)-1)^2"""
    return F.mse_loss(fake_pred, torch.ones_like(fake_pred))


def ragan_loss_d(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
    """
    Relativistic average LSGAN discriminator loss (ESRGAN-style).
    D learns: real should look MORE real than average fake, and vice versa.
    Works even when real ≈ fake because it compares RELATIVE quality.
    """
    real_mean = real_pred.mean()
    fake_mean = fake_pred.mean()
    return 0.5 * (
        F.mse_loss(real_pred - fake_mean, torch.ones_like(real_pred)) +
        F.mse_loss(fake_pred - real_mean, -torch.ones_like(fake_pred))
    )


def ragan_loss_g(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
    """
    Relativistic average LSGAN generator loss (ESRGAN-style).
    G wants: fake should look MORE real than average real.
    Key: G loss uses both real and fake predictions (relativistic).
    """
    real_mean = real_pred.mean()
    fake_mean = fake_pred.mean()
    return 0.5 * (
        F.mse_loss(real_pred - fake_mean, -torch.ones_like(real_pred)) +
        F.mse_loss(fake_pred - real_mean, torch.ones_like(fake_pred))
    )


def r1_gradient_penalty(real_images: torch.Tensor, real_pred: torch.Tensor) -> torch.Tensor:
    """R1 gradient penalty for discriminator regularization."""
    grad, = torch.autograd.grad(
        outputs=real_pred.sum(),
        inputs=real_images,
        create_graph=True,
    )
    return 0.5 * grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()


@dataclass
class GANConfig:
    generator_name: str = "ppff_base"
    generator_in_channels: int = 4
    lambda_pixel: float = 1.0
    lambda_perceptual: float = 0.01
    lambda_adv: float = 0.01
    lambda_r1: float = 10.0
    lr_g: float = 1e-5
    lr_d: float = 2e-4
    warmup_epochs: int = 0
    d_pretrain_epochs: int = 0
    ndf: int = 64
    d_layers: int = 3
    d_type: str = "ppff"
    d_num_blocks: int = 16
    d_channels: int = 128
    d_conditional: bool = False
    d_spectral_norm: bool = True
    gan_loss_type: str = "ragan"
    r1_every: int = 16
    accumulate_grad_batches: int = 1


class PPFFGANModule(pl.LightningModule):
    def __init__(
        self,
        gan_config: Optional[GANConfig] = None,
        model_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        cfg = gan_config or GANConfig()
        self.cfg = cfg

        self.generator = create_model(cfg.generator_name, in_channels=cfg.generator_in_channels)

        d_in_ch = 6 if cfg.d_conditional else 3
        if cfg.d_type == "ppff":
            self.discriminator = PPFFDiscriminator(
                in_channels=d_in_ch, num_blocks=cfg.d_num_blocks,
                channels=cfg.d_channels, use_spectral_norm=cfg.d_spectral_norm
            )
        else:
            self.discriminator = PatchDiscriminator(
                in_channels=d_in_ch, ndf=cfg.ndf, n_layers=cfg.d_layers
            )

        self.pixel_criterion = nn.MSELoss()

        if cfg.lambda_perceptual > 0:
            self.vgg_loss = VGGPerceptualLoss()
        else:
            self.vgg_loss = None

        self.d_step_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def _d_input(self, condition: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Build discriminator input (conditional or unconditional)."""
        if self.cfg.d_conditional:
            return torch.cat([condition, image], dim=1)
        return image

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        opt_g, opt_d = self.optimizers()
        x = batch["input"]
        y = batch["target"]

        decoded_yuv = x[:, :3, :, :]

        accum = self.cfg.accumulate_grad_batches
        should_step = (batch_idx + 1) % accum == 0

        if self.current_epoch < self.cfg.d_pretrain_epochs:
            real_input = self._d_input(decoded_yuv, y)
            fake_input = self._d_input(decoded_yuv, decoded_yuv)

            real_pred = self.discriminator(real_input)
            fake_pred = self.discriminator(fake_input)

            loss_d = lsgan_loss_d(real_pred, fake_pred) / accum
            self.manual_backward(loss_d)
            if should_step:
                opt_d.step()
                opt_d.zero_grad()

            self.log("train/loss_d", loss_d * accum, prog_bar=True)
            self.log("train/d_pretrain", 1.0)
            return

        fake = self.generator(x)
        fake_yuv = torch.clamp(fake, 0.0, 1.0)

        if self.current_epoch < self.cfg.d_pretrain_epochs + self.cfg.warmup_epochs:
            loss_pixel = self.pixel_criterion(fake, y) / accum
            self.manual_backward(loss_pixel)
            if should_step:
                nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                opt_g.step()
                opt_g.zero_grad()
            self.log("train/loss_g", loss_pixel * accum, prog_bar=True)
            self.log("train/loss_pixel", loss_pixel * accum)
            return

        real_input = self._d_input(decoded_yuv, y)
        fake_input = self._d_input(decoded_yuv, fake_yuv.detach())

        real_input_r1 = real_input.detach().requires_grad_(True)
        real_pred = self.discriminator(real_input_r1)
        fake_pred = self.discriminator(fake_input)

        if self.cfg.gan_loss_type == "ragan":
            loss_d = ragan_loss_d(real_pred, fake_pred) / accum
        else:
            loss_d = lsgan_loss_d(real_pred, fake_pred) / accum

        loss_r1 = torch.tensor(0.0, device=self.device)
        self.d_step_count += 1
        if self.cfg.lambda_r1 > 0 and self.d_step_count % self.cfg.r1_every == 0:
            loss_r1 = r1_gradient_penalty(real_input_r1, real_pred) * self.cfg.lambda_r1 / accum

        self.manual_backward(loss_d + loss_r1)
        if should_step:
            opt_d.step()
            opt_d.zero_grad()

        fake_input_g = self._d_input(decoded_yuv, fake_yuv)
        fake_pred_g = self.discriminator(fake_input_g)

        if self.cfg.gan_loss_type == "ragan":
            with torch.no_grad():
                real_pred_for_g = self.discriminator(real_input)
            loss_adv = ragan_loss_g(real_pred_for_g, fake_pred_g) * self.cfg.lambda_adv / accum
        else:
            loss_adv = lsgan_loss_g(fake_pred_g) * self.cfg.lambda_adv / accum

        loss_pixel = self.pixel_criterion(fake, y) * self.cfg.lambda_pixel / accum

        loss_percep = torch.tensor(0.0, device=self.device)
        if self.vgg_loss is not None:
            fake_rgb = yuv_to_rgb(fake_yuv)
            real_rgb = yuv_to_rgb(y)
            loss_percep = self.vgg_loss(fake_rgb, real_rgb) * self.cfg.lambda_perceptual / accum

        loss_g = loss_pixel + loss_adv + loss_percep

        self.manual_backward(loss_g)
        if should_step:
            nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
            opt_g.step()
            opt_g.zero_grad()

        self.log("train/loss_g", loss_g * accum, prog_bar=True)
        self.log("train/loss_d", loss_d * accum, prog_bar=True)
        self.log("train/loss_pixel", loss_pixel * accum)
        self.log("train/loss_adv", loss_adv * accum)
        self.log("train/loss_percep", loss_percep * accum)
        self.log("train/loss_r1", loss_r1 * accum)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        x = batch["input"]
        y = batch["target"]
        decoded_yuv = x[:, :3, :, :]
        pred = self.generator(x)

        loss = self.pixel_criterion(pred, y)
        pred_clamped = torch.clamp(pred, 0.0, 1.0)
        psnr = compute_psnr(pred_clamped, y)
        y_psnr = compute_y_psnr(pred_clamped, y)

        mse_uv = F.mse_loss(pred_clamped[:, 1:3], y[:, 1:3])
        uv_psnr = -10.0 * torch.log10(mse_uv.clamp(min=1e-10))

        real_input = self._d_input(decoded_yuv, y)
        fake_input = self._d_input(decoded_yuv, pred_clamped)
        real_pred = self.discriminator(real_input)
        fake_pred = self.discriminator(fake_input)
        d_acc_real = (real_pred > 0.5).float().mean()
        d_acc_fake = (fake_pred < 0.5).float().mean()

        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/psnr", psnr, prog_bar=True, on_epoch=True)
        self.log("val/y_psnr", y_psnr, prog_bar=True, on_epoch=True)
        self.log("val/uv_psnr", uv_psnr, prog_bar=True, on_epoch=True)
        self.log("val/d_acc_real", d_acc_real, on_epoch=True)
        self.log("val/d_acc_fake", d_acc_fake, on_epoch=True)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.cfg.lr_g,
            betas=(0.5, 0.999),
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.cfg.lr_d,
            betas=(0.5, 0.999),
        )
        return [opt_g, opt_d]
