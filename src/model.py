"""
CT Denoising Models

- REDCNN: Chen et al. IEEE TMI 2017 — lightweight baseline
- DilatedREDCNN: REDCNN with dilated convolutions for larger receptive field
- UNetSmall: Compact 4-level U-Net
- NAFNet: Non-linear Activation Free Network (Chen et al. ECCV 2022)
  SOTA for image restoration — uses SimpleGate + LayerNorm2d + PixelShuffle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── REDCNN ─────────────────────────────────────────────────────────────────

class REDCNN(nn.Module):
    """
    Residual Encoder-Decoder CNN for low-dose CT denoising.
    Input/Output: (B, 1, H, W) single-channel CT patches.
    """

    def __init__(self, n_channels: int = 96):
        super().__init__()
        c = n_channels

        self.enc1 = nn.Conv2d(1, c, 5, padding=2)
        self.enc2 = nn.Conv2d(c, c, 5, padding=2)
        self.enc3 = nn.Conv2d(c, c, 5, padding=2)
        self.enc4 = nn.Conv2d(c, c, 5, padding=2)
        self.enc5 = nn.Conv2d(c, c, 5, padding=2)

        self.dec5 = nn.ConvTranspose2d(c, c, 5, padding=2)
        self.dec4 = nn.ConvTranspose2d(c, c, 5, padding=2)
        self.dec3 = nn.ConvTranspose2d(c, c, 5, padding=2)
        self.dec2 = nn.ConvTranspose2d(c, c, 5, padding=2)
        self.dec1 = nn.ConvTranspose2d(c, 1, 5, padding=2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        e3 = self.relu(self.enc3(e2))
        e4 = self.relu(self.enc4(e3))
        e5 = self.relu(self.enc5(e4))

        d5 = self.relu(self.dec5(e5) + e4)
        d4 = self.relu(self.dec4(d5) + e3)
        d3 = self.relu(self.dec3(d4) + e2)
        d2 = self.relu(self.dec2(d3) + e1)
        d1 = self.dec1(d2) + x
        return d1


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention on skip connections."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.gate(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * w


class REDCNN_SE(nn.Module):
    """
    REDCNN + SE blocks on skip connections.

    Research-backed upgrade (Deep Research Report Q2):
    - SE(ratio=4) on skip connections e4, e3, e2, e1 → +0.05–0.15 dB
    - ~24K extra params (negligible vs 3.3M base)
    - Compatible with plain REDCNN checkpoint (strict=False loading)
    """

    def __init__(self, n_channels: int = 128):
        super().__init__()
        c = n_channels

        self.enc1 = nn.Conv2d(1, c, 5, padding=2)
        self.enc2 = nn.Conv2d(c, c, 5, padding=2)
        self.enc3 = nn.Conv2d(c, c, 5, padding=2)
        self.enc4 = nn.Conv2d(c, c, 5, padding=2)
        self.enc5 = nn.Conv2d(c, c, 5, padding=2)

        self.dec5 = nn.ConvTranspose2d(c, c, 5, padding=2)
        self.dec4 = nn.ConvTranspose2d(c, c, 5, padding=2)
        self.dec3 = nn.ConvTranspose2d(c, c, 5, padding=2)
        self.dec2 = nn.ConvTranspose2d(c, c, 5, padding=2)
        self.dec1 = nn.ConvTranspose2d(c, 1, 5, padding=2)

        # SE gates on skip connections (new — not in base REDCNN)
        self.se4 = SEBlock(c, reduction=4)
        self.se3 = SEBlock(c, reduction=4)
        self.se2 = SEBlock(c, reduction=4)
        self.se1 = SEBlock(c, reduction=4)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        e3 = self.relu(self.enc3(e2))
        e4 = self.relu(self.enc4(e3))
        e5 = self.relu(self.enc5(e4))

        d5 = self.relu(self.dec5(e5) + self.se4(e4))
        d4 = self.relu(self.dec4(d5) + self.se3(e3))
        d3 = self.relu(self.dec3(d4) + self.se2(e2))
        d2 = self.relu(self.dec2(d3) + self.se1(e1))
        d1 = self.dec1(d2) + x
        return d1


class DilatedREDCNN(nn.Module):
    """
    REDCNN with dilated convolutions for a larger receptive field.
    Dilations [1, 2, 4, 2, 1] give effective receptive field of ~29px vs 13px.
    """

    def __init__(self, n_channels: int = 96):
        super().__init__()
        c = n_channels
        dilations = [1, 2, 4, 2, 1]

        self.enc = nn.ModuleList([
            nn.Conv2d(1 if i == 0 else c, c, 5,
                      padding=2 * dilations[i], dilation=dilations[i])
            for i in range(5)
        ])
        self.dec = nn.ModuleList([
            nn.ConvTranspose2d(c, c if i < 4 else 1, 5,
                               padding=2 * dilations[4 - i], dilation=dilations[4 - i])
            for i in range(5)
        ])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        enc_feats = []
        h = x
        for i, layer in enumerate(self.enc):
            h = self.relu(layer(h if i == 0 else enc_feats[-1]))
            enc_feats.append(h)

        d = enc_feats[-1]
        for i, layer in enumerate(self.dec):
            skip = enc_feats[-(i + 2)] if i < 4 else x
            if i < 4:
                d = self.relu(layer(d) + skip)
            else:
                d = layer(d) + skip
        return d


# ── UNetSmall ──────────────────────────────────────────────────────────────

class UNetSmall(nn.Module):
    """Compact U-Net for CT denoising. 4-level encoder-decoder with skip connections."""

    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()
        b = base_ch

        self.enc1 = self._double_conv(in_ch, b)
        self.enc2 = self._double_conv(b, b * 2)
        self.enc3 = self._double_conv(b * 2, b * 4)
        self.enc4 = self._double_conv(b * 4, b * 8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._double_conv(b * 8, b * 16)

        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.dec4 = self._double_conv(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = self._double_conv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = self._double_conv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = self._double_conv(b * 2, b)

        self.out_conv = nn.Conv2d(b, 1, 1)

    def _double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        bn = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1) + x


# ── NAFNet ─────────────────────────────────────────────────────────────────

class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps (BCHW)."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → permute → norm → permute back
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    """
    Gating nonlinearity: split channels in half, multiply element-wise.
    Replaces all ReLU/GELU — the only nonlinearity in NAFNet.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    Non-linear Activation Free Block from Chen et al. ECCV 2022.

    Structure (two sub-layers with learnable scalars β, γ):
      1) LN → 1×1 → DW3×3 → SimpleGate → SCA → 1×1  (+β*residual)
      2) LN → 1×1 → SimpleGate → 1×1                  (+γ*residual)
    """

    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_ch = c * dw_expand
        ffn_ch = c * ffn_expand

        # Sub-layer 1: spatial mixing
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_ch, 1)
        self.dw_conv = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch)
        self.sg1 = SimpleGate()
        # Simplified channel attention (no activation, just global pool + 1×1)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, 1, bias=True),
        )
        self.conv2 = nn.Conv2d(dw_ch // 2, c, 1)

        # Sub-layer 2: channel mixing (FFN)
        self.norm2 = LayerNorm2d(c)
        self.conv3 = nn.Conv2d(c, ffn_ch, 1)
        self.sg2 = SimpleGate()
        self.conv4 = nn.Conv2d(ffn_ch // 2, c, 1)

        # Learnable per-channel scalars, initialized to 0 (stable early training)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.dw_conv(x)
        x = self.sg1(x)
        x = x * self.sca(x)
        x = self.conv2(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv3(x)
        x = self.sg2(x)
        x = self.conv4(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    """
    Non-linear Activation Free Network for image restoration.
    Chen et al. "Simple Baselines for Image Restoration", ECCV 2022.

    Architecture:
    - PixelUnshuffle downsampling / PixelShuffle upsampling (lossless)
    - NAFBlocks at each encoder/decoder level
    - Skip connections between matching encoder and decoder levels
    - Global residual (input + output)

    Default config (width=64, enc_blks=[2,2,4,8], dec_blks=[2,2,2,2]):
      ~26M parameters — strong capacity for CT denoising.
    Lite config (width=32): ~7M parameters, faster training.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 64,
        middle_blk_num: int = 12,
        enc_blks: list = None,
        dec_blks: list = None,
    ):
        super().__init__()
        if enc_blks is None:
            enc_blks = [2, 2, 4, 8]
        if dec_blks is None:
            dec_blks = [2, 2, 2, 2]

        assert len(enc_blks) == len(dec_blks)
        n_levels = len(enc_blks)

        # Input / output projections
        self.intro = nn.Conv2d(in_channels, width, 3, padding=1)
        self.outro = nn.Conv2d(width, out_channels, 3, padding=1)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        for n in enc_blks:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))
            # PixelUnshuffle(2): (B,C,H,W) → (B,4C,H/2,W/2) then reduce channels
            self.downs.append(nn.Sequential(
                nn.PixelUnshuffle(2),
                nn.Conv2d(ch * 4, ch * 2, 1),
            ))
            ch *= 2

        # Middle blocks
        self.middle_blks = nn.Sequential(*[NAFBlock(ch) for _ in range(middle_blk_num)])

        # Decoder
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        for n in dec_blks:
            # PixelShuffle(2): expand channels then (B,4C,H/2,W/2) → (B,C,H,W)
            self.ups.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, 1),
                nn.PixelShuffle(2),
            ))
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.intro(inp)

        enc_feats = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            enc_feats.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, skip in zip(self.decoders, self.ups, reversed(enc_feats)):
            x = up(x)
            x = x + skip
            x = decoder(x)

        return self.outro(x) + inp  # global residual


def build_nafnet(lite: bool = False) -> NAFNet:
    """Convenience constructor. lite=True for faster training/debugging."""
    if lite:
        return NAFNet(width=32, middle_blk_num=4, enc_blks=[1, 1, 2, 4], dec_blks=[1, 1, 1, 1])
    return NAFNet(width=64, middle_blk_num=12, enc_blks=[2, 2, 4, 8], dec_blks=[2, 2, 2, 2])
