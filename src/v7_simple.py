#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  v7_simple — 2-term loss (Charb+Grad) + WD=1e-4 fix            ║
║                                                                      ║
║  Diagnosis: v5 hit 33.68 dB with 10.4M params.                     ║
║    - FFL didn't help → residual gap is NOT frequency-domain.        ║
║    - TTA gave only +0.02 dB → NOT variance reduction issue.         ║
║    - Loss changes saturated → problem is ARCHITECTURAL capacity.    ║
║                                                                      ║
║  Solution: two changes, both publishable:                            ║
║                                                                      ║
║  1. MHDC Bottleneck (Multi-Head Dilated Convolution)                ║
║     Replaces 2× ConvBlockV4 in bottleneck with 2× MHDCBlock.       ║
║     4 parallel heads with dilation=[1,2,3]+global pooling.          ║
║     Same param count — redistributes weights across scales.         ║
║     Gives the bottleneck simultaneous local-to-global RF.           ║
║     Paper contribution: "MSPMnet-inspired multi-scale processing    ║
║     at negligible parameter cost."                                   ║
║                                                                      ║
║  2. base_ch=26 (was 24) → ~12M params (matches MSPMnet budget)     ║
║     Fair comparison: same param tier as the SOTA baseline.          ║
║                                                                      ║
║  Everything else: IDENTICAL to v5 (proven loss recipe).             ║
║  Target: ≥33.85 dB  |  Runtime: ~16h on 1080 Ti / TITAN Xp        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, time, math, glob, json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

cudnn.benchmark = True
cudnn.deterministic = False

try:
    from pytorch_msssim import ms_ssim, ssim as _ssim_fn
    _HAS_MSSSIM = True
except ImportError:
    _HAS_MSSSIM = False

from skimage.metrics import structural_similarity

print("All imports OK")
print(f"PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)")


# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

TRAIN_ROOT = os.environ.get('MAYO_DATA_ROOT', './data/Traning_Image_Data')
KERNEL     = '3mm B30'
SAVE_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
LOG_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')

# ── THE TWO CHANGES ──
BASE_CH      = 26                # v7: was 24 → ~12M params (matches MSPMnet)
USE_MHDC     = True              # v7: MHDC bottleneck for multi-scale RF

# ── Device ──
DEVICE       = 'cuda:1'

# ── Hyperparams — IDENTICAL to v5 (proven) ──
BASE_LR      = 2e-5              # FT: low LR
MIN_LR       = 1e-7
EPOCHS       = 150              # FT from E47
WARMUP       = 10
WEIGHT_DECAY = 1e-4             # FIX: paper-faithful
GRAD_CLIP    = 0.5
EMA_DECAY    = 0.999
PATIENCE     = 50

# ── Progressive schedule — adjusted for ~12M model in FP32 11.7GB ──
PATCH_SCHEDULE = {
    1:   (256, 4, 6),            # eff=24
}

# ── Loss — IDENTICAL to v5 ──
GRAD_LOSS_LAMBDA  = 0.15          # Slightly higher — edges are remaining error
# MS-SSIM and SWT REMOVED — cleaner PSNR gradient signal

# ── Validation ──
VAL_FRACTION = 0.15
VAL_PATCH    = 256
VAL_OVERLAP  = 128
VAL_TTA      = False

# ── Muon ──
MUON_LR       = 5e-4            # FT: lower Muon
MUON_MOMENTUM = 0.95

# ── Benchmark ──
MSPNET_PSNR  = 33.85
MSPNET_STD   = 1.82


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def load_ima(path: str) -> np.ndarray:
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    return arr * slope + intercept

def window_normalize(hu: np.ndarray, wl: float = 40.0, ww: float = 400.0) -> np.ndarray:
    low, high = wl - ww / 2.0, wl + ww / 2.0
    return np.clip((hu - low) / (high - low), 0.0, 1.0).astype(np.float32)

def collect_paired_paths(root: str, kernel: str = '3mm B30') -> List[Tuple[str, str]]:
    base = Path(root) / kernel
    if '3mm B30' in kernel:     fd_dir, qd_dir = 'full_3mm', 'quarter_3mm'
    elif '3mm D45' in kernel:   fd_dir, qd_dir = 'full_3mm_sharp', 'quarter_3mm_sharp'
    elif '1mm B30' in kernel:   fd_dir, qd_dir = 'full_1mm', 'quarter_1mm'
    elif '1mm D45' in kernel:   fd_dir, qd_dir = 'full_1mm_sharp', 'quarter_1mm_sharp'
    else: raise ValueError(f"Unknown kernel: {kernel}")

    fd_root, qd_root = base / fd_dir, base / qd_dir
    assert fd_root.exists(), f"Not found: {fd_root}"
    assert qd_root.exists(), f"Not found: {qd_root}"

    pairs = []
    for patient_dir in sorted(fd_root.iterdir()):
        if not patient_dir.is_dir(): continue
        pid = patient_dir.name
        qd_patient = qd_root / pid
        if not qd_patient.exists():
            print(f"  WARNING: No QD match for {pid}, skip"); continue
        fd_imas = sorted(patient_dir.rglob("*.IMA"))
        qd_imas = sorted(qd_patient.rglob("*.IMA"))
        if len(fd_imas) != len(qd_imas):
            n = min(len(fd_imas), len(qd_imas))
            print(f"  WARNING: {pid} mismatch fd={len(fd_imas)} qd={len(qd_imas)}, using {n}")
            fd_imas, qd_imas = fd_imas[:n], qd_imas[:n]
        for qd_p, fd_p in zip(qd_imas, fd_imas):
            pairs.append((str(qd_p), str(fd_p)))
    return pairs

def get_patient_id(path: str) -> str:
    for p in Path(path).parts:
        if p.startswith('L') and len(p) == 4 and p[1:].isdigit(): return p
    return Path(path).parent.parent.name

def patient_split(pairs, val_fraction=0.15):
    patient_pairs = {}
    for qd, fd in pairs:
        pid = get_patient_id(fd)
        patient_pairs.setdefault(pid, []).append((qd, fd))
    patients = sorted(patient_pairs.keys())
    n_val = max(1, int(len(patients) * val_fraction))
    val_patients, train_patients = patients[-n_val:], patients[:-n_val]
    train_pairs = [p for pid in train_patients for p in patient_pairs[pid]]
    val_pairs = [p for pid in val_patients for p in patient_pairs[pid]]
    print(f"  Train patients ({len(train_patients)}): {train_patients}")
    print(f"  Val patients   ({len(val_patients)}): {val_patients}")
    print(f"  Train slices: {len(train_pairs)}, Val slices: {len(val_pairs)}")
    return train_pairs, val_pairs

def preload_pairs(pairs, wl=40.0, ww=400.0, label=""):
    n = len(pairs)
    if n == 0:
        return np.empty((0, 512, 512), dtype=np.float32), np.empty((0, 512, 512), dtype=np.float32)
    sample = load_ima(pairs[0][0])
    h, w = sample.shape
    qd_all = np.empty((n, h, w), dtype=np.float32)
    fd_all = np.empty((n, h, w), dtype=np.float32)
    for i, (qd_path, fd_path) in enumerate(pairs):
        if i % 200 == 0: print(f"  [{label}] Loading {i}/{n}...", flush=True)
        qd_all[i] = window_normalize(load_ima(qd_path), wl, ww)
        fd_all[i] = window_normalize(load_ima(fd_path), wl, ww)
    print(f"  [{label}] Done: {n} slices ({qd_all.nbytes/1e9:.1f}+{fd_all.nbytes/1e9:.1f} GB)")
    return qd_all, fd_all


# ══════════════════════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════════════════════

def _augment_pair(qp, fp):
    if np.random.random() > 0.5:
        qp, fp = np.flip(qp, 1).copy(), np.flip(fp, 1).copy()
    if np.random.random() > 0.5:
        qp, fp = np.flip(qp, 0).copy(), np.flip(fp, 0).copy()
    k = np.random.randint(0, 4)
    if k:
        qp, fp = np.rot90(qp, k).copy(), np.rot90(fp, k).copy()
    if np.random.random() > 0.5:
        shift = np.random.uniform(-0.05, 0.05)
        qp = np.clip(qp + shift, 0.0, 1.0)
    return qp, fp

class CachedCTDataset(Dataset):
    def __init__(self, qd, fd, patch_size=64, patches_per_slice=8, augment=True):
        self.qd, self.fd, self.ps, self.pps, self.augment = qd, fd, patch_size, patches_per_slice, augment
    def set_patch_size(self, ps): self.ps = ps
    def __len__(self): return len(self.qd) * self.pps
    def __getitem__(self, idx):
        s = idx // self.pps
        qd, fd = self.qd[s], self.fd[s]
        h, w = qd.shape
        y, x = np.random.randint(0, max(1, h-self.ps)), np.random.randint(0, max(1, w-self.ps))
        qp, fp = qd[y:y+self.ps, x:x+self.ps].copy(), fd[y:y+self.ps, x:x+self.ps].copy()
        if self.augment: qp, fp = _augment_pair(qp, fp)
        return torch.from_numpy(qp[None]), torch.from_numpy(fp[None])

class CachedFullSliceDataset(Dataset):
    def __init__(self, qd, fd): self.qd, self.fd = qd, fd
    def __len__(self): return len(self.qd)
    def __getitem__(self, idx):
        return torch.from_numpy(self.qd[idx][None].copy()), torch.from_numpy(self.fd[idx][None].copy())


# ══════════════════════════════════════════════════════════════════════
# MODEL — v7: MHDC Bottleneck + base_ch=26
# ══════════════════════════════════════════════════════════════════════

def _gn(channels, groups=8):
    """GroupNorm with adaptive group count for non-standard channel widths."""
    if channels % groups == 0: return nn.GroupNorm(groups, channels)
    if channels % 4 == 0:      return nn.GroupNorm(4, channels)
    if channels % 2 == 0:      return nn.GroupNorm(2, channels)
    return nn.GroupNorm(1, channels)

def _icnr_init(tensor, scale=2):
    out_c, in_c, h, w = tensor.shape
    sub_c = out_c // (scale ** 2)
    kernel = torch.empty(sub_c, in_c, h, w)
    nn.init.kaiming_normal_(kernel)
    tensor.data.copy_(kernel.repeat_interleave(scale ** 2, dim=0))


class ConvBlockV4(nn.Module):
    """Standard 2× (Conv3×3 + GN + GELU) + residual."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), _gn(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), _gn(out_ch), nn.GELU())
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
    def forward(self, x): return self.conv(x) + self.skip(x)


class LargeKernelConvBlock(nn.Module):
    """Decomposed large-kernel depthwise for encoder levels 3–4."""
    def __init__(self, in_ch, out_ch, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), _gn(out_ch), nn.GELU())
        self.lk_branch = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, (1, kernel_size), padding=(0, pad), groups=out_ch, bias=False),
            nn.Conv2d(out_ch, out_ch, (kernel_size, 1), padding=(pad, 0), groups=out_ch, bias=False),
            _gn(out_ch), nn.GELU())
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), _gn(out_ch), nn.GELU())
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        h = self.local_conv(x)
        h = h + self.lk_branch(h)
        return self.out_conv(h) + self.skip(x)


# ═══════════════════════════════════════════════
# v7 NEW: MHDC Block — Multi-Head Dilated Conv
# ═══════════════════════════════════════════════

class MHDCBlock(nn.Module):
    """
    Multi-Head Dilated Convolution Block — MSPMnet-inspired.

    Splits channels into 4 parallel heads, each with different receptive field:
      Head 1: 3×3 dilation=1  → local context     (3×3 effective RF)
      Head 2: 3×3 dilation=2  → medium context     (5×5 effective RF)
      Head 3: 3×3 dilation=3  → long-range context  (7×7 effective RF)
      Head 4: global avg pool  → global context     (full-image RF, 0 params)

    Followed by 1×1 fusion + 3×3 refinement + residual.

    Key properties:
      - Same param count as ConvBlockV4 (redistributes, doesn't add)
      - Multi-scale RF coexists within single block
      - Dilation = linear cost, not quadratic
      - Pool branch = zero parameters, pure global context

    This is the paper's architectural contribution.
    """
    def __init__(self, in_ch, out_ch, dilations=(1, 2, 3)):
        super().__init__()
        assert out_ch % 4 == 0, f"out_ch={out_ch} must be divisible by 4 for 4-head split"
        self.head_ch = out_ch // 4

        # Head 1: local (d=1)
        self.head_local = nn.Sequential(
            nn.Conv2d(in_ch, self.head_ch, 3, padding=dilations[0],
                      dilation=dilations[0], bias=False),
            _gn(self.head_ch), nn.GELU())

        # Head 2: medium (d=2, effective 5×5)
        self.head_medium = nn.Sequential(
            nn.Conv2d(in_ch, self.head_ch, 3, padding=dilations[1],
                      dilation=dilations[1], bias=False),
            _gn(self.head_ch), nn.GELU())

        # Head 3: long-range (d=3, effective 7×7)
        self.head_long = nn.Sequential(
            nn.Conv2d(in_ch, self.head_ch, 3, padding=dilations[2],
                      dilation=dilations[2], bias=False),
            _gn(self.head_ch), nn.GELU())

        # Head 4: global context (adaptive pool → 1×1 conv → upsample)
        # Zero conv params for spatial — only channel projection
        self.head_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, self.head_ch, 1, bias=False),
            _gn(self.head_ch), nn.GELU())

        # Fusion: 1×1 to mix heads + 3×3 refinement
        self.fusion = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            _gn(out_ch), nn.GELU())

        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch), nn.GELU())

        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        h1 = self.head_local(x)                           # [B, head_ch, H, W]
        h2 = self.head_medium(x)                           # [B, head_ch, H, W]
        h3 = self.head_long(x)                             # [B, head_ch, H, W]
        h4 = self.head_global(x)                           # [B, head_ch, 1, 1]
        h4 = h4.expand(-1, -1, H, W)                      # broadcast to spatial

        h = torch.cat([h1, h2, h3, h4], dim=1)            # [B, out_ch, H, W]
        h = self.fusion(h)
        h = self.refine(h)

        return h + self.skip(x)


# ═══════════════════════════════════════════════
# Standard building blocks (unchanged)
# ═══════════════════════════════════════════════

class SESkipGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, mid, bias=False), nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False), nn.Sigmoid())
    def forward(self, skip):
        return skip * self.gate(skip).view(skip.shape[0], skip.shape[1], 1, 1)


class DownV4(nn.Module):
    def __init__(self, in_ch, out_ch, use_lk=False, lk_size=7):
        super().__init__()
        self.down = nn.Conv2d(in_ch, in_ch, 2, stride=2, groups=in_ch, bias=False)
        self.conv = LargeKernelConvBlock(in_ch, out_ch, lk_size) if use_lk else ConvBlockV4(in_ch, out_ch)
    def forward(self, x): return self.conv(self.down(x))


class UpV4(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up_conv = nn.Conv2d(in_ch, in_ch * 2, 3, padding=1, bias=False)
        _icnr_init(self.up_conv.weight, scale=2)
        self.shuffle = nn.PixelShuffle(2)
        self.conv = ConvBlockV4(in_ch // 2 + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.shuffle(self.up_conv(x))
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


# ═══════════════════════════════════════════════
# FULL MODEL
# ═══════════════════════════════════════════════

class AttentionUNet_Pro_v7(nn.Module):
    """
    v7: base_ch=26 (~12M params) + MHDC bottleneck.

    Encoder/decoder: unchanged from v4.
    Bottleneck: 2× MHDCBlock with dilation=[1,2,3]+global pool.

    The bottleneck operates at 16× downsampled resolution (32×32 for 512 input).
    At that scale, dilation=3 gives 7×7 effective RF = 112px in original space.
    Combined with the global pool head, the bottleneck now has full-image context
    while maintaining local precision — exactly the MSPMnet principle.
    """
    def __init__(self, in_channels=1, base_ch=26):
        super().__init__()
        c1 = base_ch          # 26
        c2 = base_ch * 2      # 52
        c3 = base_ch * 4      # 104
        c4 = base_ch * 8      # 208
        c5 = base_ch * 16     # 416

        # Encoder (unchanged)
        self.enc1 = ConvBlockV4(in_channels, c1)
        self.enc2 = DownV4(c1, c2, use_lk=False)
        self.enc3 = DownV4(c2, c3, use_lk=True, lk_size=7)
        self.enc4 = DownV4(c3, c4, use_lk=True, lk_size=9)

        # ── v7: MHDC Bottleneck ──
        # Two stacked MHDCBlocks with dilation ramp
        # Block 1: c4→c5 (channel expansion + multi-scale)
        # Block 2: c5→c5 (refinement at full width)
        self.bottleneck = nn.Sequential(
            MHDCBlock(c4, c5, dilations=(1, 2, 3)),
            MHDCBlock(c5, c5, dilations=(1, 2, 4)),  # d=4 in block 2 → 9×9 effective
        )

        # SE skip gates (unchanged)
        self.se4 = SESkipGate(c4, 16)
        self.se3 = SESkipGate(c3, 16)
        self.se2 = SESkipGate(c2, 16)
        self.se1 = SESkipGate(c1, 16)

        # Decoder (unchanged)
        self.up4 = UpV4(c5, c4, c4)
        self.up3 = UpV4(c4, c3, c3)
        self.up2 = UpV4(c3, c2, c2)
        self.up1 = UpV4(c2, c1, c1)

        self.final_conv = nn.Conv2d(c1, in_channels, 3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bn = self.bottleneck(e4)
        d4 = self.up4(bn, self.se4(e4))
        d3 = self.up3(d4, self.se3(e3))
        d2 = self.up2(d3, self.se2(e2))
        d1 = self.up1(d2, self.se1(e1))
        return self.final_conv(d1) + x


# ══════════════════════════════════════════════════════════════════════
# LOSSES — IDENTICAL TO v5
# ══════════════════════════════════════════════════════════════════════

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2
    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred.float() - target.float()) ** 2 + self.eps2))

def gradient_loss(pred, target):
    def _grad(x):
        return x[:,:,:,1:] - x[:,:,:,:-1], x[:,:,1:,:] - x[:,:,:-1,:]
    pdx, pdy = _grad(pred)
    tdx, tdy = _grad(target)
    return F.l1_loss(pdx, tdx) + F.l1_loss(pdy, tdy)

class HaarSWT2D(nn.Module):
    def __init__(self):
        super().__init__()
        lo = torch.tensor([1.0, 1.0]) / (2**0.5)
        hi = torch.tensor([-1.0, 1.0]) / (2**0.5)
        self.register_buffer('ll', (lo.unsqueeze(1) @ lo.unsqueeze(0)).unsqueeze(0).unsqueeze(0))
        self.register_buffer('lh', (hi.unsqueeze(1) @ lo.unsqueeze(0)).unsqueeze(0).unsqueeze(0))
        self.register_buffer('hl', (lo.unsqueeze(1) @ hi.unsqueeze(0)).unsqueeze(0).unsqueeze(0))
        self.register_buffer('hh', (hi.unsqueeze(1) @ hi.unsqueeze(0)).unsqueeze(0).unsqueeze(0))
    def forward(self, x):
        C = x.shape[1]
        ll, lh, hl, hh = [f.to(x).expand(C,-1,-1,-1) for f in (self.ll, self.lh, self.hl, self.hh)]
        xp = F.pad(x, (0,1,0,1), mode='reflect')
        return tuple(F.conv2d(xp, f, groups=C) for f in (ll, lh, hl, hh))

class SWTLossCorrected(nn.Module):
    def __init__(self, ll_w=0.05, lh_w=0.01, hl_w=0.01, hh_w=0.05):
        super().__init__()
        self.swt = HaarSWT2D()
        self.w = (ll_w, lh_w, hl_w, hh_w)
    def forward(self, pred, target):
        ps, ts = self.swt(pred), self.swt(target)
        return sum(w * F.l1_loss(p, t) for w, p, t in zip(self.w, ps, ts))

class SimpleLoss(nn.Module):
    """2-term only: Charbonnier + Gradient. Clean PSNR signal."""
    def __init__(self, grad_lambda=0.15):
        super().__init__()
        self.charbonnier = CharbonnierLoss(eps=1e-3)
        self.grad_lambda = grad_lambda
    def set_epoch(self, e): pass
    def forward(self, pred, target):
        return (self.charbonnier(pred, target)
                + self.grad_lambda * gradient_loss(pred, target))


# ══════════════════════════════════════════════════════════════════════
# EMA
# ══════════════════════════════════════════════════════════════════════

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1-self.decay)
    def apply(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.clone(); p.data.copy_(self.shadow[n])
    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup: p.data.copy_(self.backup[n])
        self.backup = {}


# ══════════════════════════════════════════════════════════════════════
# MUON + HYBRID OPTIMIZER
# ══════════════════════════════════════════════════════════════════════

class MuonOptimizer:
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 weight_decay=0.0, ns_steps=5,
                 ns_coeffs=(3.4445, -4.7750, 2.0315), eps=1e-7):
        self.param_list = list(params)
        self.lr, self.momentum, self.nesterov = lr, momentum, nesterov
        self.weight_decay, self.ns_steps, self.eps = weight_decay, ns_steps, eps
        self.a, self.b, self.c = ns_coeffs
        self.state = {}
        self.param_groups = [{'lr': lr, 'params': self.param_list}]

    @torch.no_grad()
    def step(self):
        lr = self.param_groups[0]['lr']
        for p in self.param_list:
            if p.grad is None: continue
            g = p.grad
            if id(p) not in self.state:
                self.state[id(p)] = {'buf': torch.zeros_like(g)}
            buf = self.state[id(p)]['buf']
            buf.mul_(self.momentum).add_(g)
            update = (g + self.momentum * buf) if self.nesterov else buf
            shape = update.shape
            mat = update.reshape(shape[0], -1)
            X = mat / (mat.norm() + self.eps)
            a, b, c = self.a, self.b, self.c
            for _ in range(self.ns_steps):
                A = X @ X.T
                X = a * X + b * (A @ X) + c * (A @ (A @ X))
            scale = max(1.0, (mat.shape[0] / mat.shape[1]) ** 0.5)
            if self.weight_decay > 0:
                p.data.mul_(1.0 - lr * self.weight_decay)
            p.data.add_(X.reshape(shape), alpha=-lr * scale)

    def zero_grad(self, set_to_none=True):
        for p in self.param_list:
            if p.grad is not None:
                p.grad = None if set_to_none else p.grad.zero_()


class HybridOptimizer:
    def __init__(self, model, muon_lr=0.02, adamw_lr=1e-4,
                 weight_decay=5e-3, momentum=0.95):
        muon_params, adamw_params = [], []
        for _, p in model.named_parameters():
            if not p.requires_grad: continue
            (muon_params if p.dim() >= 2 else adamw_params).append(p)
        self.muon = MuonOptimizer(muon_params, lr=muon_lr, momentum=momentum,
                                   weight_decay=weight_decay)
        self.adamw = torch.optim.AdamW(adamw_params, lr=adamw_lr, weight_decay=weight_decay)
        self.param_groups = self.muon.param_groups + self.adamw.param_groups
        print(f"  Muon: {len(muon_params)} tensors ({sum(p.numel() for p in muon_params):,} params)")
        print(f"  AdamW: {len(adamw_params)} tensors ({sum(p.numel() for p in adamw_params):,} params)")
    def step(self): self.muon.step(); self.adamw.step()
    def zero_grad(self, set_to_none=True): self.muon.zero_grad(set_to_none); self.adamw.zero_grad(set_to_none)
    def set_lr(self, lr_muon, lr_adamw):
        self.muon.param_groups[0]['lr'] = lr_muon
        for pg in self.adamw.param_groups: pg['lr'] = lr_adamw


# ══════════════════════════════════════════════════════════════════════
# INFERENCE & METRICS
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_overlap_tile(model, img, patch_size=256, overlap=64, device='cuda'):
    if img.dim() == 2: img = img.unsqueeze(0)
    C, H, W = img.shape
    stride = patch_size - overlap
    output, count = torch.zeros(C, H, W), torch.zeros(C, H, W)
    ys = list(range(0, max(H-patch_size, 0)+1, stride))
    xs = list(range(0, max(W-patch_size, 0)+1, stride))
    if not ys or ys[-1]+patch_size < H: ys.append(max(0, H-patch_size))
    if not xs or xs[-1]+patch_size < W: xs.append(max(0, W-patch_size))
    model.eval()
    for y in ys:
        for x in xs:
            y2, x2 = min(y+patch_size, H), min(x+patch_size, W)
            patch = img[:, y:y2, x:x2]
            ph, pw = patch_size-patch.shape[1], patch_size-patch.shape[2]
            if ph > 0 or pw > 0: patch = F.pad(patch, (0, pw, 0, ph))
            pred = model(patch.unsqueeze(0).to(device, memory_format=torch.channels_last)).float().squeeze(0).cpu()
            output[:, y:y2, x:x2] += pred[:, :y2-y, :x2-x]
            count[:, y:y2, x:x2] += 1
    return output / count.clamp(min=1)

@torch.no_grad()
def predict_tta(model, img, patch_size=256, overlap=128, device='cuda'):
    if img.dim() == 2: img = img.unsqueeze(0)
    preds = []
    for k in range(4):
        for flip in (False, True):
            aug = img.clone()
            if flip: aug = torch.flip(aug, [2])
            aug = torch.rot90(aug, k, [1, 2])
            p = predict_overlap_tile(model, aug, patch_size, overlap, device)
            p = torch.rot90(p, -k, [1, 2])
            if flip: p = torch.flip(p, [2])
            preds.append(p)
    return torch.stack(preds).mean(0)

def compute_metrics(pred, target, data_range=1.0):
    mse = np.mean((pred - target) ** 2)
    psnr = 10 * np.log10(data_range**2 / mse) if mse > 1e-10 else float('inf')
    ssim_val = structural_similarity(pred, target, data_range=data_range,
                                      win_size=7, gaussian_weights=True,
                                      sigma=1.5, channel_axis=None,
                                      use_sample_covariance=False)
    return psnr, ssim_val

def validate(model, val_loader, device, tta=False):
    model.eval()
    psnrs, ssims, psnrs_in = [], [], []
    infer = predict_tta if tta else predict_overlap_tile
    with torch.no_grad():
        for qd, fd in val_loader:
            pred = infer(model, qd.squeeze(0), patch_size=VAL_PATCH, overlap=VAL_OVERLAP, device=device)
            pred_np = pred.squeeze(0).numpy().clip(0, 1)
            fd_np, qd_np = fd.squeeze(0).squeeze(0).numpy(), qd.squeeze(0).squeeze(0).numpy()
            p, s = compute_metrics(pred_np, fd_np)
            pi, _ = compute_metrics(qd_np, fd_np)
            psnrs.append(p); ssims.append(s); psnrs_in.append(pi)
    return (float(np.mean(psnrs)), float(np.mean(ssims)),
            float(np.mean(psnrs_in)), float(np.std(psnrs)), float(np.std(ssims)))


# ══════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def train():
    device = torch.device(DEVICE)

    print("\n" + "=" * 75)
    print("  AttentionUNet_Pro v7 — MHDC Bottleneck + base_ch=26")
    print("  ~12M params | Multi-scale RF | target: ≥33.85 dB")
    print("=" * 75)

    # ── Data ──
    print("\n[1/4] Collecting paired paths...")
    all_pairs = collect_paired_paths(TRAIN_ROOT, KERNEL)
    print(f"  Found {len(all_pairs)} paired slices")

    print("\n[2/4] Patient-level train/val split...")
    train_pairs, val_pairs = patient_split(all_pairs, VAL_FRACTION)

    print("\n[3/4] Preloading into RAM...")
    tqd, tfd = preload_pairs(train_pairs, label="Train")
    vqd, vfd = preload_pairs(val_pairs, label="Val")
    val_ds = CachedFullSliceDataset(vqd, vfd)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # ── Model ──
    print("\n[4/4] Building model...")
    mdl = AttentionUNet_Pro_v7(in_channels=1, base_ch=BASE_CH).to(device)
    mdl = mdl.to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}")
    assert n_params < 14_000_000, f"Param budget exceeded! {n_params:,} ≥ 14M"
    print(f"  ✓ Under 14M budget ({n_params/1e6:.2f}M < 14.0M)")

    # ── Optimizer ──
    opt = HybridOptimizer(mdl, muon_lr=MUON_LR, adamw_lr=BASE_LR,
                          weight_decay=WEIGHT_DECAY, momentum=MUON_MOMENTUM)
    criterion = SimpleLoss(grad_lambda=GRAD_LOSS_LAMBDA)
    ema = EMA(mdl, decay=EMA_DECAY)

    # ── State ──
    # ── Resume from best_v7.pt ──
    resume_path = os.path.join(SAVE_DIR, 'best_v7.pt')
    if os.path.exists(resume_path):
        print(f"  Resuming from {resume_path} (2-term loss + WD fix)...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        mdl.load_state_dict(ckpt['model_state_dict'])
        best_psnr = float(ckpt.get('psnr', 0.0))
        start_epoch = int(ckpt.get('epoch', 48)) + 1
        print(f"  Resumed: E{start_epoch-1}, best PSNR={best_psnr:.2f}")
        ema = EMA(mdl, decay=EMA_DECAY)  # re-init EMA from resumed weights
    else:
        start_epoch = 1
        best_psnr = 0.0
    no_improve = 0
    current_ps, current_bs, current_ga = 256, 4, 6

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # ── Config summary ──
    print(f"\n  ┌─ v7 Config ────────────────────────────────────────────")
    print(f"  │ Architecture: AttentionUNet_Pro_v7 (MHDC bottleneck)")
    print(f"  │ base_ch={BASE_CH}, params={n_params:,}")
    print(f"  │ Bottleneck: 2× MHDCBlock, dilations=[1,2,3]+[1,2,4], +global pool")
    print(f"  │ Epochs: {EPOCHS}  LR: AdamW {BASE_LR}→{MIN_LR}  Muon {MUON_LR}")
    print(f"  │ Loss: Charbonnier + Grad({GRAD_LOSS_LAMBDA}) [2-term only]")
    print(f"  │ Progressive: {PATCH_SCHEDULE}")
    print(f"  │ EMA: {EMA_DECAY}  WD: {WEIGHT_DECAY}  Clip: {GRAD_CLIP}")
    print(f"  └───────────────────────────────────────────────────────")

    # ── Log ──
    log_path = os.path.join(LOG_DIR, 'v7s_train_log.csv')
    with open(log_path, 'w') as f:
        f.write("epoch,loss,psnr,psnr_std,ssim,ssim_std,psnr_in,lr_muon,lr_adamw\n")

    for epoch in range(start_epoch, EPOCHS + 1):
        # ── Progressive ──
        if epoch in PATCH_SCHEDULE:
            current_ps, current_bs, current_ga = PATCH_SCHEDULE[epoch]
            print(f"\n  → Progressive: ps={current_ps}, bs={current_bs}, "
                  f"ga={current_ga} (eff={current_bs*current_ga})")

        train_ds = CachedCTDataset(tqd, tfd, patch_size=current_ps,
                                    patches_per_slice=8, augment=True)
        train_loader = DataLoader(train_ds, batch_size=current_bs, shuffle=True,
                                   num_workers=4, pin_memory=True, drop_last=True,
                                   persistent_workers=True)

        # ── LR schedule (warmup + cosine) ──
        if epoch <= WARMUP:
            frac = epoch / WARMUP
            adamw_lr = MIN_LR + (BASE_LR - MIN_LR) * frac
            muon_lr = 1e-5 + (MUON_LR - 1e-5) * frac
        else:
            t = epoch - WARMUP
            T = EPOCHS - WARMUP
            adamw_lr = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1 + math.cos(math.pi * t / T))
            muon_lr = 1e-5 + 0.5 * (MUON_LR - 1e-5) * (1 + math.cos(math.pi * t / T))
        opt.set_lr(muon_lr, adamw_lr)

        # ── Train ──
        criterion.set_epoch(epoch)
        mdl.train()
        t0 = time.time()
        tot_loss, n_batches = 0, 0
        opt.zero_grad(set_to_none=True)

        for step, (qd, fd) in enumerate(train_loader):
            qd = qd.to(device, memory_format=torch.channels_last)
            fd = fd.to(device, memory_format=torch.channels_last)
            pred = mdl(qd)
            loss = criterion(pred, fd) / current_ga

            if not torch.isfinite(loss):
                print(f"  ⚠ NaN/Inf at step {step}, skip")
                opt.zero_grad(set_to_none=True); continue

            loss.backward()

            if (step + 1) % current_ga == 0:
                gn = torch.nn.utils.clip_grad_norm_(mdl.parameters(), GRAD_CLIP)
                if torch.isfinite(gn): opt.step()
                else: print(f"  ⚠ Grad explosion ({gn:.3f}), skip")
                opt.zero_grad(set_to_none=True)
                ema.update(mdl)

            tot_loss += loss.item() * current_ga
            n_batches += 1

        # ── Validate (EMA) ──
        ema.apply(mdl)
        val_psnr, val_ssim, val_psnr_in, val_psnr_std, val_ssim_std = validate(
            mdl, val_loader, device, tta=VAL_TTA)
        ema.restore(mdl)

        elapsed = time.time() - t0
        delta = val_psnr - MSPNET_PSNR
        swt_tag = ""

        print(f"E{epoch:3d} | Loss: {tot_loss/max(n_batches,1):.6f} | "
              f"PSNR_in: {val_psnr_in:5.2f} | "
              f"PSNR: {val_psnr:.2f}±{val_psnr_std:.2f} | "
              f"SSIM: {val_ssim:.4f} | "
              f"vs MSPMnet: {'+' if delta>=0 else ''}{delta:.2f} | "
              f"LR: m={muon_lr:.1e} a={adamw_lr:.1e} | "
              f"ps={current_ps}{swt_tag} | {elapsed:.0f}s")

        # ── Log ──
        with open(log_path, 'a') as f:
            f.write(f"{epoch},{tot_loss/max(n_batches,1):.6f},{val_psnr:.4f},"
                    f"{val_psnr_std:.4f},{val_ssim:.6f},{val_ssim_std:.6f},"
                    f"{val_psnr_in:.4f},{muon_lr:.2e},{adamw_lr:.2e}\n")

        # ── Checkpoint ──
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            no_improve = 0
            ema.apply(mdl)
            torch.save({
                'epoch': epoch,
                'model_state_dict': mdl.state_dict(),
                'psnr': val_psnr, 'psnr_std': val_psnr_std,
                'ssim': val_ssim, 'ssim_std': val_ssim_std,
                'config': {
                    'base_ch': BASE_CH,
                    'model': 'AttentionUNet_Pro_v7_MHDC',
                    'params': n_params,
                    'mhdc_bottleneck': True,
                    'dilations': [[1,2,3],[1,2,4]],
                    'loss': 'charb+grad+msssim+swt',
                    'val_overlap': VAL_OVERLAP,
                },
            }, os.path.join(SAVE_DIR, 'best_v7_simple.pt'))
            ema.restore(mdl)
            print(f"  ★ New best: {val_psnr:.2f} dB → saved best_v7_simple.pt")
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"\n  Early stopping at E{epoch} ({PATIENCE} epochs no improvement)")
            break

        if epoch % 25 == 0:
            ema.apply(mdl)
            torch.save({
                'epoch': epoch, 'model_state_dict': mdl.state_dict(),
                'psnr': val_psnr, 'ssim': val_ssim,
            }, os.path.join(SAVE_DIR, f'v7s_epoch{epoch}.pt'))
            ema.restore(mdl)
            print(f"  Periodic → v7s_epoch{epoch}.pt")

    # ── Final TTA eval ──
    print(f"\n{'='*75}")
    print(f"  Training complete. Best PSNR: {best_psnr:.2f} dB")

    best_path = os.path.join(SAVE_DIR, 'best_v7_simple.pt')
    if os.path.exists(best_path):
        print(f"\n  Final benchmark eval (TTA + overlap={VAL_OVERLAP})...")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        mdl.load_state_dict(ckpt['model_state_dict'])
        psnr_m, ssim_m, _, psnr_s, ssim_s = validate(mdl, val_loader, device, tta=True)
        delta = psnr_m - MSPNET_PSNR

        print(f"\n  ┌─ v7 BENCHMARK — TTA + overlap={VAL_OVERLAP} ──────────────")
        print(f"  │ AttentionUNet_v7 (MHDC)  PSNR: {psnr_m:.2f}±{psnr_s:.2f}  "
              f"SSIM: {ssim_m:.4f}±{ssim_s:.4f}")
        print(f"  │ Params: {n_params:,} ({n_params/1e6:.1f}M)")
        print(f"  │ vs MSPMnet ({MSPNET_PSNR}±{MSPNET_STD}): "
              f"{'+' if delta>=0 else ''}{delta:.2f} dB")
        print(f"  └───────────────────────────────────────────────────────")

        json.dump({
            'model': 'AttentionUNet_Pro_v7_MHDC',
            'params': n_params,
            'best_epoch': int(ckpt.get('epoch', -1)),
            'psnr_mean': psnr_m, 'psnr_std': psnr_s,
            'ssim_mean': ssim_m, 'ssim_std': ssim_s,
            'tta': True, 'overlap': VAL_OVERLAP,
            'mspnet_ref': MSPNET_PSNR, 'delta': delta,
        }, open(os.path.join(SAVE_DIR, 'eval_v7_simple.json'), 'w'), indent=2)

    print(f"{'='*75}")


if __name__ == '__main__':
    train()