"""
Stationary Wavelet Transform (SWT) Loss for LDCT denoising.

Based on NTIRE 2025 denoising winner strategy:
- Alternating L1/L2/SWT loss escapes local optima
- SWT preserves translation invariance (unlike DWT)
- Operates on wavelet subbands: LL (structure), LH/HL/HH (edges/texture)
- Penalizes high-freq subband differences more → better edge preservation

Uses PyTorch Haar wavelet implementation (no pywavelets dependency).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarSWT2D(nn.Module):
    """
    Stationary Wavelet Transform using Haar wavelets (2D, single level).
    
    Unlike DWT, SWT does NOT downsample — output subbands have same
    resolution as input. This is achieved by upsampling the filters
    instead of downsampling the signal (à trous algorithm).
    
    Returns: (LL, LH, HL, HH) each with same spatial dims as input.
    """
    def __init__(self):
        super().__init__()
        # Haar low-pass and high-pass filters
        lo = torch.tensor([1.0, 1.0]) / (2 ** 0.5)
        hi = torch.tensor([-1.0, 1.0]) / (2 ** 0.5)
        
        # 2D filters via outer product
        self.register_buffer('ll', (lo.unsqueeze(1) @ lo.unsqueeze(0)).unsqueeze(0).unsqueeze(0))
        self.register_buffer('lh', (hi.unsqueeze(1) @ lo.unsqueeze(0)).unsqueeze(0).unsqueeze(0))
        self.register_buffer('hl', (lo.unsqueeze(1) @ hi.unsqueeze(0)).unsqueeze(0).unsqueeze(0))
        self.register_buffer('hh', (hi.unsqueeze(1) @ hi.unsqueeze(0)).unsqueeze(0).unsqueeze(0))
    
    def forward(self, x: torch.Tensor):
        """
        x: (B, C, H, W) — operates per-channel via groups=C
        Returns: tuple of 4 tensors, each (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Expand filters to depthwise: (C, 1, 2, 2), match input dtype/device
        ll = self.ll.to(dtype=x.dtype, device=x.device).expand(C, -1, -1, -1)
        lh = self.lh.to(dtype=x.dtype, device=x.device).expand(C, -1, -1, -1)
        hl = self.hl.to(dtype=x.dtype, device=x.device).expand(C, -1, -1, -1)
        hh = self.hh.to(dtype=x.dtype, device=x.device).expand(C, -1, -1, -1)
        
        # Padding to maintain spatial size (same padding for 2×2 filter)
        x_pad = F.pad(x, (0, 1, 0, 1), mode='reflect')
        
        # Stationary WT: conv WITHOUT stride (preserves resolution)
        return (
            F.conv2d(x_pad, ll, groups=C),
            F.conv2d(x_pad, lh, groups=C),
            F.conv2d(x_pad, hl, groups=C),
            F.conv2d(x_pad, hh, groups=C),
        )


class SWTLoss(nn.Module):
    """
    Stationary Wavelet Transform Loss.
    
    Computes L1 loss on each wavelet subband with configurable weights.
    High-freq subbands (LH, HL, HH) can be weighted more to emphasize
    edge and texture preservation.
    
    Args:
        ll_weight: weight for low-freq (structure) subband
        hf_weight: weight for each high-freq subband (edges/texture)
    """
    def __init__(self, ll_weight: float = 1.0, hf_weight: float = 1.0):
        super().__init__()
        self.swt = HaarSWT2D()
        self.ll_weight = ll_weight
        self.hf_weight = hf_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_ll, pred_lh, pred_hl, pred_hh = self.swt(pred)
        tgt_ll, tgt_lh, tgt_hl, tgt_hh = self.swt(target)
        
        loss_ll = F.l1_loss(pred_ll, tgt_ll)
        loss_lh = F.l1_loss(pred_lh, tgt_lh)
        loss_hl = F.l1_loss(pred_hl, tgt_hl)
        loss_hh = F.l1_loss(pred_hh, tgt_hh)
        
        return (self.ll_weight * loss_ll +
                self.hf_weight * (loss_lh + loss_hl + loss_hh))


class AlternatingLoss(nn.Module):
    """
    NTIRE 2025 winner strategy: alternate between L1, L2, and SWT loss
    every N steps. This helps escape local optima by changing the loss
    landscape periodically.
    
    Args:
        switch_every: number of steps between loss switches
        swt_ll_weight: SWT low-freq weight
        swt_hf_weight: SWT high-freq weight
    """
    def __init__(self, switch_every: int = 1000,
                 swt_ll_weight: float = 1.0, swt_hf_weight: float = 1.0):
        super().__init__()
        self.switch_every = switch_every
        self.swt_loss = SWTLoss(swt_ll_weight, swt_hf_weight)
        self.step_count = 0
        self.current_loss = 'l1'
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        phase = (self.step_count // self.switch_every) % 3
        self.step_count += 1
        
        if phase == 0:
            self.current_loss = 'l1'
            return F.l1_loss(pred, target)
        elif phase == 1:
            self.current_loss = 'l2'
            return F.mse_loss(pred, target)
        else:
            self.current_loss = 'swt'
            return self.swt_loss(pred, target)
    
    def get_current_loss_name(self) -> str:
        return self.current_loss


if __name__ == '__main__':
    # Quick test
    pred = torch.randn(2, 1, 256, 256)
    target = torch.randn(2, 1, 256, 256)
    
    swt = SWTLoss()
    loss = swt(pred, target)
    print(f"SWT Loss: {loss.item():.6f}")
    
    alt = AlternatingLoss(switch_every=2)
    for i in range(9):
        l = alt(pred, target)
        print(f"Step {i}: {alt.get_current_loss_name()} = {l.item():.6f}")
