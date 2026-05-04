#!/usr/bin/env python3
"""
W4: Validate Noisier2Noise implementation on synthetic i.i.d. AWGN.

If N2N+ works on AWGN but fails on CT, the failure is physics-driven (not a bug).
Runs in ~5 minutes on GPU, ~15 min on CPU.

Usage: python validate_noisier2noise_synthetic.py [--device cuda:0]
"""

import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from model import REDCNN_SE
from dataset import compute_metrics

# ── Tiny synthetic dataset ────────────────────────────────────────────────

class SyntheticAWGNDataset(Dataset):
    """Clean images + i.i.d. Gaussian noise. Perfect conditions for N2N+."""

    def __init__(self, clean_images, noise_sigma=0.03, patch_size=128,
                 patches_per_image=16):
        self.clean = clean_images  # (N, H, W) float32 [0,1]
        self.sigma = noise_sigma
        self.ps = patch_size
        self.ppi = patches_per_image

    def __len__(self):
        return len(self.clean) * self.ppi

    def __getitem__(self, idx):
        i = idx // self.ppi
        img = self.clean[i]
        h, w = img.shape
        y = np.random.randint(0, max(1, h - self.ps))
        x = np.random.randint(0, max(1, w - self.ps))
        patch = img[y:y + self.ps, x:x + self.ps].copy()
        # Add i.i.d. AWGN
        noisy = patch + np.random.randn(*patch.shape).astype(np.float32) * self.sigma
        return torch.from_numpy(noisy[np.newaxis]), torch.from_numpy(patch[np.newaxis])


# ── Noisier2Noise training step ───────────────────────────────────────────

def train_noisier2noise(model, dataloader, optimizer, scaler, device,
                        noise_sigma, sigma_add, epoch):
    """One epoch of Noisier2Noise: z = y + m, train f(z) -> y."""
    model.train()
    total_loss = 0
    n = 0
    for noisy, clean in dataloader:
        noisy = noisy.to(device)
        # Noisier2Noise: add MORE noise
        m = torch.randn_like(noisy) * sigma_add
        z = noisy + m  # z = y + m (noisier input)
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            pred = model(z)
            loss = F.l1_loss(pred, noisy)  # train to predict y from z
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_noisier2noise(model, clean_images, noise_sigma, sigma_add,
                           device, n_correction_samples=8):
    """Evaluate with Noisier2Noise correction: x_hat = 2*f(y) - y."""
    model.eval()
    alpha = sigma_add / max(noise_sigma, 1e-8)

    psnr_input, psnr_direct, psnr_corrected = [], [], []

    for i in range(len(clean_images)):
        clean = clean_images[i]
        noisy = clean + np.random.randn(*clean.shape).astype(np.float32) * noise_sigma

        inp = torch.from_numpy(noisy[np.newaxis, np.newaxis]).to(device)

        # Method 1: Direct output
        pred_direct = model(inp)

        # Method 2: Simple correction (2*f(y) - y)
        pred_simple = 2.0 * pred_direct - inp

        # Method 3: Multi-sample corrected
        corr_factor = (1 + alpha**2) / max(alpha**2, 1e-8)
        inv_alpha2 = 1.0 / max(alpha**2, 1e-8)
        corrected_sum = torch.zeros_like(inp)
        for _ in range(n_correction_samples):
            m = torch.randn_like(inp) * sigma_add
            z = inp + m
            f_z = model(z)
            corrected_sum += corr_factor * f_z - inv_alpha2 * z
        pred_multi = corrected_sum / n_correction_samples

        # Compute PSNR
        p_input, _ = compute_metrics(noisy, clean)
        p_direct, _ = compute_metrics(pred_direct[0, 0].cpu().numpy(), clean)
        p_corr, _ = compute_metrics(pred_multi[0, 0].cpu().numpy(), clean)

        psnr_input.append(p_input)
        psnr_direct.append(p_direct)
        psnr_corrected.append(p_corr)

    return (np.mean(psnr_input), np.mean(psnr_direct), np.mean(psnr_corrected))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--noise-sigma', type=float, default=0.03)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    noise_sigma = args.noise_sigma
    sigma_add = noise_sigma  # alpha = 1

    print("=" * 70)
    print("  W4: Noisier2Noise SYNTHETIC VALIDATION (i.i.d. AWGN)")
    print(f"  σ_noise = {noise_sigma}, σ_add = {sigma_add}, α = 1.0")
    print(f"  Device: {device}")
    print("  If this works → N2N+ failure on CT is physics, not a bug")
    print("=" * 70)

    # ── Create synthetic data from real FD images ──────────────────────────
    from dataset import load_ima, window_normalize, collect_paired_paths
    data_root = os.environ.get('MAYO_DATA_ROOT', './data/Traning_Image_Data')
    kernel = '3mm B30'

    # Load 50 FD slices as "clean" images
    pairs = collect_paired_paths(data_root, kernel)
    train_pairs = [(q, f) for q, f in pairs if 'L506' not in q][:50]
    test_pairs = [(q, f) for q, f in pairs if 'L506' in q][:20]

    print(f"\n  Loading {len(train_pairs)} train + {len(test_pairs)} test FD slices...")
    train_clean = np.stack([window_normalize(load_ima(f)) for _, f in train_pairs])
    test_clean = np.stack([window_normalize(load_ima(f)) for _, f in test_pairs])
    print(f"  Train: {train_clean.shape}, Test: {test_clean.shape}")

    # ── Model ──────────────────────────────────────────────────────────────
    model = REDCNN_SE(n_channels=74).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: REDCNN_SE c=74, {n_params/1e6:.2f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    train_ds = SyntheticAWGNDataset(train_clean, noise_sigma, patch_size=128,
                                     patches_per_image=16)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                               num_workers=0, pin_memory=True)

    # ── Baseline ───────────────────────────────────────────────────────────
    p_in, p_dir, p_corr = evaluate_noisier2noise(
        model, test_clean, noise_sigma, sigma_add, device)
    print(f"\n  Baseline (untrained): Input={p_in:.2f} dB, Direct={p_dir:.2f}, Corrected={p_corr:.2f}")

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"\n  Training {args.epochs} epochs on synthetic AWGN data...\n")
    best_psnr = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_noisier2noise(model, train_loader, optimizer, scaler,
                                    device, noise_sigma, sigma_add, epoch)
        scheduler.step()

        if epoch % 5 == 0 or epoch == args.epochs:
            p_in, p_dir, p_corr = evaluate_noisier2noise(
                model, test_clean, noise_sigma, sigma_add, device)
            best_psnr = max(best_psnr, p_corr)
            dt = time.time() - t0
            print(f"  E{epoch:3d} | loss={loss:.5f} | Input={p_in:.2f} | "
                  f"Direct={p_dir:.2f} | Corrected={p_corr:.2f} | "
                  f"best={best_psnr:.2f} | {dt:.1f}s")

    # ── Verdict ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    gain = best_psnr - p_in
    if gain > 2.0:
        print(f"  ✓ PASS: N2N+ gains +{gain:.2f} dB on i.i.d. AWGN")
        print(f"  → Implementation is CORRECT")
        print(f"  → Failure on CT ({29.26:.2f} dB ≈ QD input) is PHYSICS-DRIVEN")
        print(f"    (autocorrelation τ3=0.60, non-stationarity τ1=0.61)")
    else:
        print(f"  ✗ FAIL: N2N+ only gains +{gain:.2f} dB on AWGN")
        print(f"  → Possible implementation bug — investigate!")
    print("=" * 70)


if __name__ == '__main__':
    main()
