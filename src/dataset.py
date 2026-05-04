"""
CT Image Dataset — loads paired full-dose / quarter-dose .IMA DICOM slices.

Enhancements:
- predict_overlap_tile(): overlap-tile inference for accurate full-slice evaluation
  (model trained on patches; overlap-tile avoids border artifacts)
- compute_metrics(): proper PSNR + SSIM via skimage (accurate reporting)
- Safer random crop bounds (clips to valid range instead of crashing at edge)
"""

import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pydicom
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def load_ima(path: str) -> np.ndarray:
    """Load a Siemens .IMA DICOM file and return HU-valued float32 array."""
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    return arr * slope + intercept


def window_normalize(hu: np.ndarray, wl: float = 40.0, ww: float = 400.0) -> np.ndarray:
    """Window/level normalize HU values to [0, 1] for abdomen soft-tissue."""
    low = wl - ww / 2.0
    high = wl + ww / 2.0
    return np.clip((hu - low) / (high - low), 0.0, 1.0).astype(np.float32)


def collect_paired_paths(
    training_root: str,
    kernel: str = "1mm B30",
) -> List[Tuple[str, str]]:
    """
    Collect paired (quarter-dose, full-dose) file paths from training data.

    Training structure:
      Traning_Image_Data/{kernel}/full_1mm/{PATIENT}/{dose_folder}/*.IMA
    """
    base = Path(training_root) / kernel

    if "1mm B30" in kernel:
        fd_dir, qd_dir = "full_1mm", "quarter_1mm"
    elif "1mm D45" in kernel:
        fd_dir, qd_dir = "full_1mm_sharp", "quarter_1mm_sharp"
    elif "3mm B30" in kernel:
        fd_dir, qd_dir = "full_3mm", "quarter_3mm"
    elif "3mm D45" in kernel:
        fd_dir, qd_dir = "full_3mm_sharp", "quarter_3mm_sharp"
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    fd_root = base / fd_dir
    qd_root = base / qd_dir

    pairs = []
    for patient_dir in sorted(fd_root.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name

        qd_patient = qd_root / patient_id
        if not qd_patient.exists():
            print(f"WARNING: No quarter-dose match for {patient_id}, skipping")
            continue

        fd_imas = sorted(_find_ima_files(patient_dir))
        qd_imas = sorted(_find_ima_files(qd_patient))

        if len(fd_imas) != len(qd_imas):
            print(f"WARNING: {patient_id} slice count mismatch fd={len(fd_imas)} qd={len(qd_imas)}, using min")
            n = min(len(fd_imas), len(qd_imas))
            fd_imas, qd_imas = fd_imas[:n], qd_imas[:n]

        for fd_path, qd_path in zip(fd_imas, qd_imas):
            pairs.append((str(qd_path), str(fd_path)))

    return pairs


def _find_ima_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.IMA"))


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float = 1.0,
) -> Tuple[float, float]:
    """
    Compute PSNR and SSIM for LDCT denoising evaluation.

    Uses skimage's structural_similarity (LOCAL windowed SSIM).
    The previous fallback computed GLOBAL SSIM which is incorrect
    and produces inflated values (~0.99 instead of ~0.90).

    Args:
        pred:       (H, W) float32, model output in [0, 1]
        target:     (H, W) float32, ground truth in [0, 1]
        data_range: dynamic range of the data (1.0 for [0,1] normalized)

    Returns:
        (psnr_db, ssim_value)
    """
    import numpy as np
    from skimage.metrics import structural_similarity

    # PSNR
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10:
        psnr_val = float('inf')
    else:
        psnr_val = 10.0 * np.log10(data_range ** 2 / mse)

    # SSIM — LOCAL windowed (correct)
    # win_size=7 is standard for 256×256+ images
    # gaussian_weights=True matches the original SSIM paper (Wang 2004)
    ssim_val = structural_similarity(
        pred, target,
        data_range=data_range,
        win_size=7,
        channel_axis=None,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False
    )

    return psnr_val, ssim_val


# ── Overlap-tile inference ─────────────────────────────────────────────────

@torch.no_grad()
def predict_overlap_tile(
    model: nn.Module,
    img: torch.Tensor,
    patch_size: int = 256,
    overlap: int = 64,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Overlap-tile inference for a full CT slice.

    Sliding window with `overlap` pixel overlap between adjacent patches.
    Predictions in overlap regions are averaged, eliminating border artifacts
    that arise when a model trained on patches is applied to full slices.

    Args:
        img: (1, H, W) or (H, W) tensor in [0, 1]
        patch_size: size of patches the model was trained on
        overlap: number of pixels to overlap (use ≥ 25% of patch_size)

    Returns: (1, H, W) denoised tensor
    """
    if img.dim() == 2:
        img = img.unsqueeze(0)
    C, H, W = img.shape
    stride = patch_size - overlap

    output = torch.zeros(C, H, W)
    count = torch.zeros(C, H, W)

    # Build grid of top-left corners covering the full image
    ys = list(range(0, max(H - patch_size, 0) + 1, stride))
    xs = list(range(0, max(W - patch_size, 0) + 1, stride))
    # Ensure the last strip is always covered
    if not ys or ys[-1] + patch_size < H:
        ys.append(max(0, H - patch_size))
    if not xs or xs[-1] + patch_size < W:
        xs.append(max(0, W - patch_size))

    model.eval()
    for y in ys:
        for x in xs:
            y2 = min(y + patch_size, H)
            x2 = min(x + patch_size, W)
            patch = img[:, y:y2, x:x2]

            # Pad if the slice is smaller than patch_size (rare edge case)
            ph, pw = patch_size - patch.shape[1], patch_size - patch.shape[2]
            if ph > 0 or pw > 0:
                patch = torch.nn.functional.pad(patch, (0, pw, 0, ph))

            inp = patch.unsqueeze(0).to(device)
            pred = model(inp).squeeze(0).cpu()

            # Crop back if padded
            output[:, y:y2, x:x2] += pred[:, :y2 - y, :x2 - x]
            count[:, y:y2, x:x2] += 1

    return output / count.clamp(min=1)


# ── Datasets ───────────────────────────────────────────────────────────────

class CTDenoiseDataset(Dataset):
    """On-disk dataset: loads DICOM pairs at __getitem__ time."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        patch_size: int = 64,
        patches_per_slice: int = 8,
        augment: bool = True,
        window_level: float = 40.0,
        window_width: float = 400.0,
    ):
        self.pairs = pairs
        self.patch_size = patch_size
        self.patches_per_slice = patches_per_slice
        self.augment = augment
        self.wl = window_level
        self.ww = window_width

    def __len__(self):
        return len(self.pairs) * self.patches_per_slice

    def __getitem__(self, idx):
        slice_idx = idx // self.patches_per_slice
        qd_path, fd_path = self.pairs[slice_idx]

        qd = window_normalize(load_ima(qd_path), self.wl, self.ww)
        fd = window_normalize(load_ima(fd_path), self.wl, self.ww)

        h, w = qd.shape
        ps = self.patch_size
        y = np.random.randint(0, max(1, h - ps))
        x = np.random.randint(0, max(1, w - ps))
        qp = qd[y:y + ps, x:x + ps].copy()
        fp = fd[y:y + ps, x:x + ps].copy()

        if self.augment:
            qp, fp = _augment_pair(qp, fp)

        return (torch.from_numpy(qp[np.newaxis]),
                torch.from_numpy(fp[np.newaxis]))


class CTFullSliceDataset(Dataset):
    """Full 512×512 slices for evaluation (no augmentation)."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        window_level: float = 40.0,
        window_width: float = 400.0,
    ):
        self.pairs = pairs
        self.wl = window_level
        self.ww = window_width

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        qd_path, fd_path = self.pairs[idx]
        qd = window_normalize(load_ima(qd_path), self.wl, self.ww)
        fd = window_normalize(load_ima(fd_path), self.wl, self.ww)
        return (torch.from_numpy(qd[np.newaxis]),
                torch.from_numpy(fd[np.newaxis]))


# ── Pre-cached versions (all data in RAM) ──────────────────────────────────

def preload_pairs(
    pairs: List[Tuple[str, str]],
    wl: float = 40.0,
    ww: float = 400.0,
    label: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load all paired slices into RAM as windowed float32 arrays."""
    n = len(pairs)
    sample = load_ima(pairs[0][0])
    h, w = sample.shape
    qd_all = np.empty((n, h, w), dtype=np.float32)
    fd_all = np.empty((n, h, w), dtype=np.float32)

    for i, (qd_path, fd_path) in enumerate(pairs):
        if i % 500 == 0:
            print(f"  [{label}] Loading slice {i}/{n}...", flush=True)
        qd_all[i] = window_normalize(load_ima(qd_path), wl, ww)
        fd_all[i] = window_normalize(load_ima(fd_path), wl, ww)

    print(f"  [{label}] Loaded {n} slices "
          f"({qd_all.nbytes / 1e9:.1f} + {fd_all.nbytes / 1e9:.1f} GB)", flush=True)
    return qd_all, fd_all


class CachedCTDataset(Dataset):
    """Training dataset backed by pre-loaded numpy arrays. Zero disk I/O."""

    def __init__(
        self,
        qd_stack: np.ndarray,
        fd_stack: np.ndarray,
        patch_size: int = 64,
        patches_per_slice: int = 8,
        augment: bool = True,
    ):
        self.qd = qd_stack
        self.fd = fd_stack
        self.ps = patch_size
        self.pps = patches_per_slice
        self.augment = augment

    def __len__(self):
        return len(self.qd) * self.pps

    def __getitem__(self, idx):
        s = idx // self.pps
        qd = self.qd[s]
        fd = self.fd[s]

        h, w = qd.shape
        y = np.random.randint(0, max(1, h - self.ps))
        x = np.random.randint(0, max(1, w - self.ps))
        qp = qd[y:y + self.ps, x:x + self.ps].copy()
        fp = fd[y:y + self.ps, x:x + self.ps].copy()

        if self.augment:
            qp, fp = _augment_pair(qp, fp)

        return (torch.from_numpy(qp[np.newaxis]),
                torch.from_numpy(fp[np.newaxis]))


class CachedFullSliceDataset(Dataset):
    """Evaluation dataset backed by pre-loaded numpy arrays."""

    def __init__(self, qd_stack: np.ndarray, fd_stack: np.ndarray):
        self.qd = qd_stack
        self.fd = fd_stack

    def __len__(self):
        return len(self.qd)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.qd[idx][np.newaxis].copy()),
                torch.from_numpy(self.fd[idx][np.newaxis].copy()))


# ── Augmentation helpers ───────────────────────────────────────────────────

def _augment_pair(qp: np.ndarray, fp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Synchronized geometric augmentation: flips + 90° rotations."""
    if np.random.random() > 0.5:
        qp = np.flip(qp, axis=1).copy()
        fp = np.flip(fp, axis=1).copy()
    if np.random.random() > 0.5:
        qp = np.flip(qp, axis=0).copy()
        fp = np.flip(fp, axis=0).copy()
    k = np.random.randint(0, 4)
    if k:
        qp = np.rot90(qp, k).copy()
        fp = np.rot90(fp, k).copy()
    return qp, fp
