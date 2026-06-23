"""
model_reference.py  —  Reference-only PyTorch model definition and loader.

NOT used at runtime by the PYNQ pipeline (which runs pure NumPy via
PYNQ/numpy_model.py). Kept for documentation of the trained architecture
and as a self-test tool / retraining reference.

Architecture inferred directly from best_model.pth state-dict keys:
    layer1  (Linear 20→64)  + bn1 (BatchNorm1d 64)  + ReLU
    layer2  (Linear 64→32)  + bn2 (BatchNorm1d 32)  + ReLU
    layer3  (Linear 32→5)   — final logits (no activation)

Output: 5 classes mapped to:
    0 → normal         (was WEB_BROWSING  in training)
    1 → ddos           (was BULK_TRANSFER in training — high-rate bulk)
    2 → botnet
    3 → botnet         (was HIGH_RATE     in training — periodic bursts)
    4 → other          (was ANOMALOUS     in training)

Note: class labels are a best-effort mapping.  If you retrain on a labelled
dataset that explicitly contains normal/ddos/botnet classes, update
CLASS_NAMES accordingly.

Requires (not in core requirements.txt): pip install torch
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "PYNQ")))

# ── class labels ──────────────────────────────────────────────────────────────

CLASS_NAMES = {
    0: "normal",
    1: "ddos",
    2: "botnet"
}
NUM_CLASSES  = len(CLASS_NAMES)
NUM_FEATURES = 20


# ── model definition ──────────────────────────────────────────────────────────

class TrafficClassifier(nn.Module):
    """
    MLP traffic classifier.  Architecture matches best_model.pth exactly:
        Linear(20→64) → BN(64) → ReLU → Linear(64→32) → BN(32) → ReLU → Linear(32→5)
    """

    def __init__(self, num_features: int = NUM_FEATURES,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)   # raw logits

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run forward pass and return (class_indices, confidences).

        Args:
            x: float32 tensor of shape (N, 20) — already normalized floats.
               Normalization is done by the feature extractor; do NOT pass
               raw INT8 values here.

        Returns:
            class_indices: LongTensor of shape (N,)
            confidences:   FloatTensor of shape (N,) in [0, 1]
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs  = F.softmax(logits, dim=1)
            confs, classes = torch.max(probs, dim=1)
        return classes, confs


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(path: str, device: str = "cpu") -> TrafficClassifier:
    """
    Load TrafficClassifier from a .pth checkpoint.

    Accepts both plain state-dict files and checkpoints that wrap the
    state dict under a 'state_dict' or 'model_state_dict' key.

    Always sets eval mode and pins to CPU (ARM Cortex-A9 on PYNQ-Z2).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            # Assume the dict itself is the state dict
            state_dict = checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")

    model = TrafficClassifier()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ── feature normalization (float pipeline, not INT8) ─────────────────────────

# These normalization ranges must match FEATURE_RANGES in constants.py exactly.
import torch as _torch

_RANGE_MIN_T: torch.Tensor
_RANGE_MAX_T: torch.Tensor
_RANGE_SPAN_T: torch.Tensor

def _build_norm_tensors() -> None:
    """Pre-compute normalization tensors for fast inference."""
    global _RANGE_MIN_T, _RANGE_MAX_T, _RANGE_SPAN_T
    from constants import FEATURE_ORDER, FEATURE_RANGES
    mins  = [FEATURE_RANGES[n][0] for n in FEATURE_ORDER]
    maxs  = [FEATURE_RANGES[n][1] for n in FEATURE_ORDER]
    spans = [max(hi - lo, 1e-9) for lo, hi in zip(mins, maxs)]
    _RANGE_MIN_T  = torch.tensor(mins,  dtype=torch.float32)
    _RANGE_MAX_T  = torch.tensor(maxs,  dtype=torch.float32)
    _RANGE_SPAN_T = torch.tensor(spans, dtype=torch.float32)


def normalize_features(float_vec) -> torch.Tensor:
    """
    Normalize a 20-element float list/array to a [0, 1]-ish float32 tensor.

    This converts the raw flow features into the same input domain the model
    was trained on.  The normalization matches the FEATURE_RANGES clipping and
    linear scaling in quantizer.py, but keeps float precision.

    Args:
        float_vec: list or numpy array of 20 floats (FEATURE_ORDER order).

    Returns:
        torch.Tensor of shape (1, 20), dtype float32, ready for model.predict().
    """
    if _RANGE_MIN_T is None:
        _build_norm_tensors()
    x = torch.tensor(float_vec, dtype=torch.float32)
    x = torch.clamp(x, _RANGE_MIN_T, _RANGE_MAX_T)
    x = (x - _RANGE_MIN_T) / _RANGE_SPAN_T   # [0.0, 1.0]
    return x.unsqueeze(0)                      # shape (1, 20)


# Initialise normalization tensors at import time.
try:
    _build_norm_tensors()
except ImportError:
    # constants.py not on path yet; will be initialised lazily on first call.
    _RANGE_MIN_T = _RANGE_MAX_T = _RANGE_SPAN_T = None  # type: ignore[assignment]


# ── quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    default_path = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "IDPModeltraining", "model_fp.pth"))
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    m = load_model(path)
    print(f"Model loaded from {path}")
    print(m)
    dummy = torch.randn(4, NUM_FEATURES)
    classes, confs = m.predict(dummy)
    for i, (c, cf) in enumerate(zip(classes.tolist(), confs.tolist())):
        print(f"  sample {i}: class={CLASS_NAMES[c]} ({c})  conf={cf:.3f}")