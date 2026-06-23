"""
numpy_model.py — torch-free MLP inference for the PYNQ ARM (Cortex-A9).

Loads the BN-folded floating-point model `model_fp.npz` (exported by
IDPModeltraining/export_fp_npz.py) and classifies 20-feature flow vectors into
{normal, ddos, botnet}.  BatchNorm is already folded into the linear layers, so
the forward pass is just:  normalize → (Wx+b → ReLU) ×2 → W3x+b3 → softmax.

The .npz is fully self-contained:
    W1,b1,W2,b2,W3,b3   — folded float32 weights/biases
    feat_min, feat_max  — per-feature [0,1] normalization bounds (FEATURE_ORDER)
    class_names         — ['normal','ddos','botnet']

Normalization bounds are baked into the npz, so this module does NOT depend on
constants.FEATURE_RANGES (kept consistent there for other importers).
"""

import numpy as np

# Default class map (overridden by the names embedded in the .npz on load).
CLASS_NAMES = {
    0: "normal",
    1: "ddos",
    2: "botnet",
}


class NumpyMLP:
    """Dependency-free (NumPy-only) MLP classifier."""

    def __init__(self, weight_file):
        w = np.load(weight_file, allow_pickle=True)

        # Folded float weights/biases
        self.W1 = w["W1"].astype(np.float32)
        self.b1 = w["b1"].astype(np.float32)
        self.W2 = w["W2"].astype(np.float32)
        self.b2 = w["b2"].astype(np.float32)
        self.W3 = w["W3"].astype(np.float32)
        self.b3 = w["b3"].astype(np.float32)

        # Pre-compute normalization vectors ONCE (not per inference) — keeps the
        # per-flow latency minimal on the Cortex-A9.
        self.feat_min = w["feat_min"].astype(np.float32)
        self.feat_max = w["feat_max"].astype(np.float32)
        self.span = np.where(self.feat_max - self.feat_min == 0,
                             np.float32(1.0),
                             self.feat_max - self.feat_min).astype(np.float32)

        # Class names from the npz (fall back to module default)
        if "class_names" in w.files:
            names = [str(c) for c in w["class_names"]]
            self.class_names = {i: n for i, n in enumerate(names)}
            global CLASS_NAMES
            CLASS_NAMES = dict(self.class_names)
        else:
            self.class_names = dict(CLASS_NAMES)

        self.num_features = self.W1.shape[1]   # 20

    # ── primitives ──────────────────────────────────────────────────────────
    def normalize_features(self, x):
        x = np.clip(x, self.feat_min, self.feat_max)
        return ((x - self.feat_min) / self.span).astype(np.float32)

    @staticmethod
    def relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def linear(x, weight, bias):
        return np.matmul(x, weight.T) + bias

    @staticmethod
    def softmax(x):
        x = x - np.max(x)
        e = np.exp(x)
        return e / (np.sum(e) + 1e-12)

    # ── inference ───────────────────────────────────────────────────────────
    def predict(self, feature_vector):
        x = np.asarray(feature_vector, dtype=np.float32)

        if x.shape[0] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {x.shape[0]}"
            )

        x = self.normalize_features(x)
        x = np.nan_to_num(x)

        # BN already folded → plain linear + ReLU
        x = self.relu(self.linear(x, self.W1, self.b1))
        x = self.relu(self.linear(x, self.W2, self.b2))
        logits = self.linear(x, self.W3, self.b3)

        probs = self.softmax(logits)
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])

        return {
            "class_id": class_id,
            "class_name": self.class_names.get(class_id, "unknown"),
            "confidence": confidence,
            "probabilities": {
                self.class_names.get(i, str(i)): float(probs[i])
                for i in range(len(probs))
            },
        }


if __name__ == "__main__":
    import time

    model = NumpyMLP("model_fp.npz")
    print(f"Loaded model_fp.npz  classes={list(model.class_names.values())}")

    dummy = np.random.rand(20).astype(np.float32)
    t0 = time.perf_counter()
    result = model.predict(dummy)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(result)
    print(f"inference time: {dt_ms:.3f} ms")
