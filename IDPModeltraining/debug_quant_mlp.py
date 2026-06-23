# =============================================================================
# debug_quant_mlp.py
#
# Golden-reference software debugger for FPGA correlation
#
# What this script does:
#   1. Loads your quantized .pth model
#   2. Runs one inference sample
#   3. Dumps:
#        - input feature
#        - weights
#        - MAC accumulation
#        - bias add
#        - requant shift
#        - ReLU output
#   4. Produces values to compare directly with RTL simulation
#
# =============================================================================

import torch
import torch.nn as nn
import numpy as np

# =============================================================================
# CONFIG
# =============================================================================

MODEL_PATH = "model_qat.pth"

# CHANGE THIS TO MATCH RTL
SHIFT = 1

# Input vector used in RTL testbench
# MUST MATCH EXACTLY
INPUT_VECTOR = np.array([
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10,
    11,12,13,14,15,
    16,17,18,19,20
], dtype=np.int8)

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class QuantMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = torch.relu(x)

        x = self.fc3(x)

        return x

# =============================================================================
# LOAD MODEL
# =============================================================================

model = QuantMLP()

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

# Handle either raw state_dict or wrapped dict
if "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print("\n============================================================")
print("MODEL LOADED")
print("============================================================")

# =============================================================================
# EXTRACT WEIGHTS / BIASES
# =============================================================================

fc1_w = model.fc1.weight.detach().cpu().numpy()
fc1_b = model.fc1.bias.detach().cpu().numpy()

print("\nFC1 weight shape:", fc1_w.shape)
print("FC1 bias shape  :", fc1_b.shape)

# =============================================================================
# CONVERT TO INTEGER
# =============================================================================

# If already quantized:
fc1_w = np.round(fc1_w).astype(np.int8)
fc1_b = np.round(fc1_b).astype(np.int32)

x = INPUT_VECTOR.astype(np.int8)

# =============================================================================
# DEBUG ONE NEURON
# =============================================================================

NEURON = 0

print("\n============================================================")
print(f"DEBUGGING NEURON {NEURON}")
print("============================================================")

w = fc1_w[NEURON]
b = int(fc1_b[NEURON])

print("\nINPUT VECTOR:")
print(x)

print("\nWEIGHTS:")
print(w)

print("\nBIAS:")
print(b)

# =============================================================================
# MAC ACCUMULATION
# =============================================================================

acc = 0

print("\n============================================================")
print("MAC TRACE")
print("============================================================")

for i in range(20):

    xi = int(x[i])
    wi = int(w[i])

    prod = xi * wi

    acc += prod

    print(
        f"STEP {i:02d} | "
        f"x={xi:4d} | "
        f"w={wi:4d} | "
        f"prod={prod:6d} | "
        f"acc={acc:8d}"
    )

# =============================================================================
# BIAS ADD
# =============================================================================

biased = acc + b

print("\n============================================================")
print("POST BIAS")
print("============================================================")

print("ACC       =", acc)
print("BIAS      =", b)
print("ACC+BIAS  =", biased)

# =============================================================================
# REQUANT
# =============================================================================

shifted = biased >> SHIFT

print("\n============================================================")
print("REQUANT")
print("============================================================")

print("SHIFT =", SHIFT)
print("SHIFTED =", shifted)

# =============================================================================
# RELU
# =============================================================================

relu = max(0, shifted)

# int8 saturation
relu = min(127, relu)

print("\n============================================================")
print("FINAL OUTPUT")
print("============================================================")

print("FINAL INT8 =", relu)

# =============================================================================
# FULL LAYER OUTPUTS
# =============================================================================

print("\n============================================================")
print("FULL LAYER 1 OUTPUTS")
print("============================================================")

layer1_out = []

for neuron in range(64):

    acc = 0

    for i in range(20):

        acc += int(x[i]) * int(fc1_w[neuron][i])

    acc += int(fc1_b[neuron])

    acc = acc >> SHIFT

    acc = max(0, acc)

    acc = min(127, acc)

    layer1_out.append(acc)

print(np.array(layer1_out, dtype=np.int32))

# =============================================================================
# SAVE REFERENCE FILE
# =============================================================================

with open("golden_reference.txt", "w") as f:

    f.write("=====================================================\n")
    f.write("GOLDEN REFERENCE\n")
    f.write("=====================================================\n\n")

    f.write(f"Neuron: {NEURON}\n\n")

    for i in range(20):

        xi = int(x[i])
        wi = int(w[i])

        prod = xi * wi

        f.write(
            f"STEP {i:02d} "
            f"x={xi:4d} "
            f"w={wi:4d} "
            f"prod={prod:6d}\n"
        )

    f.write(f"\nFINAL_ACC={acc}\n")
    f.write(f"FINAL_OUT={relu}\n")

print("\n============================================================")
print("golden_reference.txt generated")
print("============================================================")