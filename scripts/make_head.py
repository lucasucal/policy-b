"""Write the MLP head checkpoint: python scripts/make_head.py OUT_DIR"""
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

IN, HID = 512 * 2 + 6, 64
g = torch.Generator().manual_seed(0)
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
t = {
    "w1": torch.randn(HID, IN, generator=g) * 0.02,
    "b1": torch.zeros(HID),
    "w2": torch.randn(7, HID, generator=g) * 0.02,
    "b2": torch.tensor([0, 0, 0, 0, 0, 0, 1.0]),
}
save_file({k: v.contiguous() for k, v in t.items()}, str(out / "head.safetensors"))
print("wrote", out / "head.safetensors")
