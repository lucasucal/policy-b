"""Vision-conditioned residual controller.

A frozen CLIP image encoder embeds the wrist view and the goal image; a small
learned head maps [wrist_emb, goal_emb, joint_pos] -> 6 joint deltas + gripper.
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

ENCODER_SUBDIR = "encoder"
MAX_DELTA = 0.05


def weights_dir() -> Path:
    return Path(os.environ.get("RC_WEIGHTS_DIR") or "/weights")


def decode(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


class Controller:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        enc = weights_dir() / ENCODER_SUBDIR
        if not (enc / "config.json").is_file():
            raise RuntimeError(f"{enc} is missing: the encoder must be in the model snapshot")
        self.encoder = CLIPVisionModelWithProjection.from_pretrained(
            str(enc),
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        ).to(self.device).eval()
        self.proc = CLIPImageProcessor.from_pretrained(str(enc), local_files_only=True)
        head = load_file(str(weights_dir() / "head.safetensors"))
        self.w1 = head["w1"].float().to(self.device)
        self.b1 = head["b1"].float().to(self.device)
        self.w2 = head["w2"].float().to(self.device)
        self.b2 = head["b2"].float().to(self.device)

    @torch.no_grad()
    def embed(self, imgs):
        px = self.proc(images=imgs, return_tensors="pt")["pixel_values"].to(self.device, torch.bfloat16)
        e = self.encoder(pixel_values=px).image_embeds.float()
        return torch.nn.functional.normalize(e, dim=-1)

    @torch.no_grad()
    def act(self, obs: dict) -> dict:
        imgs = obs.get("images") or {}
        wrist = imgs.get("wrist")
        goal = obs.get("goal")
        if wrist and goal:
            e = self.embed([decode(wrist), decode(goal)])
            ew, eg = e[0], e[1]
        else:
            ew = eg = torch.zeros(512, device=self.device)
        jp = list((obs.get("proprio") or {}).get("joint_pos") or [])[:6]
        jp = [float(x) if np.isfinite(float(x)) else 0.0 for x in jp] + [0.0] * (6 - len(jp))
        x = torch.cat([ew, eg, torch.tensor(jp, device=self.device)])
        h = torch.tanh(self.w1 @ x + self.b1)
        out = (self.w2 @ h + self.b2).tolist()
        vals = [max(-MAX_DELTA, min(MAX_DELTA, float(v))) for v in out[:6]]
        grip = max(0.0, min(1.0, 0.5 + 0.5 * float(out[6])))
        return {"type": "joint_delta", "values": vals, "gripper": grip}
