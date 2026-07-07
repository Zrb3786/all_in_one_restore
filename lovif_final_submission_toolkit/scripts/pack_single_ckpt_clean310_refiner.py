#!/usr/bin/env python3
"""Pack DiffUIR clean-310 and MoCE refiner into ONE parameter file.

This creates the single trained model parameter file required for final submission.
It stores two submodules inside one checkpoint dictionary:
  - base_diffuir: clean DiffUIR checkpoint, e.g. model-310.pt
  - refiner: MoCEPromptRefiner checkpoint, e.g. refiner_step150000.pt

No category-specific routing or multiple external checkpoints are used at inference.
"""
import argparse
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_ckpt", required=True, help="DiffUIR clean checkpoint, e.g. model-310.pt")
    p.add_argument("--refiner_ckpt", required=True, help="MoCE refiner checkpoint, e.g. refiner_step150000.pt")
    p.add_argument("--out", required=True, help="Output single checkpoint .pt")
    p.add_argument("--base_milestone", type=int, default=310)
    p.add_argument("--method_name", default="DiffUIR-clean310-plus-MoCEPromptRefiner-v2-150k")
    args = p.parse_args()

    base_path = Path(args.base_ckpt)
    ref_path = Path(args.refiner_ckpt)
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    if not ref_path.exists():
        raise FileNotFoundError(ref_path)

    print(f"[INFO] loading base   : {base_path}")
    base = torch.load(str(base_path), map_location="cpu")
    print(f"[INFO] loading refiner: {ref_path}")
    ref = torch.load(str(ref_path), map_location="cpu")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "lovif_final_single_model",
        "method_name": args.method_name,
        "base_kind": "DiffUIR",
        "base_milestone": args.base_milestone,
        "base_ckpt": base,
        "refiner_kind": "MoCEPromptRefiner",
        "refiner_ckpt": ref,
        "inference_order": ["DiffUIR_base", "MoCEPromptRefiner"],
        "save_format": {"format": "JPEG", "quality": 96, "subsampling": 0, "optimize": True},
        "note": "Single checkpoint containing clean DiffUIR backbone and one all-in-one residual refiner. No external degradation labels are used at inference.",
    }
    torch.save(payload, str(out))
    print(f"[OK] saved single checkpoint: {out}")
    print(f"[OK] size MB: {out.stat().st_size / 1024 / 1024:.2f}")


if __name__ == "__main__":
    main()
