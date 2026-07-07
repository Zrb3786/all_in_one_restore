import argparse, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--refiner_ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = torch.load(args.base_ckpt, map_location="cpu")
    ref = torch.load(args.refiner_ckpt, map_location="cpu")
    torch.save({
        "type": "DiffUIR-clean310-plus-MoCEPromptRefiner",
        "base_ckpt": base,
        "refiner_ckpt": ref,
        "note": "Single checkpoint containing clean DiffUIR-310 backbone and label-free MoCE prompt residual refiner.",
    }, args.out)
    print("[OK] saved", args.out)
if __name__ == "__main__": main()
