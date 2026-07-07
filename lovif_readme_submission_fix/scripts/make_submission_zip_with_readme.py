#!/usr/bin/env python3
"""Create Codabench submission zip with flat jpg files and readme.txt."""
import argparse
from pathlib import Path
import re
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", required=True)
    p.add_argument("--zip_path", required=True)
    p.add_argument("--expect_count", type=int, default=-1)
    p.add_argument("--require_readme", action="store_true", default=True)
    args = p.parse_args()

    result_dir = Path(args.result_dir)
    zip_path = Path(args.zip_path)
    if not result_dir.exists():
        raise FileNotFoundError(result_dir)
    jpgs = sorted(result_dir.glob("*.jpg"))
    readme = result_dir / "readme.txt"
    print("[INFO] jpg count:", len(jpgs))
    print("[INFO] readme exists:", readme.exists())

    if args.expect_count > 0 and len(jpgs) != args.expect_count:
        raise SystemExit(f"[ERROR] expected {args.expect_count} jpgs, got {len(jpgs)}")
    bad = [p.name for p in jpgs if not re.match(r"^\d{4}\.jpg$", p.name)]
    if bad:
        raise SystemExit(f"[ERROR] bad filename sample={bad[:20]}")
    if args.require_readme and not readme.exists():
        raise SystemExit("[ERROR] readme.txt missing")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as z:
        for img in jpgs:
            z.write(img, arcname=img.name)
        z.write(readme, arcname="readme.txt")
    print("[OK] saved:", zip_path)


if __name__ == "__main__":
    main()
