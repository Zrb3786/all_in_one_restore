#!/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image

IMG_EXTS = {'.jpg', '.jpeg', '.png'}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--expect_count', type=int, default=-1)
    args = ap.parse_args()
    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    in_files = sorted([p for p in inp.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    out_files = sorted([p for p in out.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    out_map = {p.name: p for p in out_files}
    missing = []
    bad_size = []
    bad_mode = []
    for p in in_files:
        op = out_map.get(p.name)
        if op is None and p.suffix.lower() not in {'.jpg', '.jpeg'}:
            op = out / (p.stem + '.jpg')
        if op is None or not op.exists():
            missing.append(p.name)
            continue
        im = Image.open(p); om = Image.open(op)
        if om.mode != 'RGB': bad_mode.append(op.name)
        if im.size != om.size: bad_size.append((p.name, im.size, om.size))
    print('inputs:', len(in_files), 'outputs:', len(out_files))
    print('missing:', len(missing), missing[:10])
    print('bad_size:', len(bad_size), bad_size[:5])
    print('bad_mode:', len(bad_mode), bad_mode[:10])
    if args.expect_count > 0 and len(out_files) != args.expect_count:
        raise SystemExit(f'[ERROR] expected {args.expect_count} outputs, got {len(out_files)}')
    if missing or bad_size or bad_mode:
        raise SystemExit('[ERROR] verification failed')
    print('[OK] verification passed')

if __name__ == '__main__': main()
