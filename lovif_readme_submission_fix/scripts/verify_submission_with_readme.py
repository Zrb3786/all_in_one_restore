#!/usr/bin/env python3
import argparse
from pathlib import Path
import zipfile
import re
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip_path", required=True)
    p.add_argument("--expect_count", type=int, default=500)
    p.add_argument("--input_dir", default=None)
    args = p.parse_args()
    zpath = Path(args.zip_path)
    if not zpath.exists():
        raise FileNotFoundError(zpath)
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        jpgs = [n for n in names if n.lower().endswith('.jpg')]
        others = [n for n in names if not (n.lower().endswith('.jpg') or n == 'readme.txt')]
        print('all files:', len(names))
        print('jpg count:', len(jpgs))
        print('readme:', 'readme.txt' in names)
        print('others sample:', others[:20])
        bad = [n for n in jpgs if not re.match(r'^\d{4}\.jpg$', Path(n).name)]
        print('bad names:', len(bad), bad[:20])
        if len(jpgs) != args.expect_count:
            raise SystemExit(f'[ERROR] expected {args.expect_count} jpgs, got {len(jpgs)}')
        if 'readme.txt' not in names:
            raise SystemExit('[ERROR] readme.txt missing')
        if others:
            raise SystemExit('[ERROR] unexpected files exist')
        print('readme content:\n' + z.read('readme.txt').decode('utf-8', errors='replace'))

    if args.input_dir:
        input_dir = Path(args.input_dir)
        with zipfile.ZipFile(zpath) as z:
            tmp = Path('/tmp/lovif_verify_zip_extract')
            if tmp.exists():
                import shutil; shutil.rmtree(tmp)
            tmp.mkdir(parents=True, exist_ok=True)
            z.extractall(tmp)
        for i, jpg in enumerate(sorted(tmp.glob('*.jpg'))[:20]):
            inp = input_dir / jpg.name
            if inp.exists():
                im = Image.open(inp); om = Image.open(jpg)
                if im.size != om.size:
                    raise SystemExit(f'[ERROR] size mismatch {jpg.name}: input {im.size}, output {om.size}')
        print('[OK] optional size spot-check passed')
    print('[OK] submission zip looks valid')

if __name__ == '__main__': main()
