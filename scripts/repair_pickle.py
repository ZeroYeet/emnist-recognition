#!/usr/bin/env python3
"""Repair/diagnose a potentially corrupted or nonstandard pickle file.

Usage: python scripts/repair_pickle.py path/to/file.pkl

What it tries:
- joblib.load, pickle.load, numpy.load
- Detect common compression (gzip, bz2, lzma, zip) and try decompress
- Scan for pickle protocol headers (0x80) and attempt partial loads

WARNING: unpickling runs arbitrary code. Run only on trusted files.
"""

import argparse
import io
import os
import pickle
import joblib
import gzip
import bz2
import lzma
import zipfile
import zlib
from pathlib import Path
import sys


def try_joblib(path):
    try:
        obj = joblib.load(path)
        print("Loaded with joblib.load() ->", type(obj))
        return obj
    except Exception as e:
        print("joblib.load failed:", e)
        return None


def dump_pipeline_info(obj):
    print('\nPipeline content summary:')
    try:
        if isinstance(obj, dict):
            for k, v in obj.items():
                print(f"- {k}: {type(v)}")
                # if it's an sklearn object, show class path
                try:
                    cls = type(v)
                    mod = cls.__module__
                    name = cls.__name__
                    print(f"    class: {mod}.{name}")
                except Exception:
                    pass
        else:
            print('Object is not a dict, type =', type(obj))
    except Exception as e:
        print('Failed to inspect pipeline object:', e)


def try_pickle(path):
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print("Loaded with pickle.load() ->", type(obj))
        return obj
    except Exception as e:
        print("pickle.load failed:", e)
        return None


def try_numpy(path):
    try:
        import numpy as np
        with np.load(path, allow_pickle=True) as data:
            print("Loaded with numpy.load() -> files:", list(data.keys()))
            return data
    except Exception as e:
        print("numpy.load failed:", e)
        return None


def detect_magic(data):
    if data.startswith(b"\x1f\x8b"):
        return 'gzip'
    if data.startswith(b'BZh'):
        return 'bz2'
    if data.startswith(b'\xfd7zXZ'):
        return 'xz'
    if data.startswith(b'PK\x03\x04'):
        return 'zip'
    return None


def try_decompress_all(path, data):
    magic = detect_magic(data)
    print("Detected magic:", magic)
    candidates = []
    if magic == 'gzip' or True:
        try:
            d = gzip.decompress(data)
            print("gzip decompressed: %d -> %d bytes" % (len(data), len(d)))
            candidates.append(("gzip", d))
        except Exception:
            pass
    try:
        d = bz2.decompress(data)
        print("bz2 decompressed: %d -> %d bytes" % (len(data), len(d)))
        candidates.append(("bz2", d))
    except Exception:
        pass
    try:
        d = lzma.decompress(data)
        print("lzma decompressed: %d -> %d bytes" % (len(data), len(d)))
        candidates.append(("lzma", d))
    except Exception:
        pass
    # try zlib raw
    try:
        d = zlib.decompress(data)
        print("zlib decompressed: %d -> %d bytes" % (len(data), len(d)))
        candidates.append(("zlib", d))
    except Exception:
        pass

    if magic == 'zip':
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for name in zf.namelist():
                    print('zip entry:', name)
                    candidates.append((f'zip:{name}', zf.read(name)))
        except Exception as e:
            print('zip handling failed', e)

    return candidates


def scan_and_attempt(data, out_dir):
    # Look for PROTO opcode 0x80 which starts a protocol header
    offsets = [i for i, b in enumerate(data) if b == 0x80]
    print('Found', len(offsets), 'potential pickle offsets')
    recovered = 0
    for i, off in enumerate(offsets[:200]):
        try:
            part = data[off:]
            obj = pickle.loads(part)
            fn = out_dir / f'recovered_{off}.pkl'
            with open(fn, 'wb') as f:
                pickle.dump(obj, f)
            print('Recovered object at offset', off, '-> saved to', fn)
            recovered += 1
        except Exception:
            # ignore failures
            pass
    print('Recovered', recovered, 'objects (saved to', out_dir, ')')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('path')
    p.add_argument('--out', default='recovery', help='output folder for recovered objects')
    args = p.parse_args()

    path = Path(args.path)
    if not path.exists():
        print('File not found:', path)
        return

    # Ensure repository root (parent of scripts/) is on sys.path so local modules can be imported during unpickling
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
        print('Inserted repo root into sys.path:', repo_root)

    # Quick header diagnostics
    data_preview = path.read_bytes()[:256]
    print('File size (bytes):', path.stat().st_size)
    print('First 64 bytes repr:', data_preview[:64])
    print('First 64 bytes hex :', data_preview[:64].hex())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) quick attempts
    obj = try_joblib(path)
    if obj is not None:
        print('Success: joblib.load worked. You can save or inspect the object.')
        dump_pipeline_info(obj)
        return

    obj = try_pickle(path)
    if obj is not None:
        print('Success: pickle.load worked.')
        return

    obj = try_numpy(path)
    if obj is not None:
        print('Success: numpy.load worked (npz).')
        return

    # Read bytes and attempt decompression + scan
    data = path.read_bytes()
    candidates = try_decompress_all(path, data)
    for tag, d in candidates:
        print('Trying candidate from', tag)
        try:
            obj = pickle.loads(d)
            fn = out_dir / f'recovered_from_{tag}.pkl'
            with open(fn, 'wb') as f:
                pickle.dump(obj, f)
            print('Successfully unpickled from', tag, '->', fn)
            return
        except Exception as e:
            print('Unpickle from', tag, 'failed:', e)

    # As last resort, scan for proto header and attempt partial loads
    scan_and_attempt(data, out_dir)


if __name__ == '__main__':
    main()
