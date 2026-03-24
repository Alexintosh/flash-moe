#!/usr/bin/env python3
"""Push a Flash-MoE model directory to iPhone via USB.

Usage:
    python3 push_model_to_iphone.py /path/to/model [bundle_id]
"""

import sys
import os
import time
from pathlib import Path

from pymobiledevice3.lockdown import create_using_usbmux
from pymobiledevice3.services.house_arrest import HouseArrestService


def human_size(nbytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/BK/flash-moe-397B"
    bundle_id = sys.argv[2] if len(sys.argv) > 2 else "com.alexintosh.flashmoe"

    src = Path(src)
    if not src.is_dir():
        print(f"ERROR: {src} not found")
        sys.exit(1)

    model_name = src.name

    # Collect all files
    files = sorted(src.rglob("*"))
    files = [f for f in files if f.is_file()]
    total_size = sum(f.stat().st_size for f in files)

    print(f"=== Flash-MoE iPhone Model Push ===")
    print(f"Source:  {src}")
    print(f"Model:   {model_name}")
    print(f"Bundle:  {bundle_id}")
    print(f"Files:   {len(files)}")
    print(f"Size:    {human_size(total_size)}")
    print()

    # Connect
    print("Connecting to iPhone...")
    lockdown = create_using_usbmux()
    afc = HouseArrestService(lockdown=lockdown, bundle_id=bundle_id)

    # List existing docs
    docs = afc.listdir("/Documents")
    print(f"Documents: {docs}")
    print()

    # Create all directories
    dirs = sorted(set(f"Documents/{model_name}/{f.parent.relative_to(src)}"
                      for f in files if f.parent != src))
    dirs = [f"Documents/{model_name}"] + dirs

    for d in dirs:
        try:
            afc.makedirs(d)
            print(f"  mkdir {d}")
        except Exception:
            pass  # already exists

    # Check what's already on the phone
    existing = set()
    def list_remote_recursive(path, prefix=""):
        try:
            for entry in afc.listdir(path):
                if entry.startswith('.'):
                    continue
                full = f"{path}/{entry}"
                rel = f"{prefix}{entry}" if prefix else entry
                try:
                    # If we can list it, it's a directory
                    afc.listdir(full)
                    list_remote_recursive(full, f"{rel}/")
                except:
                    existing.add(rel)
        except:
            pass

    print("Checking existing files on iPhone...")
    list_remote_recursive(f"Documents/{model_name}")
    print(f"  {len(existing)} files already on phone")

    # Filter to only missing files
    remaining = []
    skipped_bytes = 0
    for f in files:
        rel = str(f.relative_to(src))
        if rel in existing:
            skipped_bytes += f.stat().st_size
        else:
            remaining.append(f)

    remaining_size = sum(f.stat().st_size for f in remaining)
    print(f"  Skipping {len(files) - len(remaining)} files ({human_size(skipped_bytes)})")
    print(f"  Remaining: {len(remaining)} files ({human_size(remaining_size)})")
    print()

    if not remaining:
        print("All files already on iPhone!")
        return

    # Push remaining files
    t0 = time.time()
    pushed_bytes = 0

    for i, f in enumerate(remaining):
        rel = f.relative_to(src)
        remote = f"Documents/{model_name}/{rel}"
        size = f.stat().st_size

        elapsed = time.time() - t0
        speed = pushed_bytes / elapsed if elapsed > 0 else 0
        eta = (remaining_size - pushed_bytes) / speed if speed > 0 else 0

        print(f"[{i+1}/{len(remaining)}] {rel} ({human_size(size)}) "
              f"[{human_size(speed)}/s, ETA {int(eta)}s]", end=" ... ", flush=True)

        try:
            afc.push(str(f), remote)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")

        pushed_bytes += size

    elapsed = time.time() - t0
    print(f"\n=== Done: {human_size(total_size)} in {elapsed:.0f}s "
          f"({human_size(total_size / elapsed)}/s) ===")


if __name__ == "__main__":
    main()
