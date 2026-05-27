"""
Upload LoRA checkpoints to the Modal Volume.

Run from the project root after training:
    python deploy/upload_loras.py                # uploads all checkpoints/
    python deploy/upload_loras.py --step 1500    # upload one specific step
"""

from __future__ import annotations

import argparse
from pathlib import Path

import modal


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--step", type=int, default=None, help="Upload only this step number.")
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    args = p.parse_args()

    ckpt_dir: Path = args.ckpt_dir
    if not ckpt_dir.exists():
        print(f"Checkpoint directory not found: {ckpt_dir.resolve()}")
        return

    pattern = f"lora_step_{args.step:06d}.safetensors" if args.step else "lora_step_*.safetensors"
    files = sorted(ckpt_dir.glob(pattern))

    if not files:
        print(f"No files matching '{pattern}' in {ckpt_dir.resolve()}")
        return

    volume = modal.Volume.from_name("flux-lora-weights", create_if_missing=True)

    print(f"Uploading {len(files)} file(s) to Modal Volume 'flux-lora-weights' …")
    with volume.batch_upload(force=True) as batch:
        for f in files:
            size_kb = f.stat().st_size // 1024
            print(f"  {f.name}  ({size_kb} KB)")
            batch.put_file(f, f.name)

    print("\nVerifying …")
    remote = {e.path for e in volume.listdir("/")}
    all_ok = True
    for f in files:
        ok = f.name in remote
        print(f"  {'✓' if ok else '✗ MISSING'}  {f.name}")
        if not ok:
            all_ok = False

    print("Done." if all_ok else "\nSome files failed to upload — re-run to retry.")


if __name__ == "__main__":
    main()
