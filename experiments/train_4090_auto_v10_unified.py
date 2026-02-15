"""
Unified two-stage trainer launcher (v8 -> v9) for DiTSR.

Purpose:
- Remove manual handover between v8 and v9 scripts.
- Keep existing proven training scripts unchanged in core logic.
- Enforce a reproducible handover protocol by step.

Usage example:
python experiments/train_4090_auto_v10_unified.py \
  --phase1-steps 12000 \
  --phase2-steps 8000 \
  --v8-script experiments/train_4090_auto_v8_edge_gtmask.py \
  --v9-script experiments/train_4090_auto_v9_gan_edge_gtmask.py
"""

from typing import Optional, Dict

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def run_phase(script: str, max_steps: int, extra_env: Optional[Dict[str, str]] = None):
    env = os.environ.copy()
    env["MAX_TRAIN_STEPS"] = str(int(max_steps))
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    cmd = ["python", script]
    print(f"▶ Running: {' '.join(cmd)} with MAX_TRAIN_STEPS={max_steps}")
    subprocess.run(cmd, env=env, check=True)


def copy_ckpt_for_handover(v8_last_ckpt: Path, v9_last_ckpt: Path):
    v9_last_ckpt.parent.mkdir(parents=True, exist_ok=True)
    if not v8_last_ckpt.exists():
        raise FileNotFoundError(f"v8 handover checkpoint not found: {v8_last_ckpt}")
    shutil.copy2(v8_last_ckpt, v9_last_ckpt)
    print(f"📦 Handover checkpoint copied:\n  {v8_last_ckpt}\n  -> {v9_last_ckpt}")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified v8->v9 training launcher")
    parser.add_argument("--phase1-steps", type=int, required=True, help="Max optimizer steps for v8 phase")
    parser.add_argument("--phase2-steps", type=int, required=True, help="Max optimizer steps for v9 phase")
    parser.add_argument("--v8-script", default="experiments/train_4090_auto_v8_edge_gtmask.py")
    parser.add_argument("--v9-script", default="experiments/train_4090_auto_v9_gan_edge_gtmask.py")
    parser.add_argument(
        "--v8-last-ckpt",
        default="experiments_results/train_4090_auto_v8/checkpoints/last.pth",
        help="v8 last checkpoint path used for handover",
    )
    parser.add_argument(
        "--v9-last-ckpt",
        default="experiments_results/train_4090_auto_v9_gan/checkpoints/last.pth",
        help="v9 resume checkpoint path",
    )
    parser.add_argument("--skip-phase1", action="store_true", help="Skip v8 phase and only run v9 from provided handover ckpt")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    v8_last = Path(args.v8_last_ckpt)
    v9_last = Path(args.v9_last_ckpt)

    print("=== Unified v8->v9 run plan ===")
    print(f"phase1 (v8): {args.v8_script}, steps={args.phase1_steps}, skip={args.skip_phase1}")
    print(f"phase2 (v9): {args.v9_script}, steps={args.phase2_steps}")
    print(f"handover: {v8_last} -> {v9_last}")

    if args.dry_run:
        print("[dry-run] no command executed")
        return

    if not args.skip_phase1:
        run_phase(args.v8_script, args.phase1_steps)

    copy_ckpt_for_handover(v8_last, v9_last)
    run_phase(args.v9_script, args.phase2_steps)

    print("✅ Unified pipeline finished")


if __name__ == "__main__":
    main()
