#!/usr/bin/env python3
import argparse
import subprocess
import sys

ADAPTER_TYPES = ["fpn", "fpn_se"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2, help="epochs per adapter (short smoke)")
    parser.add_argument("--smoke", action="store_true", help="use train_full_mse_adaln.py --smoke")
    parser.add_argument("--python", type=str, default=sys.executable, help="python executable")
    args = parser.parse_args()

    for adapter_type in ADAPTER_TYPES:
        cmd = [
            args.python,
            "experiments/train_full_mse_adaln.py",
            "--adapter_type",
            adapter_type,
        ]
        if args.smoke:
            cmd.append("--smoke")
        env = dict(**{**dict(os.environ)})
        env["EPOCHS_OVERRIDE"] = str(args.epochs)
        print(f"\n=== Running adapter_type={adapter_type} ===")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True, env=env)

if __name__ == "__main__":
    import os
    main()
