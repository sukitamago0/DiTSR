#!/usr/bin/env python3
"""Grid-search inference knobs on RealSR evaluator.

Design goals:
- Reuse evaluator behavior (which itself reuses training-script logic where possible).
- Be robust across evaluator CLI variants in this repo history.
- Avoid crashing on argument drift; degrade gracefully and print capability warnings.
"""

import argparse
import itertools
import json
import subprocess
from pathlib import Path


def detect_eval_cli_support(eval_script: str):
    """Detect optional flags supported by evaluator.

    This makes sweep robust when users accidentally run with an older evaluator
    that does not yet include LQ-init control flags.
    """
    out = subprocess.check_output(["python", eval_script, "--help"], text=True)
    return {
        "use_lq_init": "--use-lq-init" in out,
        "no_lq_init": "--no-lq-init" in out,
        "lq_init_strength": "--lq-init-strength" in out,
        "crop_size": "--crop-size" in out,
        "eval_size": "--eval-size" in out,
        "save_panels_dir": "--save-panels-dir" in out,
        "save_preds_dir": "--save-preds-dir" in out,
    }


def run_once(base_cmd, cfg, use_lq_init, lq_init_strength, out_json, cli_support):
    cmd = list(base_cmd)
    cmd += ["--cfg", str(cfg), "--out-json", str(out_json)]
    if use_lq_init:
        if cli_support["use_lq_init"]:
            cmd += ["--use-lq-init"]
        if cli_support["lq_init_strength"]:
            cmd += ["--lq-init-strength", str(lq_init_strength)]
    else:
        if cli_support["no_lq_init"]:
            cmd += ["--no-lq-init"]
        else:
            # Older evaluator cannot disable LQ-init; keep sweep runnable.
            print("⚠️ Evaluator does not support --no-lq-init; skipping this config.")
            return None
    subprocess.run(cmd, check=True)
    data = json.loads(Path(out_json).read_text())
    ow = data.get("overall_weighted", {})
    return {
        "cfg": cfg,
        "use_lq_init": use_lq_init,
        "lq_init_strength": lq_init_strength,
        "psnr_y": ow.get("psnr_y_pred", None),
        "ssim_y": ow.get("ssim_y_pred", None),
        "lpips": ow.get("lpips_pred", None),
        "delta_psnr_y": data.get("delta_pred_minus_bic", {}).get("psnr_y", None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train-script", default="experiments/train_4090_auto_v9_gan_edge_gtmask.py")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--crop-size", type=int, default=512)
    ap.add_argument("--crop-border", type=int, default=4)
    ap.add_argument("--save-max-per-dataset", type=int, default=0, help="Set 0 to skip saving previews in sweep")
    ap.add_argument("--use-ema", action="store_true")
    ap.add_argument("--cfg-list", default="1.0,2.0,3.0")
    # IMPORTANT: in current training logic, larger strength -> closer to LR input (more conservative).
    ap.add_argument("--init-list", default="off,0.7,0.85,0.95")
    ap.add_argument("--out-dir", default="experiments_results/realsr_eval_sweep")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfgs = [float(x.strip()) for x in args.cfg_list.split(",") if x.strip()]
    inits = []
    for t in [x.strip().lower() for x in args.init_list.split(",") if x.strip()]:
        if t == "off":
            inits.append((False, 0.0))
        else:
            inits.append((True, float(t)))

    eval_script = "experiments/eval_ditsr_realsr_pairs.py"
    cli_support = detect_eval_cli_support(eval_script)
    print(f"CLI support detected: {cli_support}")

    base_cmd = [
        "python", eval_script,
        "--train-script", args.train_script,
        "--ckpt", args.ckpt,
        "--steps", str(args.steps),
        "--crop-border", str(args.crop_border),
        "--save-max-per-dataset", str(args.save_max_per_dataset),
    ]
    if cli_support["crop_size"]:
        base_cmd += ["--crop-size", str(args.crop_size)]
    elif cli_support["eval_size"]:
        base_cmd += ["--eval-size", str(args.crop_size)]

    if cli_support["save_panels_dir"]:
        base_cmd += ["--save-panels-dir", str(out_dir / "panels")]
    elif cli_support["save_preds_dir"]:
        base_cmd += ["--save-preds-dir", str(out_dir / "preds")]

    if args.use_ema:
        base_cmd.append("--use-ema")

    if not (cli_support["use_lq_init"] and cli_support["no_lq_init"] and cli_support["lq_init_strength"]):
        print("⚠️ Evaluator does not fully expose LQ-init CLI knobs; sweep will effectively focus on cfg.")

    rows = []
    for i, (cfg, (use_lq_init, lq_init_strength)) in enumerate(itertools.product(cfgs, inits)):
        out_json = out_dir / f"run_{i:02d}.json"
        row = run_once(base_cmd, cfg, use_lq_init, lq_init_strength, out_json, cli_support)
        if row is None:
            continue
        row["json"] = str(out_json)
        rows.append(row)

    rows.sort(key=lambda r: (-(r["psnr_y"] or -1e9), r["lpips"] if r["lpips"] is not None else 1e9))
    summary = {"results": rows, "best": rows[0] if rows else None}
    (out_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["best"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
