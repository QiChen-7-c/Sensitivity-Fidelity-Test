"""Run the same sensitivity program for one or all four RB models."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIGS = {name: ROOT / "configs" / f"{name}.yaml" for name in ("ifactformer", "fno", "afno", "swin")}

parser = argparse.ArgumentParser(description="Unified sensitivity launcher.")
parser.add_argument("--model", choices=(*CONFIGS, "all"), required=True)
parser.add_argument("--load-dir", type=Path, default=None)
parser.add_argument("--checkpoint", default="checkpoint_best.pt")
parser.add_argument("--device", default=None)
args = parser.parse_args()
models = CONFIGS if args.model == "all" else {args.model: CONFIGS[args.model]}
for name, config_path in models.items():
    # The sensitivity script reads the model config from the checkpoint. The standard
    # output location is configs/<model>.yaml's log_dir (outputs/<model>).
    load_dir = args.load_dir if args.load_dir is not None and len(models) == 1 else ROOT / "outputs" / name
    cmd = [sys.executable, str(ROOT / "sensitivity.py"), "--load-dir", str(load_dir), "--checkpoint", args.checkpoint, "--model-type", name, "--physical-io"]
    if args.device:
        cmd += ["--device", args.device]
    subprocess.run(cmd, check=True, cwd=ROOT)
