"""Unified launcher for the four distributed RB2D training entries."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = {
    "ifactformer": ROOT / "train_ddp.py",
    "fno": ROOT / "train_FNO.py",
    "afno": ROOT / "train_AFNO.py",
    "swin": ROOT / "train_Swin.py",
}
parser = argparse.ArgumentParser(description="Unified launcher for the four RB models.")
parser.add_argument("--model", choices=(*SCRIPTS, "all"), required=True)
parser.add_argument("--config", type=Path, default=None)
parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra arguments passed to the selected trainer.")
args = parser.parse_args()
selected = SCRIPTS if args.model == "all" else {args.model: SCRIPTS[args.model]}
for name, script in selected.items():
    cmd = [sys.executable, str(script)]
    if args.config is not None and len(selected) == 1:
        cmd += ["--config", str(args.config)]
    cmd += args.extra
    subprocess.run(cmd, check=True, cwd=ROOT)
