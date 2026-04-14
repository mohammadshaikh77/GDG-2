"""Master runner for launching experiment shell pipelines with configurable paths."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


SCRIPT_BY_EXP = {
    1: "run_exp1.sh",
    2: "run_exp2.sh",
    3: "run_exp3.sh",
    4: "run_exp4.sh",
    5: "run_exp5.sh",
    6: "run_exp6.sh",
}


def parse_args() -> argparse.Namespace:
    """Returns command-line arguments for experiment dispatch."""
    parser = argparse.ArgumentParser(description="Run an experiment pipeline script.")
    parser.add_argument("--exp", type=int, required=True, choices=sorted(SCRIPT_BY_EXP.keys()))
    parser.add_argument("--data_dir", type=str, default="./data/wikitext103")
    parser.add_argument("--results_root", type=str, default="./results")
    parser.add_argument("--plots_root", type=str, default="./plots")
    parser.add_argument("--source_root", type=str, default="./results", help="Used by exp5 as source for exp1 checkpoints.")
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    """Runs the requested experiment shell script with injected environment variables."""
    repo_root = Path(__file__).resolve().parent
    script_name = SCRIPT_BY_EXP[args.exp]
    script_path = repo_root / "scripts" / script_name

    out_dir = Path(args.results_root) / f"exp{args.exp}"
    plots_dir = Path(args.plots_root) / f"exp{args.exp}"

    env = os.environ.copy()
    env["DATA_DIR"] = args.data_dir
    env["OUT_DIR"] = str(out_dir)
    env["PLOTS_DIR"] = str(plots_dir)

    if args.exp == 5:
        env["SOURCE_DIR"] = str(Path(args.source_root) / "exp1")

    print(f"Running experiment {args.exp} via {script_path}")
    print(f"DATA_DIR={env['DATA_DIR']}")
    print(f"OUT_DIR={env['OUT_DIR']}")
    print(f"PLOTS_DIR={env['PLOTS_DIR']}")
    if args.exp == 5:
        print(f"SOURCE_DIR={env['SOURCE_DIR']}")

    subprocess.run(["bash", str(script_path)], cwd=str(repo_root), env=env, check=True)


if __name__ == "__main__":
    run_pipeline(parse_args())
