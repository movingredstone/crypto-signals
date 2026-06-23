#!/usr/bin/env python3
"""
Parallel fold evaluation launcher for local multi-core execution.

Runs baseline and stress evaluations across multiple intervals
as independent parallel processes, each using 9 workers internally.
Total: up to 36 concurrent worker processes (4 processes × 9 workers).

Usage:
    python run_parallel.py              # Full: baseline + stress, 1h + 4h
    python run_parallel.py --quick      # Quick: 500 experiments only
    python run_parallel.py --intervals 1h --mode baseline  # Single mode
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
MAIN_PY = PROJECT_DIR / "main.py"


def run_fold_eval(
    interval: str,
    mode: str,
    experiments: int,
    workers: int,
    seed: int,
    output_dir: str = "results/fold_eval",
) -> subprocess.Popen:
    """Launch a single fold-eval process."""
    cmd = [
        sys.executable, "-u", str(MAIN_PY), "fold-eval",
        "--symbol", "BTCUSDT",
        "--intervals", interval,
        "--experiments", str(experiments),
        "--workers", str(workers),
        "--seed", str(seed),
        "--mode", mode,
        "--output-dir", output_dir,
    ]
    
    # Ensure PYTHONUNBUFFERED for real-time progress
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(PROJECT_DIR),
    )
    return proc


def stream_output(proc: subprocess.Popen, label: str):
    """Stream process output to terminal with label prefix."""
    prefix = f"[{label}]"
    for line in proc.stdout:
        # Print progress lines only (skip empty lines)
        line = line.rstrip()
        if line:
            print(f"{prefix} {line}")


def monitor_processes(procs: dict[str, subprocess.Popen]):
    """Monitor all running processes, stream their output, wait for completion."""
    threads = []
    for label, proc in procs.items():
        t = threading.Thread(target=stream_output, args=(proc, label), daemon=True)
        t.start()
        threads.append(t)
    
    # Wait for all to complete
    exit_codes = {}
    for label, proc in procs.items():
        proc.wait()
        exit_codes[label] = proc.returncode
    
    # Let output threads catch up
    time.sleep(0.5)
    
    return exit_codes


def collect_results(output_dir: str) -> dict:
    """Collect and summarize all results from output directory."""
    out = Path(output_dir)
    summary = {
        "baseline": {},
        "stress": {},
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    
    for mode in ["baseline", "stress"]:
        json_files = sorted(
            out.glob(f"BTCUSDT_{mode}_*_summary.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for jf in json_files:
            try:
                with open(jf) as f:
                    data = json.load(f)
                intervals = data.get("intervals", [])
                for iv in intervals:
                    if iv not in summary[mode]:
                        summary[mode][iv] = data.get("survival_counts", {})
                    print(f"  {jf.name}: {iv} — {data.get('survival_counts', {})}")
            except Exception as e:
                print(f"  Error reading {jf.name}: {e}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Parallel fold evaluation launcher")
    parser.add_argument("--intervals", nargs="+", default=["1h", "4h"])
    parser.add_argument("--mode", choices=["baseline", "stress", "all"], default="all")
    parser.add_argument("--experiments", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/fold_eval")
    parser.add_argument("--quick", action="store_true", help="500 experiments, 1h only, baseline only")
    args = parser.parse_args()
    
    # --quick overrides defaults (but explicit flags take precedence)
    if args.quick:
        if args.intervals == parser.get_default("intervals"):
            args.intervals = ["1h", "4h"]
        if args.mode == parser.get_default("mode"):
            args.mode = "baseline"
        if args.experiments == parser.get_default("experiments"):
            args.experiments = 500
    
    modes = ["baseline", "stress"] if args.mode == "all" else [args.mode]
    
    print(f"{'='*70}")
    print(f"Parallel Fold Evaluation Launcher")
    print(f"{'='*70}")
    print(f"Intervals:  {', '.join(args.intervals)}")
    print(f"Modes:      {', '.join(modes)}")
    print(f"Experiments: {args.experiments}/interval")
    print(f"Workers:    {args.workers}/process")
    print(f"Total parallel processes: {len(args.intervals) * len(modes)}")
    print(f"Total worker threads: {len(args.intervals) * len(modes) * args.workers}")
    print(f"{'='*70}\n")
    
    # Launch all processes
    procs = {}
    seed_offset = 0
    for mode in modes:
        for interval in args.intervals:
            label = f"{mode[:4]}/{interval}"
            proc = run_fold_eval(
                interval=interval,
                mode=mode,
                experiments=args.experiments,
                workers=args.workers,
                seed=args.seed + seed_offset,
                output_dir=args.output_dir,
            )
            procs[label] = proc
            seed_offset += 1
            print(f"Started: {label} (PID {proc.pid})")
            # Stagger starts to avoid I/O contention
            time.sleep(1)
    
    print(f"\nRunning {len(procs)} processes...\n")
    
    start_time = time.time()
    exit_codes = monitor_processes(procs)
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"All processes completed in {elapsed:.0f}s")
    print(f"{'='*70}")
    
    for label, code in exit_codes.items():
        status = "✅" if code == 0 else f"❌ (exit {code})"
        print(f"  {label}: {status}")
    
    # Collect summary
    print(f"\nSurvival Summary:")
    collect_results(args.output_dir)
    
    print(f"\nDone. Full results in {args.output_dir}/")


if __name__ == "__main__":
    main()
