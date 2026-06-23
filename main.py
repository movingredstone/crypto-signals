import argparse
import pandas as pd
from src.reports import format_summary


def main():
    parser = argparse.ArgumentParser(description="investmentsystem controller")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("telegram", help="Run Telegram bot")

    bt = sub.add_parser("backtest", help="Run backtest")
    bt.add_argument("--symbol", default="BTCUSDT")
    bt.add_argument("--interval", default="15m")

    rs = sub.add_parser("research", help="Run factor-combination research")
    rs.add_argument("--symbol", default="BTCUSDT")
    rs.add_argument("--interval", default="1h")
    rs.add_argument("--max-experiments", type=int, default=300)
    rs.add_argument("--workers", type=int, default=4)

    opt = sub.add_parser("optimize", help="Run multi-stage BTC futures parameter optimization")
    opt.add_argument("--symbol", default="BTCUSDT")
    opt.add_argument("--intervals", nargs="+", default=["1h"])
    opt.add_argument("--experiments", type=int, default=5000)
    opt.add_argument("--workers", type=int, default=4)
    opt.add_argument("--stage", choices=["broad", "refine", "stress", "all"], default="broad")
    opt.add_argument("--top-n", type=int, default=50)
    opt.add_argument("--seed", type=int, default=42)
    opt.add_argument("--wf-folds", type=int, default=4)
    opt.add_argument("--source-top-path", default=None)
    opt.add_argument("--refine-from-top", type=int, default=20)
    opt.add_argument("--output-dir", default="results/optimization")

    fe = sub.add_parser("fold-eval", help="Evaluate strategies across 7 half-year folds")
    fe.add_argument("--symbol", default="BTCUSDT")
    fe.add_argument("--intervals", nargs="+", default=["15m", "1h", "4h"])
    fe.add_argument("--experiments", type=int, default=5000)
    fe.add_argument("--workers", type=int, default=9)
    fe.add_argument("--top-n", type=int, default=50)
    fe.add_argument("--seed", type=int, default=42)
    fe.add_argument("--output-dir", default="results/fold_eval")
    fe.add_argument("--mode", choices=["baseline", "stress"], default="baseline")

    tv = sub.add_parser("tvt", help="TVT forward cross-validation on top candidates")
    tv.add_argument("--symbol", default="BTCUSDT")
    tv.add_argument("--source-csv", required=True, help="CSV with candidate params (from fold eval)")
    tv.add_argument("--top-n", type=int, default=10, help="How many top candidates to test")
    tv.add_argument("--output-dir", default="results/tvt")

    pp = sub.add_parser("paper", help="Run paper trading on top 3 strategies (forward 2026)")
    pp.add_argument("--mode", choices=["baseline", "optimized"], default="baseline",
                    help="baseline=all regimes+equal weight, optimized=regime filter+risk-parity")
    pp.add_argument("--end-date", default=None,
                    help="End date for paper trading window (YYYY-MM-DD). Default: today.")

    args = parser.parse_args()

    if args.command == "telegram":
        from src.telegram_bot import run_telegram_bot
        run_telegram_bot(config_path="config.yaml")

    elif args.command == "backtest":
        from src.backtester import run_backtest

        summary = run_backtest(
            symbol=args.symbol.upper(),
            interval=args.interval,
            config_path="config.yaml",
        )

        print(format_summary(summary))

    elif args.command == "research":
        from src.research_engine import run_research, format_research_report

        report = run_research(
            symbol=args.symbol.upper(),
            interval=args.interval,
            max_experiments=args.max_experiments,
            config_path="config.yaml",
            max_workers=args.workers,
        )

        print(format_research_report(report))

    elif args.command == "optimize":
        from src.optimizer import run_optimization, format_optimization_report

        report = run_optimization(
            symbol=args.symbol.upper(),
            intervals=args.intervals,
            experiments=args.experiments,
            workers=args.workers,
            stage=args.stage,
            config_path="config.yaml",
            output_dir=args.output_dir,
            top_n=args.top_n,
            seed=args.seed,
            wf_folds=args.wf_folds,
            source_top_path=args.source_top_path,
            refine_from_top=args.refine_from_top,
        )

        print(format_optimization_report(report))

    elif args.command == "fold-eval":
        from src.fold_evaluator import evaluate_folds, format_fold_report

        result = evaluate_folds(
            symbol=args.symbol.upper(),
            intervals=args.intervals,
            experiments=args.experiments,
            workers=args.workers,
            top_n=args.top_n,
            seed=args.seed,
            config_path="config.yaml",
            output_dir=args.output_dir,
            mode=args.mode,
        )

        # Print compact summary
        df = result["df"]
        if not df.empty:
            print(format_fold_report(df, args.symbol.upper(), result["mode"], args.intervals, args.experiments))

    elif args.command == "tvt":
        import json
        from src.tvt_evaluator import run_tvt, format_tvt_report

        source = pd.read_csv(args.source_csv)
        # Sort by survival then mean return, take top N
        if "survival_pct" in source.columns and "mean_return" in source.columns:
            source = source.sort_values(
                ["survival_pct", "mean_return"], ascending=[False, False]
            )
        source = source.head(args.top_n)

        candidates = []
        for _, row in source.iterrows():
            try:
                params = json.loads(row["params_json"])
            except Exception:
                params = {}
            candidates.append(params)

        print(f"Loaded {len(candidates)} candidates from {args.source_csv}")
        result = run_tvt(
            candidates=candidates,
            symbol=args.symbol.upper(),
            config_path="config.yaml",
            output_dir=args.output_dir,
        )

        df = result["df"]
        if not df.empty:
            print(format_tvt_report(df, args.symbol.upper(), len(candidates)))

    elif args.command == "paper":
        from src.paper_trader import run_paper_trading, format_paper_report

        result = run_paper_trading(
            config_path="config.yaml",
            output_dir="results/paper",
            mode=getattr(args, "mode", "baseline"),
            end_date=args.end_date,
        )

        print()
        print(format_paper_report(result))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
