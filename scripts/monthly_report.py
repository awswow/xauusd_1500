"""
Monthly P&L breakdown from existing ledger CSV files.

Reads all sa160_YYYY/ledger.csv files and produces a month-by-month
breakdown showing: trades, win rate, total R, worst day, and whether
the monthly brake fired.

Usage:
  python scripts/monthly_report.py
  python scripts/monthly_report.py --pattern "sa160_*"
  python scripts/monthly_report.py --pattern "sa120_*"
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="data/derived/runs")
    ap.add_argument("--pattern",  default="sa160_*",
                    help="Glob pattern to match run dirs (e.g. 'sa160_*')")
    ap.add_argument("--monthly_limit_r", type=float, default=5.0)
    args = ap.parse_args()

    base = Path(args.runs_dir)
    ledger_files = sorted(base.glob(f"{args.pattern}/ledger.csv"))

    if not ledger_files:
        print(f"No ledger files found under {base}/{args.pattern}/ledger.csv")
        return

    print(f"\nMonthly P&L Report — pattern: {args.pattern}")
    print(f"Brake threshold: -{args.monthly_limit_r:.1f}R\n")

    # month_key -> list of r_multiple floats
    monthly: dict[str, list[float]] = defaultdict(list)
    daily:   dict[str, list[float]] = defaultdict(list)

    for lf in ledger_files:
        with lf.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    entry_time = row.get("entry_time_utc", "")
                    r = float(row.get("r_multiple", 0.0))
                    if not entry_time:
                        continue
                    mk = entry_time[:7]   # "YYYY-MM"
                    dk = entry_time[:10]  # "YYYY-MM-DD"
                    monthly[mk].append(r)
                    daily[dk].append(r)
                except (ValueError, KeyError):
                    continue

    if not monthly:
        print("No trades found in any ledger file.")
        return

    sep = "=" * 84
    hdr = f"  {'Month':<8}  {'Trades':>6}  {'Win%':>5}  {'Total R':>8}  "
    hdr += f"{'Avg R':>6}  {'Worst Day':>10}  {'Brake?':>8}"
    print(sep)
    print(hdr)
    print(f"  {'-'*8}  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*10}  {'-'*8}")

    running_r     = 0.0
    total_trades  = 0
    monthly_brake_count = 0
    yearly_r: dict[str, float] = defaultdict(float)

    for mk in sorted(monthly.keys()):
        rs    = monthly[mk]
        n     = len(rs)
        wins  = sum(1 for r in rs if r > 0)
        total = sum(rs)

        # worst single day in this month
        month_days  = [dk for dk in daily if dk.startswith(mk)]
        worst_day   = min((sum(daily[dk]) for dk in month_days), default=0.0)

        # did the monthly brake fire? (any sequential sequence sums to <= -limit)
        brake_fired = total <= -args.monthly_limit_r

        running_r     += total
        total_trades  += n
        yearly_r[mk[:4]] += total
        if brake_fired:
            monthly_brake_count += 1

        brake_str = "  FIRED" if brake_fired else ""
        print(f"  {mk:<8}  {n:>6}  {wins/n*100:>4.1f}%  {total:>+8.2f}  "
              f"{total/n:>+5.2f}  {worst_day:>+10.2f}  {brake_str:<8}")

        # Print year separator
        if mk[5:] == "12" or mk == sorted(monthly.keys())[-1]:
            yr = mk[:4]
            print(f"  {'-'*8}  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*10}  {'-'*8}")
            print(f"  {yr+' TOT':<8}  {'':>6}  {'':>5}  {yearly_r[yr]:>+8.2f}")
            print()

    print(sep)
    all_r = [r for rs in monthly.values() for r in rs]
    wins  = sum(1 for r in all_r if r > 0)
    print(f"  {'TOTAL':<8}  {total_trades:>6}  {wins/total_trades*100:>4.1f}%  "
          f"{running_r:>+8.2f}  {running_r/total_trades:>+6.2f}  "
          f"{'':>10}  {monthly_brake_count} brakes")
    print(sep)

    print(f"\n  Monthly brake fired {monthly_brake_count}x "
          f"across {len(monthly)} months "
          f"({monthly_brake_count/len(monthly)*100:.0f}% of months affected)")

    # Identify best and worst months
    best_3  = sorted(monthly.items(), key=lambda x: sum(x[1]), reverse=True)[:3]
    worst_3 = sorted(monthly.items(), key=lambda x: sum(x[1]))[:3]
    print(f"\n  Best months:  " + "  |  ".join(f"{k}: {sum(v):+.2f}R" for k, v in best_3))
    print(f"  Worst months: " + "  |  ".join(f"{k}: {sum(v):+.2f}R" for k, v in worst_3))

    # Seasonal pattern (average by calendar month)
    print(f"\n  Seasonal average (across all years):")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    seasonal: dict[str, list[float]] = defaultdict(list)
    for mk, rs in monthly.items():
        seasonal[mk[5:]].append(sum(rs))   # "01" through "12"

    season_line = "  "
    for mo in ["01","02","03","04","05","06","07","08","09","10","11","12"]:
        vals = seasonal.get(mo, [])
        avg  = sum(vals) / len(vals) if vals else 0.0
        name = month_names[int(mo) - 1]
        season_line += f"{name}:{avg:+.1f}R  "
    print(season_line)
    print()


if __name__ == "__main__":
    main()
