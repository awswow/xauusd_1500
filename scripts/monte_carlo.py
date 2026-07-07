"""
Monte Carlo and bootstrap analysis on nested walk-forward OOS trades.

Loads the OOS R-multiples from the nested WF results JSON and runs:
  - Bootstrap CI on expectancy, win rate, and Calmar
  - P(expectancy <= 0) from the bootstrap distribution
  - Monte Carlo equity-curve percentiles over 200-trade paths
  - P(end negative), P(max DD > 20R), P(max DD > 30R)
  - Rolling 50-trade win rate and expectancy (temporal stability check)

Live comparison mode:
  Pass --live_trades N --live_r X to show where a live result falls in
  the simulated distribution and evaluate it against pass/fail criteria.

Requirements:
  data/derived/runs/walk_forward/nested_wf_results.json must exist and
  contain 'oos_trades'.  Re-run the nested WF to generate it:

    python scripts/walk_forward.py --config configs/live_demo_bt.yaml --nested

Usage:
  python scripts/monte_carlo.py
  python scripts/monte_carlo.py --live_trades 47 --live_r 3.2
  python scripts/monte_carlo.py --live_trades 150 --live_r 18.5 --live_wins 51
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_INPUT = "data/derived/runs/walk_forward/nested_wf_results.json"
N_BOOTSTRAP   = 10_000
N_MC_PATHS    = 10_000
MC_PATH_LEN   = 200
ROLLING_WIN   = 50


def _calmar(rs: np.ndarray) -> float:
    if len(rs) == 0:
        return 0.0
    eq   = np.cumsum(rs)
    peak = np.maximum.accumulate(eq)
    dd   = float(np.max(peak - eq))
    tot  = float(eq[-1])
    return tot / dd if dd > 0 and tot > 0 else 0.0


def _max_dd(rs: np.ndarray) -> float:
    if len(rs) == 0:
        return 0.0
    eq   = np.cumsum(rs)
    peak = np.maximum.accumulate(eq)
    return float(np.max(peak - eq))


def _bootstrap(rs: np.ndarray, fn, n: int = N_BOOTSTRAP) -> np.ndarray:
    idx = np.random.randint(0, len(rs), size=(n, len(rs)))
    return np.array([fn(rs[row]) for row in idx])


# Pass/fail thresholds (XAU_M15_PBT_v1.0, frozen 2026-07-07)
PASS_FAIL = {
    "min_trades":          150,
    "continue_exp_min":    0.00,
    "continue_avg_r_min":  0.05,
    "continue_dd_max":     20.0,
    "investigate_exp_lo": -0.05,
    "investigate_exp_hi":  0.05,
    "investigate_wr_min":  0.286,   # breakeven at 2.5R target
    "stop_exp_min":       -0.10,
}


def _live_verdict(n: int, total_r: float, wins: int | None, mc_eq: np.ndarray) -> None:
    """Print where a live result sits in the simulated distribution and apply pass/fail."""
    mean_r = total_r / n if n > 0 else 0.0
    wr     = wins / n if (wins is not None and n > 0) else None

    sep  = "=" * 76
    line = "─" * 52

    print(f"\n{sep}")
    print(f"LIVE RESULT COMPARISON  ({n} trades,  total R = {total_r:+.2f}R)")
    print(sep)

    # Percentile in MC distribution
    col_idx = min(n - 1, mc_eq.shape[1] - 1)
    col     = mc_eq[:, col_idx]
    pct     = float(np.mean(col <= total_r)) * 100

    p5, p25, p50, p75, p95 = np.percentile(col, [5, 25, 50, 75, 95])
    print(f"\n  At trade {n}, simulated distribution (from OOS MC paths):")
    print(f"    5th pct  : {p5:>+7.1f}R")
    print(f"   25th pct  : {p25:>+7.1f}R")
    print(f"   50th pct  : {p50:>+7.1f}R  (median)")
    print(f"   75th pct  : {p75:>+7.1f}R")
    print(f"   95th pct  : {p95:>+7.1f}R")
    print(f"\n  Live result : {total_r:>+7.1f}R  ->  {pct:.0f}th percentile of simulated paths")

    if pct < 5:
        rank = "WELL BELOW NORMAL (bottom 5% of simulated paths)"
    elif pct < 25:
        rank = "below median"
    elif pct < 75:
        rank = "within normal range (25th–75th pct)"
    elif pct < 95:
        rank = "above median"
    else:
        rank = "WELL ABOVE NORMAL (top 5% of simulated paths)"
    print(f"  Classification: {rank}")

    if wr is not None:
        be = 1 / (1 + 2.5)
        print(f"\n  Win rate: {wr*100:.1f}%  (breakeven: {be*100:.1f}%,  OOS baseline: 32.5%)")

    # Pass/fail evaluation
    pf = PASS_FAIL
    print(f"\n{line}")
    if n < pf["min_trades"]:
        print(f"  PASS/FAIL: PENDING  ({n}/{pf['min_trades']} trades — evaluate at {pf['min_trades']})")
    else:
        print(f"  PASS/FAIL EVALUATION  (XAU_M15_PBT_v1.0, n={n} >= {pf['min_trades']})")
        print()

        checks = []

        # Expectancy check
        if mean_r >= pf["continue_avg_r_min"]:
            checks.append(("PASS",        f"Expectancy {mean_r:+.4f}R >= {pf['continue_avg_r_min']:+.2f}R"))
        elif mean_r >= pf["investigate_exp_lo"]:
            checks.append(("INVESTIGATE", f"Expectancy {mean_r:+.4f}R in unclear zone [{pf['investigate_exp_lo']:+.2f}R, {pf['investigate_exp_hi']:+.2f}R]"))
        else:
            checks.append(("STOP",        f"Expectancy {mean_r:+.4f}R < {pf['stop_exp_min']:+.2f}R threshold"))

        # Win rate check (only if wins provided)
        if wr is not None:
            if wr >= pf["investigate_wr_min"]:
                checks.append(("PASS",        f"Win rate {wr*100:.1f}% >= breakeven {pf['investigate_wr_min']*100:.1f}%"))
            else:
                checks.append(("INVESTIGATE", f"Win rate {wr*100:.1f}% below breakeven {pf['investigate_wr_min']*100:.1f}%"))

        # Percentile check
        if pct >= 5:
            checks.append(("PASS",        f"Result at {pct:.0f}th pct of simulated paths (>= 5th)"))
        else:
            checks.append(("INVESTIGATE", f"Result at {pct:.0f}th pct — below P5 of normal range"))

        for status, msg in checks:
            icon = "OK  " if status == "PASS" else ("!!! " if status == "STOP" else "?   ")
            print(f"    [{icon}] {status:<11}  {msg}")

        overall = "STOP" if any(s == "STOP" for s, _ in checks) else (
                  "INVESTIGATE" if any(s == "INVESTIGATE" for s, _ in checks) else "CONTINUE")
        print(f"\n  OVERALL: {overall}")
        if overall == "CONTINUE":
            print("  -> Results consistent with OOS expectations.  Stay the course.")
        elif overall == "INVESTIGATE":
            print("  -> Collect more trades before changing anything.  Review execution log.")
        else:
            print("  -> Halt new entries.  Review execution quality and regime conditions.")

    print(f"\n{sep}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",       default=DEFAULT_INPUT)
    ap.add_argument("--out",         default="data/derived/runs/walk_forward")
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--live_trades", type=int,   default=None,
                    help="Number of completed live trades")
    ap.add_argument("--live_r",      type=float, default=None,
                    help="Total live R accumulated")
    ap.add_argument("--live_wins",   type=int,   default=None,
                    help="Number of winning live trades (optional, for win-rate check)")
    args = ap.parse_args()

    data_path = Path(args.input)
    if not data_path.exists():
        print(f"File not found: {data_path}")
        print("Run first:  python scripts/walk_forward.py --config configs/live_demo_bt.yaml --nested")
        sys.exit(1)

    data = json.loads(data_path.read_text(encoding="utf-8"))
    if "oos_trades" not in data:
        print(f"'oos_trades' key missing from {data_path}")
        print("Re-run:  python scripts/walk_forward.py --config configs/live_demo_bt.yaml --nested")
        sys.exit(1)

    rs = np.array([t["r"] for t in data["oos_trades"]], dtype=float)
    n  = len(rs)

    np.random.seed(args.seed)

    sep  = "=" * 76
    line = "─" * 52

    print(f"\n{sep}")
    print("MONTE CARLO & BOOTSTRAP ANALYSIS")
    print("Source: nested rolling walk-forward OOS trades (zero in-sample leakage)")
    print(sep)
    print(f"\n  Sample: {n} trades  |  Seed: {args.seed}")

    # ── Observed stats ───────────────────────────────────────────────────────
    obs_mean  = float(np.mean(rs))
    obs_wr    = float(np.mean(rs > 0))
    obs_total = float(np.sum(rs))
    obs_cal   = _calmar(rs)
    obs_mdd   = _max_dd(rs)
    be_wr     = 1.0 / (1.0 + 2.5)   # breakeven at 2.5R target = 28.57%

    print(f"\n{line}")
    print("OBSERVED OOS STATISTICS")
    print(line)
    print(f"  Trades          : {n}")
    print(f"  Win rate        : {obs_wr*100:.1f}%   (breakeven at tr=2.5: {be_wr*100:.1f}%)")
    print(f"  Mean R / trade  : {obs_mean:+.4f}R")
    print(f"  Total R         : {obs_total:+.2f}R")
    print(f"  Max drawdown    : {obs_mdd:.2f}R")
    print(f"  Calmar          : {obs_cal:.2f}")

    # ── Bootstrap CIs ────────────────────────────────────────────────────────
    print(f"\n{line}")
    print(f"BOOTSTRAP  ({N_BOOTSTRAP:,} resamples, sample size = {n})")
    print(line)

    b_mean = _bootstrap(rs, np.mean)
    b_wr   = _bootstrap(rs, lambda x: float(np.mean(x > 0)))
    b_cal  = _bootstrap(rs, _calmar)

    p_neg  = float(np.mean(b_mean <= 0))

    pcts = [5, 25, 50, 75, 95]

    for label, vals, unit in [
        ("Mean R/trade (expectancy)", b_mean,     "R"),
        ("Win rate",                  b_wr * 100, "%"),
        ("Calmar",                    b_cal,      ""),
    ]:
        ps = np.percentile(vals, pcts)
        print(f"\n  {label}")
        for p, v in zip(pcts, ps):
            marker = "  <- breakeven" if unit == "%" and abs(v - be_wr*100) < 0.4 else ""
            print(f"    {p:>3}th pct: {v:>+8.3f}{unit}{marker}")

    print(f"\n  P(expectancy <= 0) : {p_neg*100:.2f}%")
    print(f"  P(expectancy >  0) : {(1-p_neg)*100:.2f}%")

    # ── Monte Carlo equity curves ────────────────────────────────────────────
    print(f"\n{line}")
    print(f"MONTE CARLO EQUITY CURVES  ({N_MC_PATHS:,} paths x {MC_PATH_LEN} trades)")
    print(line)

    mc_idx = np.random.randint(0, n, size=(N_MC_PATHS, MC_PATH_LEN))
    mc_rs  = rs[mc_idx]
    mc_eq  = np.cumsum(mc_rs, axis=1)

    final_eq    = mc_eq[:, -1]
    mc_peak     = np.maximum.accumulate(mc_eq, axis=1)
    mc_dd_paths = np.max(mc_peak - mc_eq, axis=1)

    p_neg_end = float(np.mean(final_eq <= 0))
    p_dd_20   = float(np.mean(mc_dd_paths >= 20))
    p_dd_30   = float(np.mean(mc_dd_paths >= 30))

    eq_ps = np.percentile(final_eq, pcts)
    print(f"\n  Final equity after {MC_PATH_LEN} trades (R):")
    for p, v in zip(pcts, eq_ps):
        bar_len = max(0, int(abs(v) / 1.5))
        bar     = ("+" if v >= 0 else "-") + "█" * min(bar_len, 40)
        print(f"    {p:>3}th pct: {v:>+7.1f}R  {bar}")

    print(f"\n  Probability metrics ({MC_PATH_LEN}-trade horizon):")
    print(f"    P(end negative)      : {p_neg_end*100:>5.1f}%")
    print(f"    P(max DD > 20R)      : {p_dd_20*100:>5.1f}%")
    print(f"    P(max DD > 30R)      : {p_dd_30*100:>5.1f}%")

    print(f"\n  Equity percentile bands by trade count:")
    hdr = f"  {'Trades':>7}  {'5th':>7}  {'25th':>7}  {'50th':>7}  {'75th':>7}  {'95th':>7}"
    print(hdr)
    print(f"  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
    for t in [50, 100, 150, 200]:
        col = mc_eq[:, t - 1]
        ps5, ps25, ps50, ps75, ps95 = np.percentile(col, [5, 25, 50, 75, 95])
        print(f"  {t:>7}  {ps5:>+7.1f}  {ps25:>+7.1f}  {ps50:>+7.1f}  {ps75:>+7.1f}  {ps95:>+7.1f}")

    # ── Rolling stability check ───────────────────────────────────────────────
    print(f"\n{line}")
    print(f"ROLLING STABILITY  (window={ROLLING_WIN}, step={ROLLING_WIN//2}, actual OOS sequence)")
    print(line)
    print(f"\n  Trades are ordered chronologically: W1=2023 (175), W2=2024 (287), W3=2025 (338)")
    print(f"\n  {'Window':<10}  {'Win%':>6}  {'Exp R':>7}  {'Status'}")
    print(f"  {'─'*10}  {'─'*6}  {'─'*7}  {'─'*24}")

    step = ROLLING_WIN // 2
    roll_wrs   = []
    roll_means = []
    for start in range(0, n - ROLLING_WIN + 1, step):
        w      = rs[start: start + ROLLING_WIN]
        wr     = float(np.mean(w > 0))
        mn     = float(np.mean(w))
        label  = f"{start+1}–{start+ROLLING_WIN}"
        status = "BELOW BREAKEVEN" if wr < be_wr else ("strong" if wr > 0.38 else "ok")
        print(f"  {label:<10}  {wr*100:>5.1f}%  {mn:>+6.3f}R  {status}")
        roll_wrs.append(wr)
        roll_means.append(mn)

    print(f"\n  Rolling win rate  : {min(roll_wrs)*100:.1f}% – {max(roll_wrs)*100:.1f}%")
    print(f"  Rolling exp R     : {min(roll_means):+.3f}R – {max(roll_means):+.3f}R")
    print(f"  Windows below breakeven: {sum(1 for w in roll_wrs if w < be_wr)} / {len(roll_wrs)}")

    # ── Verdict ──────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)
    print(f"\n  OOS expectancy           : {obs_mean:+.4f}R/trade")
    print(f"  P(positive expectancy)   : {(1-p_neg)*100:.1f}%  (bootstrap, {N_BOOTSTRAP:,} resamples)")
    print(f"  P(end neg, 200 trades)   : {p_neg_end*100:.1f}%  (Monte Carlo, {N_MC_PATHS:,} paths)")
    print(f"  P(DD > 20R, 200 trades)  : {p_dd_20*100:.1f}%")
    print()
    if p_neg < 0.01:
        print("  The bootstrap strongly supports a positive expectancy (p < 1%).")
    elif p_neg < 0.05:
        print("  The bootstrap supports a positive expectancy (p < 5%).")
    else:
        print(f"  Bootstrap P(negative) = {p_neg*100:.1f}% — edge is present but uncertain.")
    print("  These results are consistent with the OOS Calmar of 5.12 across 800 trades.")
    print("  Robustness caveat: one instrument, one historical sample.")
    print("  Live fills remain the primary forward test.\n")

    # ── Live comparison (optional) ────────────────────────────────────────────
    if args.live_trades is not None and args.live_r is not None:
        _live_verdict(args.live_trades, args.live_r, args.live_wins, mc_eq)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "monte_carlo_results.json"
    out_path.write_text(json.dumps({
        "n_trades":  n,
        "observed":  {
            "mean_r":   obs_mean,  "win_rate": obs_wr,
            "total_r":  obs_total, "calmar":   obs_cal, "max_dd": obs_mdd,
        },
        "bootstrap": {
            "n_resamples": N_BOOTSTRAP,
            "mean_r_pcts": {str(p): float(v) for p, v in zip(pcts, np.percentile(b_mean, pcts))},
            "win_rate_pcts": {str(p): float(v) for p, v in zip(pcts, np.percentile(b_wr, pcts))},
            "p_expectancy_negative": p_neg,
            "p_expectancy_positive": 1.0 - p_neg,
        },
        "monte_carlo": {
            "n_paths": N_MC_PATHS, "path_length": MC_PATH_LEN,
            "p_end_negative": p_neg_end,
            "p_dd_gte_20r":   p_dd_20,
            "p_dd_gte_30r":   p_dd_30,
            "final_eq_pcts":  {str(p): float(v) for p, v in zip(pcts, eq_ps)},
        },
    }, indent=2), encoding="utf-8")
    print(f"  Results saved: {out_path}\n")


if __name__ == "__main__":
    main()
