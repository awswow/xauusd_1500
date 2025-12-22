# scripts/analyze_ledger_regime.py
from __future__ import annotations

import argparse
import pandas as pd


def max_drawdown_r(r: pd.Series) -> float:
    if r.empty:
        return 0.0
    eq = r.fillna(0.0).cumsum()
    peak = eq.cummax()
    dd = eq - peak
    return float(dd.min())


def bucket_summary(df: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    g = df.groupby(bucket_col, dropna=False)
    out = pd.DataFrame({
        "trades": g.size(),
        "win_rate": g.apply(lambda x: float((x["r_multiple"] > 0).mean())),
        "avg_r": g["r_multiple"].mean(),
        "median_r": g["r_multiple"].median(),
        "total_r": g["r_multiple"].sum(),
        "max_dd_r": g.apply(lambda x: max_drawdown_r(x["r_multiple"])),
    }).sort_index()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.ledger)

    # Ensure numeric types
    for c in ["adx", "atr", "sma_slope", "sma_slope_atr", "r_multiple"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filter rows that have regime metrics
    have = df.dropna(subset=["r_multiple", "adx", "atr", "sma_slope"], how="any").copy()
    if have.empty:
        raise SystemExit("No usable rows with r_multiple + (adx, atr, sma_slope). Did you patch logging?")

    print("\n=== OVERALL ===")
    print(have["r_multiple"].describe())
    print("max_dd_r:", max_drawdown_r(have["r_multiple"]))

    # --- ADX quartiles ---
    have["adx_q"] = pd.qcut(have["adx"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    print("\n=== ADX quartiles ===")
    print(bucket_summary(have, "adx_q"))

    # --- ATR quartiles (compression detection proxy) ---
    have["atr_q"] = pd.qcut(have["atr"], 4, labels=["Q1_lowATR", "Q2", "Q3", "Q4_highATR"])
    print("\n=== ATR quartiles ===")
    print(bucket_summary(have, "atr_q"))

    # --- SMA slope sign + magnitude bins ---
    # use slope_atr if available (better normalization)
    if have["sma_slope_atr"].notna().any():
        slope = have["sma_slope_atr"]
        col = "sma_slope_atr"
        # bins: negative, small positive, medium, large
        have["slope_bin"] = pd.cut(
            slope,
            bins=[-10, 0, 0.02, 0.05, 10],
            labels=["neg_or_flat", "pos_small", "pos_med", "pos_large"],
            include_lowest=True,
        )
        print("\n=== SMA slope / ATR bins (normalized) ===")
        print(bucket_summary(have, "slope_bin"))
    else:
        slope = have["sma_slope"]
        have["slope_sign"] = pd.cut(
            slope,
            bins=[-1e9, 0, 1e9],
            labels=["neg_or_flat", "positive"],
            include_lowest=True,
        )
        print("\n=== SMA slope sign ===")
        print(bucket_summary(have, "slope_sign"))


if __name__ == "__main__":
    main()
