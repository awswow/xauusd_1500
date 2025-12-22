import argparse
import json
import sys
from pathlib import Path

# Allow running without "pip install -e ."
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from xauusd100.mt5.connector import MT5Connector, MT5Config
from xauusd100.mt5.symbols import get_symbol_spec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--out", default="data/derived/symbol_info.json")
    args = p.parse_args()

    MT5Connector(MT5Config()).connect()
    spec = get_symbol_spec(args.symbol)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec.__dict__, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
