import argparse
import sys
from pathlib import Path

# Allow running without "pip install -e ."
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from xauusd100.engine.live_runner import LiveApp


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    args = p.parse_args()

    LiveApp(cfg_path=args.config).run()


if __name__ == "__main__":
    main()
