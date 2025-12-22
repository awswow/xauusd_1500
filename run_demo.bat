@echo off
cd C:\Users\bloom\Desktop\xauusd_100_ready
call venv\Scripts\activate
python scripts\demo_executor_mt5.py --config configs\live.yaml --poll_sec 10
