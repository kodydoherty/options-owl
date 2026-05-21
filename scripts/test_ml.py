from options_owl.risk.ml_exit import predict_sell
import sys

tests = [
    ('SPY winning', dict(ticker='SPY', entry_premium=1.50, current_premium=1.80, peak_premium=2.00, minutes_since_entry=15, now_hour=10, now_minute=30, is_call=True)),
    ('SPY losing', dict(ticker='SPY', entry_premium=1.50, current_premium=0.90, peak_premium=1.50, minutes_since_entry=30, now_hour=14, now_minute=0, is_call=True)),
    ('TSLA big win', dict(ticker='TSLA', entry_premium=3.00, current_premium=4.50, peak_premium=5.00, minutes_since_entry=45, now_hour=15, now_minute=0, is_call=True)),
    ('NVDA early', dict(ticker='NVDA', entry_premium=1.00, current_premium=1.10, peak_premium=1.15, minutes_since_entry=5, now_hour=9, now_minute=45, is_call=True)),
    ('Generic', dict(ticker='ZZZZ', entry_premium=1.50, current_premium=1.80, peak_premium=2.00, minutes_since_entry=15, now_hour=10, now_minute=30, is_call=True)),
]

all_ok = True
for name, kwargs in tests:
    r = predict_sell(**kwargs)
    status = 'OK' if r.model_used != 'none' else 'FAIL'
    if status == 'FAIL':
        all_ok = False
    print(f'{status} {name:15} sell={r.should_sell} prob={r.sell_probability:.2f} future={r.expected_future_pnl:.1f}% model={r.model_used} reason={r.reason}')

print()
if all_ok:
    print('ALL ML MODELS LOADING AND WORKING')
else:
    print('SOME MODELS FAILED')
    sys.exit(1)
