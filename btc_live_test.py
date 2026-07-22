#!/usr/bin/env python3
"""
BTC/USDT LIVE DATA TEST - NO STOP LOSS
Momentum-Based Entry Strategy
Entry: When momentum spike > 10% upward direction
Only opens when momentum shows upward direction
"""

import json
import numpy as np
import pandas as pd
import ccxt
from datetime import datetime, timedelta, timezone
import sys
from typing import Dict, List, Tuple, Optional

STARTING_EQUITY = 100.0
COMPOUND_PCT = 0.5
MAX_TRADES = 1000
TAKER_FEE = 0.0005

# Momentum settings
MOMENTUM_WINDOW = 5  # bars to calculate momentum
MOMENTUM_THRESHOLD = 0.10  # 10% upward momentum required to enter

EXCHANGES_CONFIG = {
    "coinbase": {"enableRateLimit": True},
    "toobit": {"enableRateLimit": True},
}

def load_data(source: str, days: int) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data from exchange."""
    try:
        ExchangeClass = getattr(ccxt, source)
        exchange = ExchangeClass(EXCHANGES_CONFIG.get(source, {"enableRateLimit": True}))
        
        symbol = "BTC/USDT"
        timeframe = "1m"
        
        since = exchange.parse8601(
            (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        
        allbars = []
        fetch_count = 0
        
        while True:
            bars = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=1000,
            )
            
            if not bars:
                break
            
            allbars.extend(bars)
            fetch_count += 1
            since = bars[-1][0] + 60000
            
            if len(bars) < 1000:
                break
        
        df = pd.DataFrame(
            allbars,
            columns=["ts", "open", "high", "low", "close", "volume"],
        )
        
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.dropna(subset=["open", "high", "low", "close", "volume", "ts"]).reset_index(drop=True)
        
        return df
    
    except Exception as e:
        print(f"  X Error loading from {source}: {str(e)}", file=sys.stderr)
        return None


def calculate_momentum(df: pd.DataFrame, idx: int, window: int) -> float:
    """Calculate momentum as % change over window bars."""
    if idx < window:
        return 0.0
    
    past_price = float(df["close"].iloc[idx - window])
    current_price = float(df["close"].iloc[idx])
    
    if past_price == 0:
        return 0.0
    
    momentum = ((current_price - past_price) / past_price)
    return momentum


def has_valid_entry(df: pd.DataFrame, start_idx: int) -> Tuple[bool, float]:
    """
    Check if current bar has valid momentum for entry.
    Entry conditions:
    - Momentum > 10% upward
    - Only enter on upward momentum
    Returns: (is_valid, momentum_pct)
    """
    momentum = calculate_momentum(df, start_idx, MOMENTUM_WINDOW)
    
    # Only enter if momentum is positive AND above threshold
    is_valid = momentum > MOMENTUM_THRESHOLD
    
    return is_valid, momentum


def simulate_trade(df: pd.DataFrame, start_idx: int, fee_side: float, equity: float, notional: float, tp_pct: float) -> Tuple[Optional[Dict], Optional[float], Optional[int], float]:
    """Simulate a single trade with TP only (NO STOP LOSS). Requires momentum entry signal."""
    
    # Check momentum entry condition
    is_valid_entry, entry_momentum = has_valid_entry(df, start_idx)
    
    if not is_valid_entry:
        # No valid entry signal
        return None, None, None, entry_momentum
    
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    ts = df["ts"].to_numpy(dtype="datetime64[ns]")
    close = df["close"].to_numpy()
    
    entry_idx = start_idx + 1
    if entry_idx >= len(df):
        return None, None, None, entry_momentum
    
    entry = float(df["close"].iloc[entry_idx])
    tp_price = entry * (1.0 + tp_pct)
    qty = max(0.001, (equity * notional) / max(entry, 1e-9))
    
    future_high = high[entry_idx + 1:]
    
    tp_hits = np.where(future_high >= tp_price)[0]
    tp_i = tp_hits[0] if len(tp_hits) else None
    
    if tp_i is None:
        # Mark-to-market at end (no TP hit)
        last_idx = len(df) - 1
        last_px = float(df["close"].iloc[last_idx])
        gross = (last_px - entry) * qty
        fees = (entry * qty + last_px * qty) * fee_side
        pnl = gross - fees
        trade = {
            "side": "long",
            "opened": str(pd.Timestamp(ts[start_idx])),
            "entry": round(entry, 2),
            "exit": round(last_px, 2),
            "reason": "no_tp_hit",
            "qty": round(qty, 5),
            "gross": round(gross, 4),
            "fees": round(fees, 4),
            "pnl": round(pnl, 4),
            "exit_ts": str(pd.Timestamp(ts[last_idx])),
            "tp_hit": False,
            "hold_bars": last_idx - entry_idx,
            "leverage": notional,
            "entry_momentum": round(entry_momentum * 100, 2),
        }
        return trade, pnl, last_idx, entry_momentum
    
    # TP hit
    exit_idx = entry_idx + 1 + int(tp_i)
    exit_px = tp_price
    
    gross = (exit_px - entry) * qty
    fees = (entry * qty + exit_px * qty) * fee_side
    pnl = gross - fees
    
    trade = {
        "side": "long",
        "opened": str(pd.Timestamp(ts[start_idx])),
        "entry": round(entry, 2),
        "exit": round(exit_px, 2),
        "reason": "tp_hit",
        "qty": round(qty, 5),
        "gross": round(gross, 4),
        "fees": round(fees, 4),
        "pnl": round(pnl, 4),
        "exit_ts": str(pd.Timestamp(ts[exit_idx])),
        "tp_hit": True,
        "hold_bars": int(tp_i),
        "leverage": notional,
        "entry_momentum": round(entry_momentum * 100, 2),
    }
    return trade, pnl, exit_idx, entry_momentum


def run_backtest(df: pd.DataFrame, source: str, tp_pct: float, notional: float) -> Dict:
    """Run backtest on data with momentum entry filter."""
    equity = STARTING_EQUITY
    trades = []
    skipped_entries = 0
    cursor = 0
    completed = 0
    peak_equity = STARTING_EQUITY
    max_drawdown = 0
    
    while cursor < len(df) - 2 and completed < MAX_TRADES:
        trade, pnl, exit_idx, momentum = simulate_trade(df, cursor, TAKER_FEE, equity, notional, tp_pct)
        
        if trade is None:
            # Check if we skipped due to no momentum signal
            if momentum <= MOMENTUM_THRESHOLD:
                skipped_entries += 1
            cursor += 1
            continue
        
        trades.append(trade)
        completed += 1
        if pnl > 0:
            equity += pnl * COMPOUND_PCT
        else:
            equity += pnl
        
        # Track drawdown
        if equity > peak_equity:
            peak_equity = equity
        current_dd = ((peak_equity - equity) / peak_equity) * 100
        if current_dd > max_drawdown:
            max_drawdown = current_dd
        
        if trade["reason"] == "no_tp_hit":
            break
        cursor = exit_idx + 1
    
    tp_hits = sum(1 for t in trades if t["tp_hit"])
    no_tp_hits = sum(1 for t in trades if not t["tp_hit"])
    
    total_pnl = sum(t["pnl"] for t in trades)
    total_fees = sum(t["fees"] for t in trades)
    winning_trades = sum(1 for t in trades if t["pnl"] > 0)
    losing_trades = sum(1 for t in trades if t["pnl"] < 0)
    
    win_rate = (winning_trades / len(trades) * 100) if trades else 0
    avg_win = np.mean([t["pnl"] for t in trades if t["pnl"] > 0]) if winning_trades else 0
    avg_loss = np.mean([t["pnl"] for t in trades if t["pnl"] < 0]) if losing_trades else 0
    profit_factor = abs(sum(t["pnl"] for t in trades if t["pnl"] > 0) / sum(t["pnl"] for t in trades if t["pnl"] < 0)) if losing_trades else float('inf')
    
    avg_hold_bars = np.mean([t["hold_bars"] for t in trades]) if trades else 0
    avg_entry_momentum = np.mean([t["entry_momentum"] for t in trades]) if trades else 0
    
    return {
        "source": source,
        "starting_equity": STARTING_EQUITY,
        "ending_equity": round(equity, 4),
        "total_pnl": round(total_pnl, 4),
        "return_pct": round((equity - STARTING_EQUITY) / STARTING_EQUITY * 100, 2),
        "trade_count": len(trades),
        "skipped_entries": skipped_entries,
        "tp_hits": tp_hits,
        "no_tp_hits": no_tp_hits,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate_pct": round(win_rate, 2),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "inf",
        "avg_hold_bars": round(avg_hold_bars, 1),
        "total_fees": round(total_fees, 4),
        "max_drawdown_pct": round(max_drawdown, 2),
        "peak_equity": round(peak_equity, 4),
        "avg_entry_momentum": round(avg_entry_momentum, 2),
        "compound_pct_on_wins": COMPOUND_PCT,
        "tp_pct": tp_pct,
        "notional": notional,
        "momentum_threshold": MOMENTUM_THRESHOLD,
        "momentum_window": MOMENTUM_WINDOW,
        "data_points": len(df),
        "timespan": f"{df['ts'].min()} to {df['ts'].max()}",
        "trades": trades,
    }


def generate_report(results: Dict, tp_pct: float, notional: float) -> str:
    """Generate comprehensive live data test report with momentum analysis."""
    report = []
    report.append("=" * 100)
    report.append("BTC/USDT LIVE DATA TEST - MOMENTUM-BASED ENTRY")
    report.append(f"Starting Equity: $100 | {notional}x Leverage | {tp_pct*100:.2f}% TP | NO STOP LOSS")
    report.append(f"Entry: Momentum > 10% Upward | 5-Bar Window")
    report.append("=" * 100)
    report.append("")
    report.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    report.append("")
    
    report.append("STRATEGY PARAMETERS:")
    report.append(f"  * Starting Equity: ${STARTING_EQUITY}")
    report.append(f"  * Notional Multiplier: {notional}x LEVERAGE")
    report.append(f"  * Take-Profit: {tp_pct*100:.2f}%")
    report.append(f"  * Stop-Loss: NONE (momentum hold)")
    report.append(f"  * Compound on Wins: {COMPOUND_PCT*100:.0f}%")
    report.append(f"  * Taker Fee: {TAKER_FEE*100:.04f}%")
    report.append(f"  * Max Trades: {MAX_TRADES}")
    report.append("")
    
    report.append("MOMENTUM ENTRY FILTER:")
    report.append(f"  * Momentum Window: {MOMENTUM_WINDOW} bars")
    report.append(f"  * Entry Threshold: > {MOMENTUM_THRESHOLD*100:.1f}% upward momentum")
    report.append(f"  * Only enter on UPWARD momentum spikes")
    report.append(f"  * Skip entries during downtrends or flat momentum")
    report.append("")
    
    # Summary table
    report.append("RESULTS BY EXCHANGE:")
    report.append("-" * 100)
    report.append(f"{'Exchange':<12} {'Ending Equity':<15} {'Return %':<12} {'Trades':<8} {'Skipped':<10} {'Win %':<10} {'Avg Momentum':<15}")
    report.append("-" * 100)
    
    for exchange in ["coinbase", "toobit"]:
        if exchange in results:
            r = results[exchange]
            report.append(
                f"{exchange:<12} ${r['ending_equity']:<14.2f} {r['return_pct']:>11.2f}% {r['trade_count']:>7} "
                f"{r['skipped_entries']:>9} {r['win_rate_pct']:>9.2f}% {r['avg_entry_momentum']:>14.2f}%"
            )
    
    report.append("-" * 100)
    report.append("")
    
    # Detailed results
    for exchange in ["coinbase", "toobit"]:
        if exchange not in results:
            continue
        
        r = results[exchange]
        report.append(f"\n{exchange.upper()} DETAILED RESULTS:")
        report.append("=" * 100)
        report.append(f"Data Points: {r['data_points']:,}")
        report.append(f"Timespan: {r['timespan']}")
        report.append("")
        report.append("EQUITY SUMMARY:")
        report.append(f"  Starting Equity: ${r['starting_equity']:.2f}")
        report.append(f"  Ending Equity: ${r['ending_equity']:.2f}")
        report.append(f"  Peak Equity: ${r['peak_equity']:.2f}")
        report.append(f"  Total P&L: ${r['total_pnl']:.4f}")
        report.append(f"  Return: {r['return_pct']:.2f}%")
        report.append(f"  Max Drawdown: {r['max_drawdown_pct']:.2f}%")
        report.append("")
        report.append("ENTRY ANALYSIS:")
        report.append(f"  Total Bars Scanned: {r['data_points']:,}")
        report.append(f"  Valid Entry Signals (>10% momentum): {r['trade_count']}")
        report.append(f"  Skipped Entries (no momentum): {r['skipped_entries']}")
        report.append(f"  Average Entry Momentum: {r['avg_entry_momentum']:.2f}%")
        report.append("")
        report.append("TRADE STATISTICS:")
        report.append(f"  Total Trades Executed: {r['trade_count']}")
        report.append(f"    - TP Hits: {r['tp_hits']} ({r['tp_hits']/max(r['trade_count'], 1)*100:.1f}%)")
        report.append(f"    - No TP (MTM): {r['no_tp_hits']} ({r['no_tp_hits']/max(r['trade_count'], 1)*100:.1f}%)")
        report.append(f"  Winning Trades: {r['winning_trades']} | Losing Trades: {r['losing_trades']}")
        report.append(f"  Win Rate: {r['win_rate_pct']:.2f}%")
        report.append(f"  Avg Win: ${r['avg_win']:.4f} | Avg Loss: ${r['avg_loss']:.4f}")
        report.append(f"  Profit Factor: {r['profit_factor']}")
        report.append(f"  Avg Hold Time: {r['avg_hold_bars']:.1f} bars (1m candles)")
        report.append(f"  Total Fees Paid: ${r['total_fees']:.4f}")
        report.append("")
    
    # Risk/Reward Analysis
    report.append("\n" + "=" * 100)
    report.append("RISK/REWARD ANALYSIS:")
    report.append("=" * 100)
    
    for exchange in ["coinbase", "toobit"]:
        if exchange not in results:
            continue
        
        r = results[exchange]
        if r['avg_loss'] != 0:
            rr_ratio = abs(r['avg_win'] / r['avg_loss'])
        else:
            rr_ratio = 0
        
        report.append(f"\n{exchange.upper()}:")
        report.append(f"  Risk/Reward Ratio: {rr_ratio:.2f}:1")
        report.append(f"  Max Drawdown: {r['max_drawdown_pct']:.2f}%")
        
        if r['return_pct'] > 0:
            report.append(f"  Status: PROFITABLE (${r['total_pnl']:.4f})")
        elif r['return_pct'] == 0:
            report.append(f"  Status: BREAK EVEN")
        else:
            report.append(f"  Status: LOSS (${r['total_pnl']:.4f})")
        
        if r['max_drawdown_pct'] > 0:
            sharpe_like = r['return_pct'] / max(r['max_drawdown_pct'], 0.01)
            report.append(f"  Sharpe-like metric: {sharpe_like:.2f}")
    
    report.append("\n" + "=" * 100)
    report.append("CLAUDE CO-INVESTMENT READINESS:")
    report.append("=" * 100)
    
    ready = False
    for exchange in ["coinbase", "toobit"]:
        if exchange in results:
            r = results[exchange]
            if r['return_pct'] > 0 and r['win_rate_pct'] > 40 and r['trade_count'] > 10:
                ready = True
                report.append(f"\nRECOMMENDATION: {exchange.upper()} IS READY")
                report.append(f"  - Positive return: {r['return_pct']:+.2f}%")
                report.append(f"  - Win rate: {r['win_rate_pct']:.2f}%")
                report.append(f"  - Trade count: {r['trade_count']} momentum signals detected")
                report.append(f"  - Avg entry momentum: {r['avg_entry_momentum']:.2f}%")
                report.append(f"  - Risk controlled: {r['max_drawdown_pct']:.2f}% max DD")
                report.append(f"  - Deploy for live trading")
    
    if not ready:
        report.append("\nNOT READY FOR LIVE DEPLOYMENT")
        best_exch = None
        best_return = -999
        for exchange in ["coinbase", "toobit"]:
            if exchange in results and results[exchange]['return_pct'] > best_return:
                best_return = results[exchange]['return_pct']
                best_exch = exchange
        if best_exch:
            r = results[best_exch]
            report.append(f"\nBest so far: {best_exch.upper()}")
            report.append(f"  - Return: {r['return_pct']:+.2f}%")
            report.append(f"  - Momentum signals: {r['trade_count']}")
            report.append(f"  - Need: Positive return + >40% win rate + 10+ trades")
            if r['trade_count'] < 5:
                report.append(f"  - Try: Increase days OR reduce momentum threshold to {MOMENTUM_THRESHOLD*100*0.75:.1f}%")
    
    report.append("")
    report.append("=" * 100)
    
    return "\n".join(report)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--tp", type=float, default=0.02, help="Take-profit percentage (e.g., 0.005 for 0.5%)")
    ap.add_argument("--notional", type=float, default=10.0, help="Leverage multiplier (e.g., 2 for 2x)")
    ap.add_argument("--out-report", default="btc_live_test_report.txt")
    ap.add_argument("--out-json", default="btc_live_test_results.json")
    args = ap.parse_args()
    
    results = {}
    
    print("\n" + "="*100)
    print("BTC/USDT LIVE DATA TEST - MOMENTUM-BASED ENTRY")
    print(f"$100 Starting Equity | {args.notional}x Leverage | {args.tp*100:.2f}% TP | NO STOP LOSS")
    print(f"Entry Filter: Momentum > 10% Upward | {MOMENTUM_WINDOW}-Bar Window")
    print("="*100 + "\n")
    
    for source in ["coinbase", "toobit"]:
        print(f"Testing {source.upper()}...")
        print(f"  Loading {args.days} days of 1m data...", end=" ", flush=True)
        
        df = load_data(source, args.days)
        if df is None or len(df) == 0:
            print("X FAILED")
            continue
        
        print(f"OK - Loaded {len(df):,} candles")
        print(f"  Analyzing momentum entries...", end=" ", flush=True)
        
        result = run_backtest(df, source, args.tp, args.notional)
        results[source] = result
        
        print(f"OK - {result['trade_count']} momentum signals, {result['return_pct']:+.2f}% return\n")
    
    # Generate and save report
    report = generate_report(results, args.tp, args.notional)
    print(report)
    
    with open(args.out_report, "w", encoding="utf-8") as f:
        f.write(report)
    
    # Save full JSON results
    json_output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "starting_equity": STARTING_EQUITY,
            "notional_multiplier": args.notional,
            "tp_pct": args.tp,
            "stop_loss": "NONE",
            "compound_pct": COMPOUND_PCT,
            "taker_fee": TAKER_FEE,
            "days": args.days,
            "momentum_window": MOMENTUM_WINDOW,
            "momentum_threshold": MOMENTUM_THRESHOLD,
            "strategy": "momentum_spike_entry_10pct_upward",
        },
        "results": results,
    }
    
    with open(args.out_json, "w") as f:
        json.dump(json_output, f, indent=2)
    
    print(f"\nOK Report saved to: {args.out_report}")
    print(f"OK JSON results saved to: {args.out_json}")


if __name__ == "__main__":
    main()
