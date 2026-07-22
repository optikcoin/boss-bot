#!/usr/bin/env python3
"""
BTC/USDT MOMENTUM SCALPING AUDIT - NO STOP LOSS
Tests 1000 trades across multiple TP levels
Optimized for high-momentum entry strategy with Claude co-investment

Multi-TP Testing:
  - 0.10% TP: Ultra-tight scalp
  - 0.15% TP: Recommended baseline
  - 0.25% TP: Medium scalp
"""

import json
import numpy as np
import pandas as pd
import ccxt
from datetime import datetime, timedelta, timezone
import sys
from typing import Dict, List, Tuple, Optional

STARTING_EQUITY = 100.0
X_NOTIONAL = 1.0
COMPOUND_PCT = 0.5
MAX_TRADES = 1000
TAKER_FEE = 0.0005

# NO STOP LOSS - letting positions run
TP_LEVELS = [0.0010, 0.0015, 0.0025]  # 0.10%, 0.15%, 0.25%

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


def simulate_trade(df: pd.DataFrame, start_idx: int, tp_pct: float, fee_side: float, equity: float) -> Tuple[Optional[Dict], Optional[float], Optional[int]]:
    """Simulate a single trade with TP only (NO STOP LOSS)."""
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    ts = df["ts"].to_numpy(dtype="datetime64[ns]")
    
    entry_idx = start_idx + 1
    if entry_idx >= len(df):
        return None, None, None
    
    entry = float(df["close"].iloc[entry_idx])
    tp_price = entry * (1.0 + tp_pct)
    qty = max(0.001, (equity * X_NOTIONAL) / max(entry, 1e-9))
    
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
        }
        return trade, pnl, last_idx
    
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
    }
    return trade, pnl, exit_idx


def run_backtest(df: pd.DataFrame, source: str, tp_pct: float) -> Dict:
    """Run backtest on data with specific TP level."""
    equity = STARTING_EQUITY
    trades = []
    cursor = 0
    completed = 0
    
    while cursor < len(df) - 2 and completed < MAX_TRADES:
        trade, pnl, exit_idx = simulate_trade(df, cursor, tp_pct, TAKER_FEE, equity)
        if trade is None:
            cursor += 1
            continue
        trades.append(trade)
        completed += 1
        if pnl > 0:
            equity += pnl * COMPOUND_PCT
        else:
            equity += pnl
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
    
    return {
        "source": source,
        "tp_pct": tp_pct,
        "starting_equity": STARTING_EQUITY,
        "ending_equity": round(equity, 4),
        "total_pnl": round(total_pnl, 4),
        "return_pct": round((equity - STARTING_EQUITY) / STARTING_EQUITY * 100, 2),
        "trade_count": len(trades),
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
        "compound_pct_on_wins": COMPOUND_PCT,
        "data_points": len(df),
        "timespan": f"{df['ts'].min()} to {df['ts'].max()}",
        "trades": trades,
    }


def generate_audit_report(results: Dict) -> str:
    """Generate comprehensive multi-TP audit report."""
    report = []
    report.append("=" * 100)
    report.append("BTC/USDT MOMENTUM SCALPING - MULTI-TP AUDIT")
    report.append("NO STOP LOSS | 1000 TRADE TARGET | CLAUDE CO-INVESTMENT")
    report.append("=" * 100)
    report.append("")
    report.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    report.append("")
    
    report.append("STRATEGY PARAMETERS:")
    report.append(f"  * Starting Equity: ${STARTING_EQUITY}")
    report.append(f"  * Notional Multiplier: {X_NOTIONAL}x")
    report.append(f"  * TP Levels Testing: {', '.join(f'{tp*100:.2f}%' for tp in TP_LEVELS)}")
    report.append(f"  * Stop-Loss: NONE (momentum hold)")
    report.append(f"  * Compound on Wins: {COMPOUND_PCT*100:.0f}%")
    report.append(f"  * Taker Fee: {TAKER_FEE*100:.04f}%")
    report.append(f"  * Max Trades: {MAX_TRADES}")
    report.append("")
    
    # Summary comparison table
    report.append("EXCHANGE + TP COMPARISON:")
    report.append("-" * 100)
    report.append(f"{'Exchange':<12} {'TP %':<8} {'Equity':<12} {'Return %':<12} {'Trades':<8} {'Win %':<10} {'Avg Hold':<10} {'PF':<8}")
    report.append("-" * 100)
    
    for exchange in ["coinbase", "toobit"]:
        for tp_level in TP_LEVELS:
            key = f"{exchange}_{tp_level}"
            if key in results:
                r = results[key]
                report.append(
                    f"{exchange:<12} {r['tp_pct']*100:>6.2f}% ${r['ending_equity']:<11.2f} {r['return_pct']:>11.2f}% "
                    f"{r['trade_count']:>7} {r['win_rate_pct']:>9.2f}% {r['avg_hold_bars']:>9.1f}b {str(r['profit_factor']):<8}"
                )
    
    report.append("-" * 100)
    report.append("")
    
    # Detailed results per exchange/TP
    for exchange in ["coinbase", "toobit"]:
        report.append(f"\n{exchange.upper()} RESULTS:")
        report.append("=" * 100)
        
        for tp_level in TP_LEVELS:
            key = f"{exchange}_{tp_level}"
            if key not in results:
                continue
            
            r = results[key]
            report.append(f"\nTP LEVEL: {r['tp_pct']*100:.2f}%")
            report.append("-" * 100)
            report.append(f"Data Points: {r['data_points']:,}")
            report.append(f"Timespan: {r['timespan']}")
            report.append("")
            report.append(f"Starting Equity: ${r['starting_equity']:.2f}")
            report.append(f"Ending Equity: ${r['ending_equity']:.2f}")
            report.append(f"Total P&L: ${r['total_pnl']:.4f}")
            report.append(f"Return: {r['return_pct']:.2f}%")
            report.append("")
            report.append(f"Total Trades: {r['trade_count']}")
            report.append(f"  - TP Hits: {r['tp_hits']} ({r['tp_hits']/r['trade_count']*100:.1f}%)")
            report.append(f"  - No TP (MTM): {r['no_tp_hits']} ({r['no_tp_hits']/r['trade_count']*100:.1f}%)")
            report.append(f"  - Avg Hold: {r['avg_hold_bars']:.1f} bars")
            report.append("")
            report.append(f"Winning Trades: {r['winning_trades']} | Losing Trades: {r['losing_trades']}")
            report.append(f"Win Rate: {r['win_rate_pct']:.2f}%")
            report.append(f"Avg Win: ${r['avg_win']:.4f} | Avg Loss: ${r['avg_loss']:.4f}")
            report.append(f"Profit Factor: {r['profit_factor']}")
            report.append(f"Total Fees Paid: ${r['total_fees']:.4f}")
            report.append("")
    
    # Recommendations
    report.append("\n" + "=" * 100)
    report.append("RECOMMENDATIONS FOR CLAUDE CO-INVESTMENT:")
    report.append("=" * 100)
    
    best_result = None
    best_key = None
    best_return = -999
    
    for key, r in results.items():
        if r['return_pct'] > best_return:
            best_return = r['return_pct']
            best_key = key
            best_result = r
    
    if best_result:
        exchange, tp = best_key.rsplit("_", 1)
        report.append(f"\nBest Performer: {exchange.upper()} @ {float(tp)*100:.2f}% TP")
        report.append(f"  - Return: {best_result['return_pct']:+.2f}%")
        report.append(f"  - Win Rate: {best_result['win_rate_pct']:.2f}%")
        report.append(f"  - Avg Hold: {best_result['avg_hold_bars']:.1f} bars")
        report.append(f"  - Trades: {best_result['trade_count']}")
    
    report.append("")
    report.append("=" * 100)
    
    return "\n".join(report)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--out-report", default="btc_momentum_report.txt")
    ap.add_argument("--out-json", default="btc_momentum_results.json")
    args = ap.parse_args()
    
    results = {}
    
    print("\n" + "="*100)
    print("BTC/USDT MOMENTUM SCALPING - MULTI-TP AUDIT")
    print("="*100 + "\n")
    
    for source in ["coinbase", "toobit"]:
        print(f"Testing {source.upper()}...")
        print(f"  Loading {args.days} days of 1m data...", end=" ", flush=True)
        
        df = load_data(source, args.days)
        if df is None or len(df) == 0:
            print("X FAILED")
            continue
        
        print(f"OK - Loaded {len(df):,} candles")
        
        for tp_pct in TP_LEVELS:
            print(f"  Testing TP={tp_pct*100:.2f}%...", end=" ", flush=True)
            
            result = run_backtest(df, source, tp_pct)
            results[f"{source}_{tp_pct}"] = result
            
            print(f"OK - {result['trade_count']} trades, {result['return_pct']:+.2f}% return\n")
    
    # Generate and save report
    report = generate_audit_report(results)
    print(report)
    
    with open(args.out_report, "w", encoding="utf-8") as f:
        f.write(report)
    
    # Save full JSON results
    json_output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "starting_equity": STARTING_EQUITY,
            "notional_multiplier": X_NOTIONAL,
            "tp_levels": TP_LEVELS,
            "compound_pct": COMPOUND_PCT,
            "taker_fee": TAKER_FEE,
            "days": args.days,
            "strategy": "momentum_scalping_no_sl",
        },
        "results": results,
    }
    
    with open(args.out_json, "w") as f:
        json.dump(json_output, f, indent=2)
    
    print(f"\nOK Report saved to: {args.out_report}")
    print(f"OK JSON results saved to: {args.out_json}")


if __name__ == "__main__":
    main()
