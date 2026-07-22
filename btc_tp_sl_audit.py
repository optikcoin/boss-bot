#!/usr/bin/env python3
"""
BTCUSDT 1m TP+SL backtest - MULTI-EXCHANGE AUDIT REPORT
Tests against Binance, Coinbase, and Toobit APIs.

Changes vs original:
- Added a real stop-loss: SL_PCT = 0.25 (a 25% adverse price move)
- Same-bar ambiguity: SL triggers first (conservative)
- Open trades at data end: marked-to-market, not discarded
- COMPOUND_PCT = 0.5 (50% compounding on wins only)
- NEW: Multi-exchange audit with detailed reporting
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
MAX_TRADES = 500

TP_PCT = 0.02
SL_PCT = 0.25
TAKER_FEE = 0.0005

EXCHANGES_CONFIG = {
    "binance": {"enableRateLimit": True},
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
        print(f"  ✗ Error loading from {source}: {str(e)}", file=sys.stderr)
        return None


def simulate_trade(df: pd.DataFrame, start_idx: int, fee_side: float, equity: float) -> Tuple[Optional[Dict], Optional[float], Optional[int]]:
    """Simulate a single trade with TP/SL."""
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    ts = df["ts"].to_numpy(dtype="datetime64[ns]")
    
    entry_idx = start_idx + 1
    if entry_idx >= len(df):
        return None, None, None
    
    entry = float(df["close"].iloc[entry_idx])
    tp_price = entry * (1.0 + TP_PCT)
    sl_price = entry * (1.0 - SL_PCT)
    qty = max(0.001, (equity * X_NOTIONAL) / max(entry, 1e-9))
    
    future_high = high[entry_idx + 1:]
    future_low = low[entry_idx + 1:]
    
    tp_hits = np.where(future_high >= tp_price)[0]
    sl_hits = np.where(future_low <= sl_price)[0]
    
    tp_i = tp_hits[0] if len(tp_hits) else None
    sl_i = sl_hits[0] if len(sl_hits) else None
    
    if tp_i is None and sl_i is None:
        # Mark-to-market at end
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
            "reason": "open_at_end",
            "qty": round(qty, 5),
            "gross": round(gross, 4),
            "fees": round(fees, 4),
            "pnl": round(pnl, 4),
            "exit_ts": str(pd.Timestamp(ts[last_idx])),
            "tp_hit": False,
        }
        return trade, pnl, last_idx
    
    if sl_i is not None and (tp_i is None or sl_i <= tp_i):
        exit_idx = entry_idx + 1 + int(sl_i)
        exit_px = sl_price
        reason = "sl"
        hit_tp = False
    else:
        exit_idx = entry_idx + 1 + int(tp_i)
        exit_px = tp_price
        reason = "tp"
        hit_tp = True
    
    gross = (exit_px - entry) * qty
    fees = (entry * qty + exit_px * qty) * fee_side
    pnl = gross - fees
    
    trade = {
        "side": "long",
        "opened": str(pd.Timestamp(ts[start_idx])),
        "entry": round(entry, 2),
        "exit": round(exit_px, 2),
        "reason": reason,
        "qty": round(qty, 5),
        "gross": round(gross, 4),
        "fees": round(fees, 4),
        "pnl": round(pnl, 4),
        "exit_ts": str(pd.Timestamp(ts[exit_idx])),
        "tp_hit": hit_tp,
    }
    return trade, pnl, exit_idx


def run_backtest(df: pd.DataFrame, source: str) -> Dict:
    """Run backtest on data."""
    equity = STARTING_EQUITY
    trades = []
    cursor = 0
    completed = 0
    
    while cursor < len(df) - 2 and completed < MAX_TRADES:
        trade, pnl, exit_idx = simulate_trade(df, cursor, TAKER_FEE, equity)
        if trade is None:
            cursor += 1
            continue
        trades.append(trade)
        completed += 1
        if pnl > 0:
            equity += pnl * COMPOUND_PCT
        else:
            equity += pnl
        if trade["reason"] == "open_at_end":
            break
        cursor = exit_idx + 1
    
    tp_wins = sum(1 for t in trades if t["reason"] == "tp")
    sl_losses = sum(1 for t in trades if t["reason"] == "sl")
    open_at_end = sum(1 for t in trades if t["reason"] == "open_at_end")
    
    total_pnl = sum(t["pnl"] for t in trades)
    total_fees = sum(t["fees"] for t in trades)
    winning_trades = sum(1 for t in trades if t["pnl"] > 0)
    losing_trades = sum(1 for t in trades if t["pnl"] < 0)
    
    win_rate = (winning_trades / len(trades) * 100) if trades else 0
    avg_win = np.mean([t["pnl"] for t in trades if t["pnl"] > 0]) if winning_trades else 0
    avg_loss = np.mean([t["pnl"] for t in trades if t["pnl"] < 0]) if losing_trades else 0
    profit_factor = abs(sum(t["pnl"] for t in trades if t["pnl"] > 0) / sum(t["pnl"] for t in trades if t["pnl"] < 0)) if losing_trades else float('inf')
    
    return {
        "source": source,
        "starting_equity": STARTING_EQUITY,
        "ending_equity": round(equity, 4),
        "total_pnl": round(total_pnl, 4),
        "return_pct": round((equity - STARTING_EQUITY) / STARTING_EQUITY * 100, 2),
        "trade_count": len(trades),
        "tp_wins": tp_wins,
        "sl_losses": sl_losses,
        "open_at_end": open_at_end,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate_pct": round(win_rate, 2),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "inf",
        "total_fees": round(total_fees, 4),
        "compound_pct_on_wins": COMPOUND_PCT,
        "tp_pct": TP_PCT,
        "sl_pct": SL_PCT,
        "data_points": len(df),
        "timespan": f"{df['ts'].min()} to {df['ts'].max()}",
        "trades": trades,
    }


def generate_audit_report(results: Dict[str, Dict]) -> str:
    """Generate comprehensive audit report."""
    report = []
    report.append("=" * 90)
    report.append("BTC/USDT 1m TP+SL BACKTEST - MULTI-EXCHANGE AUDIT REPORT")
    report.append("=" * 90)
    report.append("")
    report.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    report.append("")
    
    report.append("STRATEGY PARAMETERS:")
    report.append(f"  • Starting Equity: ${STARTING_EQUITY}")
    report.append(f"  • Notional Multiplier: {X_NOTIONAL}x")
    report.append(f"  • Take-Profit: {TP_PCT*100:.2f}%")
    report.append(f"  • Stop-Loss: {SL_PCT*100:.2f}%")
    report.append(f"  • Compound on Wins: {COMPOUND_PCT*100:.0f}%")
    report.append(f"  • Taker Fee: {TAKER_FEE*100:.04f}%")
    report.append(f"  • Max Trades: {MAX_TRADES}")
    report.append("")
    
    # Summary table
    report.append("EXCHANGE COMPARISON:")
    report.append("-" * 90)
    report.append(f"{'Exchange':<12} {'Equity':<12} {'Return %':<12} {'Trades':<10} {'Win Rate':<12} {'Profit Factor':<12}")
    report.append("-" * 90)
    
    for source in ["binance", "coinbase", "toobit"]:
        if source in results and results[source].get("trade_count", 0) > 0:
            r = results[source]
            report.append(
                f"{source:<12} ${r['ending_equity']:<11.2f} {r['return_pct']:<11.2f}% {r['trade_count']:<9} "
                f"{r['win_rate_pct']:<11.2f}% {str(r['profit_factor']):<11}"
            )
    
    report.append("-" * 90)
    report.append("")
    
    # Detailed results per exchange
    for source in ["binance", "coinbase", "toobit"]:
        if source not in results:
            continue
        
        r = results[source]
        report.append(f"\n{source.upper()} DETAILED RESULTS:")
        report.append("-" * 90)
        report.append(f"Data Points: {r['data_points']:,}")
        report.append(f"Timespan: {r['timespan']}")
        report.append("")
        report.append(f"Starting Equity: ${r['starting_equity']:.2f}")
        report.append(f"Ending Equity: ${r['ending_equity']:.2f}")
        report.append(f"Total P&L: ${r['total_pnl']:.4f}")
        report.append(f"Return: {r['return_pct']:.2f}%")
        report.append("")
        report.append(f"Total Trades: {r['trade_count']}")
        report.append(f"  ├─ TP Wins: {r['tp_wins']} ({r['tp_wins']/r['trade_count']*100:.1f}%)")
        report.append(f"  ├─ SL Losses: {r['sl_losses']} ({r['sl_losses']/r['trade_count']*100:.1f}%)")
        report.append(f"  └─ Open at End: {r['open_at_end']} ({r['open_at_end']/r['trade_count']*100:.1f}%)")
        report.append("")
        report.append(f"Winning Trades: {r['winning_trades']} | Losing Trades: {r['losing_trades']}")
        report.append(f"Win Rate: {r['win_rate_pct']:.2f}%")
        report.append(f"Avg Win: ${r['avg_win']:.4f} | Avg Loss: ${r['avg_loss']:.4f}")
        report.append(f"Profit Factor: {r['profit_factor']}")
        report.append(f"Total Fees Paid: ${r['total_fees']:.4f}")
        report.append("")
    
    # Data consistency checks
    report.append("\nDATA CONSISTENCY AUDIT:")
    report.append("-" * 90)
    
    if "binance" in results and "coinbase" in results:
        b_trades = results["binance"].get("trade_count", 0)
        c_trades = results["coinbase"].get("trade_count", 0)
        delta = abs(b_trades - c_trades)
        report.append(f"Binance vs Coinbase Trade Count: {b_trades} vs {c_trades} (Δ: {delta})")
        if delta == 0:
            report.append("  ✓ Trade counts match (consistent data)")
        else:
            report.append(f"  ⚠ Trade counts differ by {delta} (possible data quality issue)")
    
    if "binance" in results and "toobit" in results:
        b_trades = results["binance"].get("trade_count", 0)
        t_trades = results["toobit"].get("trade_count", 0)
        delta = abs(b_trades - t_trades)
        report.append(f"Binance vs Toobit Trade Count: {b_trades} vs {t_trades} (Δ: {delta})")
        if delta <= 5:
            report.append("  ✓ Trade counts similar (acceptable variance)")
        else:
            report.append(f"  ⚠ Significant trade count variance ({delta})")
    
    report.append("")
    report.append("=" * 90)
    
    return "\n".join(report)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--out-report", default="btc_audit_report.txt")
    ap.add_argument("--out-json", default="btc_audit_results.json")
    args = ap.parse_args()
    
    results = {}
    
    print("\n" + "="*90)
    print("BTC/USDT 1m TP+SL BACKTEST - MULTI-EXCHANGE AUDIT")
    print("="*90 + "\n")
    
    for source in ["binance", "coinbase", "toobit"]:
        print(f"Testing {source.upper()}...")
        print(f"  Loading {args.days} days of 1m data...", end=" ", flush=True)
        
        df = load_data(source, args.days)
        if df is None or len(df) == 0:
            print("✗ FAILED")
            continue
        
        print(f"✓ Loaded {len(df):,} candles")
        print(f"  Running backtest...", end=" ", flush=True)
        
        result = run_backtest(df, source)
        results[source] = result
        
        print(f"✓ {result['trade_count']} trades")
        print(f"  Equity: ${result['starting_equity']:.2f} → ${result['ending_equity']:.2f} ({result['return_pct']:+.2f}%)\n")
    
    # Generate and save report
    report = generate_audit_report(results)
    print(report)
    
    with open(args.out_report, "w") as f:
        f.write(report)
    
    # Save full JSON results
    json_output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "starting_equity": STARTING_EQUITY,
            "notional_multiplier": X_NOTIONAL,
            "tp_pct": TP_PCT,
            "sl_pct": SL_PCT,
            "compound_pct": COMPOUND_PCT,
            "taker_fee": TAKER_FEE,
            "days": args.days,
        },
        "results": results,
    }
    
    with open(args.out_json, "w") as f:
        json.dump(json_output, f, indent=2)
    
    print(f"\n✓ Report saved to: {args.out_report}")
    print(f"✓ JSON results saved to: {args.out_json}")


if __name__ == "__main__":
    main()
