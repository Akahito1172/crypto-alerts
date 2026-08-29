import os
import time
import requests
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timezone

# --- ENVIRONMENT VARIABLES ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
CMC_API_KEY = os.getenv("CMC_API_KEY", "")

EXCHANGE_ID = os.getenv("EXCHANGE_ID", "okx")
TOP_N_CMC = int(os.getenv("TOP_N_CMC", "200")) 
CAPITAL_USDT = float(os.getenv("CAPITAL_USDT", "280"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.5"))
MAX_ALERTS = int(os.getenv("MAX_ALERTS", "10"))
BUY_SCORE = int(os.getenv("BUY_SCORE", "4"))

def get_exchange():
    exchange_class = getattr(ccxt, EXCHANGE_ID)
    return exchange_class({"enableRateLimit": True})

def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        response = requests.post(url, json=payload, timeout=15)
        return response.status_code == 200
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

def ema(series, period): return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def atr(df, period=14):
    prev_close = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def fetch_closed_ohlcv(exchange, symbol, timeframe, limit):
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not raw: return None
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    if len(df) > 1: df = df.iloc[:-1].reset_index(drop=True)
    return df

def add_indicators(df):
    out = df.copy()
    out["ema50"] = ema(out["close"], 50)
    out["ema200"] = ema(out["close"], 200)
    out["rsi14"] = rsi(out["close"], 14)
    out["macd"] = ema(out["close"], 12) - ema(out["close"], 26)
    out["macd_signal"] = ema(out["macd"], 9)
    out["vol_avg20"] = out["volume"].rolling(20).mean()
    out["atr14"] = atr(out, 14)
    conditions = pd.concat([
        out["ema50"] > out["ema200"], out["close"] > out["ema50"],
        (out["rsi14"] >= 50) & (out["rsi14"] <= 70), out["macd"] > out["macd_signal"],
        out["volume"] > out["vol_avg20"]
    ], axis=1).fillna(False).astype(int)
    out["score"] = conditions.sum(axis=1)
    return out

def get_cmc_top_symbols(exchange, limit=100):
    if not CMC_API_KEY:
        print("No CMC API Key found. Fallback to volume filter.")
        return get_volume_filter(exchange, limit)

    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    headers = {'X-CMC_PRO_API_KEY': CMC_API_KEY}
    params = {'start': 1, 'limit': limit, 'convert': 'USD'}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        data = response.json()
        if 'data' not in data:
            print(f"CMC API Error. Fallback.")
            return get_volume_filter(exchange, limit)
        cmc_symbols_raw = [coin['symbol'] for coin in data['data']]
    except Exception as e:
        print(f"CMC Request Exception: {e}")
        return get_volume_filter(exchange, limit)

    if not exchange.markets: exchange.load_markets()
    stables = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "USDP", "EUR", "GBP", "TRY", "BRL"}
    valid_symbols = []
    for sym in cmc_symbols_raw:
        if sym in stables: continue
        pair = f"{sym}/USDT"
        if pair in exchange.markets and exchange.markets[pair].get("active") is True:
            valid_symbols.append(pair)
    print(f"Mapped {len(valid_symbols)} CMC coins.")
    return valid_symbols

def get_volume_filter(exchange, top_n):
    tickers = exchange.fetch_tickers()
    rows = []
    for symbol, ticker in tickers.items():
        if symbol.endswith("/USDT") and not symbol.startswith(("USD", "EUR", "GBP")):
            qv = ticker.get("quoteVolume") or 0
            rows.append((symbol, float(qv)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in rows[:top_n]]

def get_min_cost(exchange, symbol):
    try:
        if not exchange.markets: exchange.load_markets()
        market = exchange.market(symbol)
        min_cost = market.get("limits", {}).get("cost", {}).get("min")
        if min_cost is not None: return float(min_cost)
    except: pass
    return 5.0

def fmt(x):
    try: return f"{float(x):.8g}"
    except: return str(x)

def build_buy_reasons(row):
    reasons = []
    if bool(row["ema50"] > row["ema200"]): reasons.append("EMA50>EMA200")
    if bool(row["close"] > row["ema50"]): reasons.append("ราคา>EMA50")
    if bool(50 <= row["rsi14"] <= 70): reasons.append("RSI 50-70")
    if bool(row["macd"] > row["macd_signal"]): reasons.append("MACD บวก")
    if bool(row["volume"] > row["vol_avg20"]): reasons.append("Vol สูง")
    return reasons

def make_alert(exchange, symbol, kind, row, reasons):
    price = float(row["close"])
    atr_val = float(row["atr14"]) if pd.notna(row["atr14"]) else 0.0
    stop_distance = 2.0 * atr_val
    stop = price - stop_distance
    tp1 = price + 1.0 * stop_distance
    tp2 = price + 2.0 * stop_distance
    tp3 = price + 3.0 * stop_distance
    base = symbol.split("/")[0]

    lines = [f"[{kind}] {symbol}", f"เวลา UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}", f"ราคา: {fmt(price)}", f"คะแนน: {int(float(row.get('score', 0)))}/5"]
    if reasons: lines.append(f"เหตุผล: {', '.join(reasons)}")

    if kind.startswith("BUY"):
        risk_amount = CAPITAL_USDT * RISK_PER_TRADE_PCT / 100.0
        lines.append(f"ทุน: {CAPITAL_USDT:.2f} USDT | เสี่ยง: {RISK_PER_TRADE_PCT}% = {risk_amount:.2f} USDT")
        if stop_distance > 0:
            qty = risk_amount / stop_distance
            notional = qty * price
            min_cost = get_min_cost(exchange, symbol)
            lines.append(f"ขนาดแนะนำ: {fmt(qty)} {base} (~{fmt(notional)} USDT)")
            lines.append("แผนความเสี่ยง/ทำกำไร:")
            lines.append(f"🛑 Stop: {fmt(stop)}")
            lines.append(f"🎯 TP1: {fmt(tp1)} | ปิดบางส่วน/เลื่อน Stop เป็นทุน")
            lines.append(f"🎯 TP2: {fmt(tp2)} | ปิดบางส่วน/เริ่ม Trailing")
            lines.append(f"🎯 TP3: {fmt(tp3)} | ปิดที่เหลือ")
            if notional < min_cost: lines.append(f"⚠️ เตือน: ขนาดต่ำกว่าขั้นต่ำ Exchange ({fmt(min_cost)} USDT)")
    else:
        lines.append("สถานะ: พิจารณาออก/ลดไม้/เฝ้าระวัง")
        if stop_distance > 0: lines.append(f"แนวระวัง: {fmt(stop)}")

    lines.append("หมายเหตุ: ไม่ใช่คำสั่งลงทุน")
    return "\n".join(lines)

def process_symbol(exchange, symbol):
    d1 = fetch_closed_ohlcv(exchange, symbol, "1d", 300)
    h4 = fetch_closed_ohlcv(exchange, symbol, "4h", 350)
    h1 = fetch_closed_ohlcv(exchange, symbol, "1h", 350)
    if d1 is None or h4 is None or h1 is None or len(d1)<220 or len(h4)<220 or len(h1)<220: return []

    d1, h4, h1 = add_indicators(d1), add_indicators(h4), add_indicators(h1)
    last_d1, curr_h4, prev_h4, curr_h1 = d1.iloc[-1], h4.iloc[-1], h4.iloc[-2], h1.iloc[-1]
    alerts = []

    daily_uptrend = bool(last_d1["ema50"] > last_d1["ema200"] and last_d1["close"] > last_d1["ema50"])
    curr_score, prev_score = float(curr_h4.get("score", 0)), float(prev_h4.get("score", 0))
    buy_cross = bool(curr_score >= BUY_SCORE and prev_score < BUY_SCORE and curr_h4["rsi14"] < 75)

    if daily_uptrend and buy_cross:
        h1_confirm = bool(curr_h1["close"] > curr_h1["ema50"] and curr_h1["rsi14"] > 50 and curr_h1["macd"] > curr_h1["macd_signal"])
        kind = "BUY_SETUP" if h1_confirm else "WATCH_BUY"
        reasons = build_buy_reasons(curr_h4)
        if h1_confirm: reasons.append("1H ยืนยัน")
        alerts.append({"kind": kind, "text": make_alert(exchange, symbol, kind, curr_h4, reasons)})

    exit_reasons = []
    if bool(curr_h4["ema50"] < curr_h4["ema200"]) and bool(prev_h4["ema50"] >= prev_h4["ema200"]): exit_reasons.append("EMA50 ตัดลง")
    if bool(curr_h4["close"] < curr_h4["ema200"]) and bool(prev_h4["close"] >= prev_h4["ema200"]): exit_reasons.append("หลุด EMA200")
    if bool(curr_h4["rsi14"] > 85) and bool(prev_h4["rsi14"] <= 85): exit_reasons.append("RSI Overbought")
    
    if exit_reasons:
        kind = "TAKE_PROFIT_WATCH" if "RSI Overbought" in exit_reasons else "EXIT_WATCH"
        alerts.append({"kind": kind, "text": make_alert(exchange, symbol, kind, curr_h4, exit_reasons)})
    return alerts

def main():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    exchange = get_exchange()
    symbols = get_cmc_top_symbols(exchange, TOP_N_CMC)
    if not symbols: return
    
    print(f"Scanning {len(symbols)} CMC Top coins...")
    all_alerts = []
    for symbol in symbols:
        try:
            alerts = process_symbol(exchange, symbol)
            all_alerts.extend(alerts)
            time.sleep(0.2)
        except Exception as e:
            print(f"ERROR {symbol}: {e}")

    buy_alerts = [a for a in all_alerts if a["kind"].startswith("BUY")]
    exit_alerts = [a for a in all_alerts if not a["kind"].startswith("BUY")]
    selected = (buy_alerts + exit_alerts)[:MAX_ALERTS]

    if selected:
        for alert in selected:
            send_telegram(alert["text"])
            time.sleep(1.2)
    else:
        # 🌟 ระบบรายงานตัว (Heartbeat)
        heartbeat_msg = (
            f"🤖 บอทยังทำงานอยู่นะครับ\n"
            f"เวลา: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC\n"
            f"สแกนไป {len(symbols)} เหรียญจาก CMC\n"
            f"ยังไม่พบจังหวะที่ปลอดภัยในขณะนี้ ✅\n"
            f"(รักษาเงินต้นคือสิ่งสำคัญที่สุดครับ)"
        )
        send_telegram(heartbeat_msg)
        print("No strong signals. Sent heartbeat.")

if __name__ == "__main__":
    main()
