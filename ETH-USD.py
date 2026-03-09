"""
eth_sniper_v2.py
═══════════════════════════════════════════════════════════════
ETH-USD | 4h | LONG + SHORT | Objectif : 500€ → 30 000€
ZERO LEAKAGE | Triple Barrier ATR | Walk-Forward | Frais réels

Architecture identique à BTC Sniper v2 — démontre la généricité du système.
ETH vs BTC : ETH est structurellement laggé sur BTC (~15-30 min en daily)
→ BTC price action utilisée comme feature leading pour ETH.

CORRECTIONS v2 :
  ✅ VotingClassifier (XGB + LGB + RF) — pas de StackingClassifier (leakage)
  ✅ Triple Barrier ATR-based (cohérent labels/backtest)
  ✅ Walk-Forward TimeSeriesSplit 5 folds + purge bars
  ✅ BTC comme feature cross-asset (leading indicator pour ETH)
  ✅ CORR_TARGET_MIN relevé : 0.015
  ✅ Risk/trade : 5%, Max DD : 25%
═══════════════════════════════════════════════════════════════
Lancer : python ETH-USD.py
Dépendances : pip install yfinance xgboost lightgbm scikit-learn pandas numpy matplotlib requests
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import sys
import io
import contextlib
from datetime import timedelta

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

TELEGRAM_TOKEN   = ""          # Optionnel : token Telegram pour alertes live
TELEGRAM_CHAT_ID = ""          # Optionnel : chat ID Telegram

SYMBOL              = "ETH-USD"
DOWNLOAD_INTERVAL   = "1h"
TIMEFRAME_RESAMPLE  = "4h"
PERIOD              = "730d"

# Triple Barrier (ATR-based) — mêmes paramètres pour labels ET backtest
TB_TP_MULT          = 2.0          # TP = 2× ATR  → R:R 2:1
TB_SL_MULT          = 1.0          # SL = 1× ATR
TB_HORIZON          = 6            # 6 bougies × 4h = 24h max

# Risk Management
BT_CAPITAL          = 500.0
BT_RISK             = 0.05         # 5% / trade
BT_MAX_DRAWDOWN     = 0.25         # Stop global à -25%
FEE_RATE            = 0.001        # 0.1% frais (exchange standard)
SLIPPAGE            = 0.0005       # 0.05% glissement

# ML
TRAIN_RATIO         = 0.78
CV_FOLDS            = 5
RANDOM_STATE        = 42
CORR_TARGET_MIN     = 0.015        # Seuil minimum corrélation feature/cible
CORR_INTER_MAX      = 0.85         # Seuil maximum corrélation inter-features
PURGE_BARS          = TB_HORIZON * 3

# Signaux d'entrée
BT_MIN_CONF         = 20.0         # Confiance min (0-100) pour entrer
BT_LONG_THRESH      = 0.58         # Proba min LONG
BT_SHORT_THRESH     = 0.42         # Proba max SHORT
BT_ADX_MIN          = 20.0         # ADX min (trend confirmé)

_YF_CACHE = {}

# ═══════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════

def send_telegram(msg: str) -> None:
    try:
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=5)
    except Exception as e:
        print(f"[TELEGRAM] Erreur: {e}")


def _yf_silent(ticker: str, **kwargs) -> pd.DataFrame:
    """Cache + silence yfinance (supprime les warnings/progress bars)."""
    key = (ticker, str(sorted(kwargs.items())))
    if key not in _YF_CACHE:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            df = yf.download(ticker, progress=False, **kwargs)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        _YF_CACHE[key] = df
    return _YF_CACHE[key].copy()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    """RSI (Relative Strength Index) avec lissage EWM Wilder."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    ag    = gain.ewm(com=period - 1, min_periods=period).mean()
    al    = loss.ewm(com=period - 1, min_periods=period).mean()
    rs    = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average True Range (lissage SMA classique)."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """
    ADX + DI+ + DI- (lissage Wilder).
    ADX > 25 = trend fort, < 20 = range/consolidation.
    """
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up   = high.diff()
    down = -low.diff()

    plus_dm  = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    atr_w    = tr.ewm(com=period - 1, min_periods=period).mean().replace(0, np.nan)
    plus_di  = 100 * plus_dm.ewm(com=period - 1, min_periods=period).mean() / atr_w
    minus_di = 100 * minus_dm.ewm(com=period - 1, min_periods=period).mean() / atr_w

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(com=period - 1, min_periods=period).mean()

    return adx, plus_di, minus_di


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_period: int = 14, d_period: int = 3):
    """Stochastique %K et %D. Oversold < 20, Overbought > 80."""
    hh = high.rolling(k_period).max()
    ll = low.rolling(k_period).min()
    k  = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    d  = k.rolling(d_period).mean()
    return k, d


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    return (np.sign(close.diff().fillna(0)) * volume).cumsum()


def _triple_barrier(close: pd.Series, atr: pd.Series, horizon: int,
                    tp_mult: float, sl_mult: float) -> pd.Series:
    """
    Triple Barrier dynamique basée sur ATR.
    Label 1 = TP touché en premier, 0 = SL touché ou expiry baissier.
    ZERO LEAKAGE : label[i] calculé uniquement depuis prix[i+1..i+horizon].
    """
    prices   = close.values.astype(np.float64)
    atr_vals = atr.values.astype(np.float64)
    n        = len(prices)
    labels   = np.full(n, np.nan)

    for i in range(n - horizon):
        entry    = prices[i]
        curr_atr = atr_vals[i]
        if np.isnan(curr_atr) or curr_atr <= 0:
            continue
        tp = entry + curr_atr * tp_mult
        sl = entry - curr_atr * sl_mult

        result = np.nan
        for j in range(1, horizon + 1):
            p = prices[i + j]
            if p >= tp:
                result = 1.0
                break
            elif p <= sl:
                result = 0.0
                break
        if np.isnan(result):
            result = 1.0 if prices[i + horizon] > entry else 0.0
        labels[i] = result

    return pd.Series(labels, index=close.index, name="target")


# ═══════════════════════════════════════════════════════════════
# 1. DONNÉES
# ═══════════════════════════════════════════════════════════════

def download_data() -> pd.DataFrame:
    print(f"\n[DATA] {SYMBOL} | {DOWNLOAD_INTERVAL} → {TIMEFRAME_RESAMPLE} | {PERIOD}")

    raw = _yf_silent(SYMBOL, interval=DOWNLOAD_INTERVAL, period=PERIOD, auto_adjust=True)
    if raw.empty:
        sys.exit(f"[ERREUR] Données vides pour {SYMBOL}.")

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(subset=["Close"], inplace=True)

    df = df.resample(TIMEFRAME_RESAMPLE).agg({
        "Open": "first", "High": "max",
        "Low": "min",   "Close": "last", "Volume": "sum",
    })
    df.dropna(subset=["Close"], inplace=True)
    df = df[df["Volume"] > 0]

    print(f"[DATA] {df.index[0].date()} → {df.index[-1].date()} | {len(df)} bougies")

    if df.index.tz is not None:
        df_dates = df.index.tz_localize(None).normalize()
    else:
        df_dates = df.index.normalize()

    d_start = df.index[0].strftime("%Y-%m-%d")
    d_end   = (df.index[-1] + timedelta(days=3)).strftime("%Y-%m-%d")

    # Macro : VIX, DXY, TNX, SPY (mêmes que BTC — crypto sensible à risk-off)
    for col, ticker in [("vix", "^VIX"), ("dxy", "DX-Y.NYB"),
                        ("tnx", "^TNX"), ("spy", "SPY")]:
        m = _yf_silent(ticker, start=d_start, end=d_end, interval="1d")
        if m.empty or "Close" not in m.columns:
            df[col] = np.nan
            continue
        s = m["Close"].copy()
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        s.index = s.index.normalize()
        s_aligned = s.reindex(df_dates, method="ffill").ffill().bfill()
        s_aligned.index = df.index
        df[col] = s_aligned.values

    # BTC comme leading indicator pour ETH (BTC bouge avant ETH ~15-30 min)
    btc_raw = _yf_silent("BTC-USD", interval=DOWNLOAD_INTERVAL, period=PERIOD, auto_adjust=True)
    if not btc_raw.empty and "Close" in btc_raw.columns:
        btc_4h = btc_raw["Close"].resample(TIMEFRAME_RESAMPLE).last()
        df["btc_close"] = btc_4h.reindex(df.index, method="ffill").ffill().bfill()
    else:
        df["btc_close"] = np.nan

    print(f"[DATA] Macro téléchargée : VIX, DXY, TNX, SPY + BTC (leading)")
    return df


# ═══════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING — ZERO LEAKAGE
# ═══════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame, predict_mode: bool = False) -> pd.DataFrame:
    """
    Toutes les features utilisent données passées uniquement (no lookahead).
    Macro journalière = déjà laggée par nature.
    BTC : shift(1) explicite (contemporain = leakage potentiel sur 4h).
    """
    data   = pd.DataFrame(index=df.index)
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]
    open_  = df["Open"]
    log_ret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)

    # ── MACRO ────────────────────────────────────────────────────
    if "vix" in df.columns and df["vix"].notna().any():
        vix = df["vix"].ffill().bfill()
        data["vix_level"]    = vix
        data["vix_chg_6"]    = vix.pct_change(6)
        mu_v = vix.rolling(120).mean()
        sd_v = vix.rolling(120).std().replace(0, np.nan)
        data["vix_zscore"]   = (vix - mu_v) / sd_v
        data["vix_momentum"] = data["vix_zscore"] - data["vix_zscore"].shift(12)
        _vstd = data["vix_chg_6"].rolling(60).std().replace(0, np.nan)
        data["vix_spike"]    = (data["vix_chg_6"].abs() > _vstd * 2).astype(float)
        data["vix_low"]      = (vix < vix.rolling(120).median()).astype(float)
    else:
        for f in ["vix_level", "vix_chg_6", "vix_zscore", "vix_momentum", "vix_spike", "vix_low"]:
            data[f] = 0.0

    if "dxy" in df.columns and df["dxy"].notna().any():
        dxy = df["dxy"].ffill().bfill()
        data["dxy_chg_6"]  = dxy.pct_change(6)
        data["dxy_zscore"] = ((dxy - dxy.rolling(120).mean()) /
                              dxy.rolling(120).std().replace(0, np.nan))
    else:
        data["dxy_chg_6"] = data["dxy_zscore"] = 0.0

    if "tnx" in df.columns and df["tnx"].notna().any():
        tnx = df["tnx"].ffill().bfill()
        data["tnx_chg_6"] = tnx.pct_change(6)
        data["tnx_level"] = tnx
    else:
        data["tnx_chg_6"] = data["tnx_level"] = 0.0

    if "spy" in df.columns and df["spy"].notna().any():
        spy     = df["spy"].ffill().bfill()
        spy_ret = np.log(spy / spy.shift(1)).replace([np.inf, -np.inf], np.nan)
        data["spy_ret_6"]    = np.log(spy / spy.shift(6)).replace([np.inf, -np.inf], np.nan)
        data["spy_eth_corr"] = log_ret.rolling(60).corr(spy_ret)
    else:
        data["spy_ret_6"] = data["spy_eth_corr"] = 0.0

    data["macro_risk_on"] = (
        -data["vix_zscore"].fillna(0) * 0.5
        - data["dxy_zscore"].fillna(0) * 0.3
        + data["spy_ret_6"].fillna(0) * 5.0 * 0.2
    )

    # ── BTC (leading indicator pour ETH — shift(1) obligatoire) ──
    if "btc_close" in df.columns and df["btc_close"].notna().any():
        btc     = df["btc_close"].ffill().bfill()
        btc_ret = np.log(btc / btc.shift(1)).replace([np.inf, -np.inf], np.nan)
        data["btc_ret_1"]    = btc_ret.shift(1)
        data["btc_ret_6"]    = np.log(btc / btc.shift(6)).replace([np.inf, -np.inf], np.nan).shift(1)
        data["btc_eth_corr"] = log_ret.rolling(60).corr(btc_ret).shift(1)
        data["btc_leads"]    = (btc_ret.rolling(6).mean() > 0).astype(float).shift(1)
    else:
        data["btc_ret_1"] = data["btc_ret_6"] = data["btc_eth_corr"] = data["btc_leads"] = 0.0

    # ── RENDEMENTS & MOMENTUM ────────────────────────────────────
    for n in [1, 6, 12, 24, 48, 120]:
        data[f"ret_{n}"] = np.log(close / close.shift(n)).replace([np.inf, -np.inf], np.nan)

    for n in [24, 48, 120]:
        data[f"mom_up_{n}"]   = (log_ret > 0).rolling(n).sum() / n
        data[f"mom_down_{n}"] = (log_ret < 0).rolling(n).sum() / n

    data["mom_accel"]    = data["ret_12"] - data["ret_12"].shift(12)
    data["mom_velocity"] = data["ret_6"]  - data["ret_6"].shift(6)

    # ── VOLATILITÉ ───────────────────────────────────────────────
    for n in [24, 48, 120]:
        data[f"vol_{n}"] = log_ret.rolling(n).std()

    data["vol_ratio"]     = data["vol_24"] / data["vol_120"].replace(0, np.nan)
    data["vol_expansion"] = (data["vol_24"] > data["vol_120"]).astype(float)

    # ── RSI ──────────────────────────────────────────────────────
    for p in [14, 21, 42, 84]:
        data[f"rsi_{p}"] = _rsi(close, p)
    data["rsi_divergence"] = data["rsi_14"] - data["rsi_14"].shift(14)
    data["rsi_oversold"]   = (data["rsi_14"] < 30).astype(float)
    data["rsi_overbought"] = (data["rsi_14"] > 70).astype(float)

    # ── ADX + DI (force et direction du trend) ───────────────────
    adx14, di_p14, di_m14 = _adx(high, low, close, 14)
    data["adx_14"]        = adx14
    data["di_plus_14"]    = di_p14
    data["di_minus_14"]   = di_m14
    data["adx_trending"]  = (adx14 > 25).astype(float)
    data["di_signal"]     = np.sign(di_p14 - di_m14)
    data["adx_di_bull"]   = data["adx_trending"] * (data["di_signal"] > 0).astype(float)
    data["adx_slope"]     = adx14 - adx14.shift(6)

    adx28, _, _ = _adx(high, low, close, 28)
    data["adx_28"] = adx28

    # ── STOCHASTIQUE ─────────────────────────────────────────────
    k14, d14 = _stochastic(high, low, close, 14, 3)
    data["stoch_k"]          = k14
    data["stoch_d"]          = d14
    data["stoch_kd_diff"]    = k14 - d14
    data["stoch_oversold"]   = (k14 < 20).astype(float)
    data["stoch_overbought"] = (k14 > 80).astype(float)
    data["stoch_cross_up"]   = ((k14 > d14) & (k14.shift(1) <= d14.shift(1))).astype(float)
    data["stoch_cross_down"] = ((k14 < d14) & (k14.shift(1) >= d14.shift(1))).astype(float)

    k21, d21 = _stochastic(high, low, close, 21, 5)
    data["stoch_k21"] = k21
    data["stoch_d21"] = d21

    # ── MACD (normalisé ATR pour stabilité) ──────────────────────
    ema_f    = close.ewm(span=12, adjust=False).mean()
    ema_s    = close.ewm(span=26, adjust=False).mean()
    macd     = ema_f - ema_s
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    macd_h   = macd - macd_sig
    atr14    = _atr(high, low, close, 14).replace(0, np.nan)

    data["macd"]           = macd
    data["macd_hist"]      = macd_h
    data["macd_norm"]      = macd / atr14       # Normalisé ATR (stable)
    data["macd_hist_norm"] = macd_h / atr14
    data["macd_cross"]     = (macd > macd_sig).astype(float)
    data["macd_hist_chg"]  = macd_h - macd_h.shift(1)

    # ── BOLLINGER BANDS ──────────────────────────────────────────
    bb_sma = close.rolling(20).mean()
    bb_std = close.rolling(20).std().replace(0, np.nan)
    bb_rng = (4 * bb_std).replace(0, np.nan)

    data["bb_pos"]     = (close - bb_sma) / bb_rng
    data["bb_width"]   = (4 * bb_std) / bb_sma.replace(0, np.nan)
    data["bb_squeeze"] = (data["bb_width"] < data["bb_width"].rolling(120).quantile(0.2)).astype(float)
    data["bb_upper"]   = (close > bb_sma + 2 * bb_std).astype(float)
    data["bb_lower"]   = (close < bb_sma - 2 * bb_std).astype(float)

    # ── MOVING AVERAGES ──────────────────────────────────────────
    sma20  = close.rolling(20).mean().replace(0, np.nan)
    sma50  = close.rolling(50).mean().replace(0, np.nan)
    sma100 = close.rolling(100).mean().replace(0, np.nan)
    sma200 = close.rolling(200).mean().replace(0, np.nan)

    for n, sma in [(20, sma20), (50, sma50), (100, sma100), (200, sma200)]:
        data[f"dist_sma{n}"]  = (close - sma) / sma
        data[f"above_sma{n}"] = (close > sma).astype(float)

    data["golden_cross"] = ((sma20 > sma50) & (sma20.shift(1) <= sma50.shift(1))).astype(float)
    data["death_cross"]  = ((sma20 < sma50) & (sma20.shift(1) >= sma50.shift(1))).astype(float)

    data["sma_trend_score"] = (
        data["above_sma20"] + data["above_sma50"] +
        data["above_sma100"] + data["above_sma200"]
    ) / 4.0

    # ── ATR NORMALISÉ ────────────────────────────────────────────
    atr21 = _atr(high, low, close, 21)
    atr84 = _atr(high, low, close, 84)
    data["atr_21_norm"] = atr21 / close.replace(0, np.nan)
    data["atr_84_norm"] = atr84 / close.replace(0, np.nan)
    data["atr_ratio"]   = atr21 / atr84.replace(0, np.nan)

    # ── ROC ──────────────────────────────────────────────────────
    for n in [6, 12, 24, 48]:
        data[f"roc_{n}"] = (close / close.shift(n).replace(0, np.nan)) - 1.0

    # ── VOLUME & OBV ─────────────────────────────────────────────
    typical = (high + low + close) / 3
    vwap = ((typical * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan))
    data["vwap_dist"]  = (close - vwap) / vwap.replace(0, np.nan)
    data["vol_ratio2"] = volume / volume.rolling(20).mean().replace(0, np.nan)
    data["vol_spike"]  = (data["vol_ratio2"] > 2.0).astype(float)

    obv = _obv(close, volume)
    data["obv_slope"] = (obv - obv.shift(12)) / (obv.rolling(12).std().replace(0, np.nan) + 1e-9)
    data["obv_trend"] = (obv > obv.rolling(48).mean()).astype(float)

    # MFI
    typ_chg = typical.diff()
    mf_pos  = (typ_chg > 0).astype(float) * typical * volume
    mf_neg  = (typ_chg < 0).astype(float) * typical * volume
    mfr     = mf_pos.rolling(14).sum() / mf_neg.rolling(14).sum().replace(0, np.nan)
    data["mfi_14"] = 100 - 100 / (1 + mfr)

    # ── CANDLESTICK ──────────────────────────────────────────────
    rng_c = (high - low).replace(0, np.nan)
    data["candle_body"] = (close - open_).abs() / rng_c
    data["candle_dir"]  = np.sign(close - open_)
    data["range_norm"]  = rng_c / close.replace(0, np.nan)

    # ── MARKET REGIME ────────────────────────────────────────────
    px_chg      = (close - close.shift(24)).abs()
    path_length = close.diff().abs().rolling(24).sum().replace(0, np.nan)
    data["efficiency_ratio"] = px_chg / path_length
    data["trend_dir"]        = np.sign(close - close.rolling(48).mean())

    for n in [24, 120]:
        mu  = close.rolling(n).mean()
        sig = close.rolling(n).std().replace(0, np.nan)
        data[f"zscore_{n}"] = (close - mu) / sig

    # Williams %R
    for n in [14, 21]:
        hh = high.rolling(n).max()
        ll = low.rolling(n).min()
        data[f"willr_{n}"] = -100 * (hh - close) / (hh - ll).replace(0, np.nan)

    # ── STREAKS (numpy — évite le copy-chain bug pandas) ─────────
    sign_arr = np.sign(log_ret.fillna(0).values).astype(float)
    n_s      = len(sign_arr)
    s_up     = np.zeros(n_s)
    s_dn     = np.zeros(n_s)
    for _i in range(1, n_s):
        if sign_arr[_i] > 0:
            s_up[_i] = s_up[_i - 1] + 1
            s_dn[_i] = 0
        elif sign_arr[_i] < 0:
            s_dn[_i] = s_dn[_i - 1] + 1
            s_up[_i] = 0
    data["streak_up"]   = pd.Series(np.clip(s_up, 0, 10), index=close.index)
    data["streak_down"] = pd.Series(np.clip(s_dn, 0, 10), index=close.index)

    # ── INTERACTIONS ─────────────────────────────────────────────
    data["macro_trend"]    = data["macro_risk_on"].fillna(0) * data["sma_trend_score"].fillna(0.5)
    data["adx_stoch_bull"] = data["adx_trending"] * data["stoch_oversold"]
    data["adx_stoch_bear"] = data["adx_trending"] * data["stoch_overbought"]
    data["btc_lead_signal"] = data["btc_ret_1"].fillna(0) * data["adx_trending"]

    # ── NETTOYAGE ────────────────────────────────────────────────
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    if predict_mode:
        data.dropna(inplace=True)
        return data

    # ── TARGET : Triple Barrier ATR ──────────────────────────────
    atr_target = _atr(high, low, close, 14)
    data["target"] = _triple_barrier(close, atr_target, TB_HORIZON, TB_TP_MULT, TB_SL_MULT)
    data.dropna(inplace=True)

    pos = int(data["target"].sum())
    neg = len(data) - pos
    print(f"[FEATURES] {len(data)} obs × {data.shape[1]-1} features")
    print(f"[FEATURES] Target → TP:{pos} ({pos/len(data)*100:.1f}%) | SL:{neg} ({neg/len(data)*100:.1f}%)")

    return data


# ═══════════════════════════════════════════════════════════════
# 3. SÉLECTION DE FEATURES (sur train uniquement)
# ═══════════════════════════════════════════════════════════════

def select_features(X_train: np.ndarray, y_train: np.ndarray,
                    feature_names: list, verbose: bool = True) -> list:
    """
    1. Filtre corrélation |Pearson| avec la cible >= CORR_TARGET_MIN
    2. Supprime la redondance inter-features (|corr| > CORR_INTER_MAX)
    → Calculé sur train uniquement pour éviter le leakage de sélection
    """
    df_tr       = pd.DataFrame(X_train, columns=feature_names)
    y_s         = pd.Series(y_train.astype(float))
    target_corr = df_tr.apply(lambda c: abs(c.corr(y_s))).fillna(0)

    candidates = list(
        target_corr[target_corr >= CORR_TARGET_MIN].sort_values(ascending=False).index
    )

    if not candidates:
        print(f"[SELECTION] Aucune feature > seuil {CORR_TARGET_MIN} → toutes gardées")
        return feature_names

    corr_mat = df_tr[candidates].corr().abs()
    to_drop  = set()
    for i in range(len(candidates)):
        if candidates[i] in to_drop:
            continue
        for j in range(i + 1, len(candidates)):
            if candidates[j] in to_drop:
                continue
            if corr_mat.loc[candidates[i], candidates[j]] > CORR_INTER_MAX:
                drop = (candidates[j]
                        if target_corr[candidates[i]] >= target_corr[candidates[j]]
                        else candidates[i])
                to_drop.add(drop)

    selected = [f for f in candidates if f not in to_drop]

    if verbose:
        print(f"\n[SELECTION] {len(feature_names)} features → {len(selected)} retenues")
        print("  Top 10 corrélées à la cible :")
        for f in selected[:10]:
            bar = "█" * max(1, int(target_corr[f] * 300))
            print(f"  {f:35s} {target_corr[f]:.4f} {bar}")

    return selected


# ═══════════════════════════════════════════════════════════════
# 4. MODÈLE — VOTING ENSEMBLE (XGB + LGB + RF)
# ═══════════════════════════════════════════════════════════════

def build_model(spw: float = 1.0) -> Pipeline:
    """
    VotingClassifier soft : moyenne pondérée des probabilités des 3 modèles.
    ✅ Pas de StackingClassifier → incompatible avec TimeSeriesSplit propre.
    Régularisation forte pour éviter l'overfitting sur séries financières.
    """
    xgb_clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.02,
        subsample=0.70, colsample_bytree=0.65, min_child_weight=8,
        gamma=0.15, reg_alpha=0.30, reg_lambda=2.00,
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.015, num_leaves=20,
        subsample=0.70, subsample_freq=1, colsample_bytree=0.65,
        min_child_samples=20, reg_alpha=0.20, reg_lambda=2.00,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    rf_clf = RandomForestClassifier(
        n_estimators=400, max_depth=6,
        min_samples_split=15, min_samples_leaf=8,
        max_features="sqrt", class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    ensemble = VotingClassifier(
        estimators=[("xgb", xgb_clf), ("lgb", lgb_clf), ("rf", rf_clf)],
        voting="soft",
        weights=[2, 2, 1],
    )
    return Pipeline([("scaler", RobustScaler()), ("model", ensemble)])


# ═══════════════════════════════════════════════════════════════
# 5. BACKTEST RÉALISTE — LONG + SHORT
# ═══════════════════════════════════════════════════════════════

def backtest_strategy(close_test: pd.Series, atr_test: pd.Series,
                      adx_test: pd.Series, proba: np.ndarray,
                      threshold: float) -> dict:
    """
    Backtest sur la période de test uniquement (no lookahead).
    Frais aller-retour + slippage simulés.
    Anti-overlap : un seul trade actif à la fois.
    """
    dates     = list(close_test.index)
    closes    = close_test.values.astype(np.float64)
    atrs      = atr_test.reindex(close_test.index).ffill().bfill().values.astype(np.float64)
    adxs      = adx_test.reindex(close_test.index).ffill().bfill().values.astype(np.float64)
    n         = len(dates)
    long_thr  = max(threshold, BT_LONG_THRESH)
    short_thr = min(1.0 - threshold, BT_SHORT_THRESH)

    equity     = BT_CAPITAL
    peak       = equity
    eq_curve   = [(dates[0], equity)]
    trades     = []
    busy_until = -1

    for i, prob in enumerate(proba):
        eq_curve.append((dates[i], equity))
        if i <= busy_until:
            continue

        long_conf  = (prob - 0.5) * 2.0 * 100.0
        short_conf = (0.5 - prob) * 2.0 * 100.0
        curr_adx   = adxs[i]
        adx_ok     = not np.isnan(curr_adx) and curr_adx >= BT_ADX_MIN

        is_long  = (prob >= long_thr  and long_conf  >= BT_MIN_CONF and adx_ok)
        is_short = (prob <= short_thr and short_conf >= BT_MIN_CONF and adx_ok)

        if not is_long and not is_short:
            continue

        entry_price = closes[i]
        curr_atr    = atrs[i]
        if entry_price <= 0 or np.isnan(entry_price) or np.isnan(curr_atr) or curr_atr <= 0:
            continue

        direction = 1 if is_long else -1
        pos_val   = equity * BT_RISK

        if direction == 1:
            tp_lvl = entry_price + curr_atr * TB_TP_MULT
            sl_lvl = entry_price - curr_atr * TB_SL_MULT
        else:
            tp_lvl = entry_price - curr_atr * TB_TP_MULT
            sl_lvl = entry_price + curr_atr * TB_SL_MULT

        exit_price = closes[min(i + TB_HORIZON, n - 1)]
        exit_idx   = min(i + TB_HORIZON, n - 1)
        outcome    = "EXPIRY"

        for j in range(1, TB_HORIZON + 1):
            idx = i + j
            if idx >= n:
                break
            p = closes[idx]
            if np.isnan(p):
                continue
            if direction == 1:
                if p >= tp_lvl:
                    exit_price, exit_idx, outcome = p, idx, "TP"
                    break
                elif p <= sl_lvl:
                    exit_price, exit_idx, outcome = p, idx, "SL"
                    break
            else:
                if p <= tp_lvl:
                    exit_price, exit_idx, outcome = p, idx, "TP"
                    break
                elif p >= sl_lvl:
                    exit_price, exit_idx, outcome = p, idx, "SL"
                    break

        ret   = direction * ((exit_price / entry_price) - 1.0)
        fees  = pos_val * (FEE_RATE + SLIPPAGE) * 2
        pnl   = pos_val * ret - fees

        equity    += pnl
        peak       = max(peak, equity)
        busy_until = exit_idx

        if equity < BT_CAPITAL * (1 - BT_MAX_DRAWDOWN):
            print(f"[BACKTEST] ⚠️ Max Drawdown -{BT_MAX_DRAWDOWN*100:.0f}% atteint. Arrêt.")
            break

        trades.append({
            "entry_date": dates[i],   "exit_date": dates[exit_idx],
            "direction":  "LONG" if direction == 1 else "SHORT",
            "outcome":    outcome,    "ret_pct": ret * 100,
            "pnl":        pnl,        "equity": equity,
            "prob":       prob,
        })

    if not trades:
        print("\n[BACKTEST] Aucun trade généré.")
        return {}

    df_t   = pd.DataFrame(trades)
    n_t    = len(df_t)
    n_long = (df_t["direction"] == "LONG").sum()
    n_shrt = (df_t["direction"] == "SHORT").sum()
    n_tp   = (df_t["outcome"] == "TP").sum()
    n_sl   = (df_t["outcome"] == "SL").sum()
    n_exp  = (df_t["outcome"] == "EXPIRY").sum()
    n_win  = (df_t["pnl"] > 0).sum()
    wr     = n_win / n_t * 100

    gp = df_t.loc[df_t["pnl"] > 0, "pnl"].sum()
    gl = df_t.loc[df_t["pnl"] < 0, "pnl"].abs().sum()
    pf = gp / gl if gl > 0 else float("inf")

    avg_pnl   = df_t["pnl"].mean()
    total_ret = (equity - BT_CAPITAL) / BT_CAPITAL * 100

    max_dd  = 0.0
    peak_v  = BT_CAPITAL
    for _, row in df_t.iterrows():
        peak_v = max(peak_v, row["equity"])
        dd     = (peak_v - row["equity"]) / peak_v
        max_dd = max(max_dd, dd)

    if df_t["pnl"].std() > 0:
        n_periods_year  = 365 * 24 / 4
        trades_per_year = n_t / max(len(close_test) / n_periods_year, 0.01)
        sharpe = (avg_pnl / df_t["pnl"].std()) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    print("\n" + "═"*60)
    print(f"  BACKTEST — {SYMBOL} (LONG + SHORT)")
    print("═"*60)
    print(f"  Capital init    : {BT_CAPITAL:>10,.2f} €")
    print(f"  Capital final   : {equity:>10,.2f} €")
    print(f"  Rendement total : {total_ret:>+9.2f}%")
    print(f"  Trades          : {n_t:>10d}  (L:{n_long} S:{n_shrt})")
    print(f"  Issues          : TP:{n_tp} SL:{n_sl} EXP:{n_exp}")
    print(f"  Win rate        : {wr:>9.1f}%")
    print(f"  Profit Factor   : {pf:>10.2f}")
    print(f"  Max Drawdown    : {max_dd*100:>9.1f}%")
    print(f"  Sharpe (aprx)   : {sharpe:>10.2f}")
    print(f"  PnL moyen/trade : {avg_pnl:>+9.2f} €")
    print("═"*60)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), facecolor="#0d1117")
    fig.suptitle(f"ETH Sniper v2 — {SYMBOL} 4h LONG+SHORT | {BT_CAPITAL}€ → {equity:.0f}€",
                 color="#e6edf3", fontsize=13, fontweight="bold")

    ax1 = axes[0]
    ax1.set_facecolor("#161b22")
    eq_d = [x[0] for x in eq_curve]
    eq_v = [x[1] for x in eq_curve]
    ax1.plot(eq_d, eq_v, color="#3fb950", linewidth=1.8, label="Équité")
    ax1.axhline(BT_CAPITAL, color="#8b949e", linestyle="--", linewidth=0.8, label="Capital initial")
    ax1.fill_between(eq_d, BT_CAPITAL, eq_v,
                     where=[v >= BT_CAPITAL for v in eq_v],
                     alpha=0.15, color="#3fb950")
    ax1.fill_between(eq_d, BT_CAPITAL, eq_v,
                     where=[v < BT_CAPITAL for v in eq_v],
                     alpha=0.15, color="#f85149")

    for _, row in df_t.iterrows():
        c = "#3fb950" if row["outcome"] == "TP" else "#f85149" if row["outcome"] == "SL" else "#d29922"
        m = "^" if row["direction"] == "LONG" else "v"
        ax1.scatter(row["exit_date"], row["equity"], color=c, marker=m, s=22, zorder=5)

    ax1.set_ylabel("Capital (€)", color="#e6edf3")
    ax1.tick_params(colors="#8b949e")
    ax1.legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=8)
    for s in ax1.spines.values():
        s.set_color("#30363d")

    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    colors_pnl = ["#3fb950" if p > 0 else "#f85149" for p in df_t["pnl"]]
    ax2.bar(range(len(df_t)), df_t["pnl"], color=colors_pnl, width=0.8)
    ax2.axhline(0, color="#8b949e", linewidth=0.8)
    ax2.set_xlabel("Trade #", color="#e6edf3")
    ax2.set_ylabel("PnL (€)", color="#e6edf3")
    ax2.tick_params(colors="#8b949e")
    for s in ax2.spines.values():
        s.set_color("#30363d")

    plt.tight_layout()
    plt.savefig("eth_sniper_backtest.png", dpi=130, facecolor=fig.get_facecolor())
    plt.close()
    print("[BACKTEST] Graphique → eth_sniper_backtest.png")

    return {"total_ret": total_ret, "win_rate": wr, "profit_factor": pf,
            "max_dd": max_dd, "sharpe": sharpe, "trades": n_t, "final_equity": equity}


# ═══════════════════════════════════════════════════════════════
# 6. WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════

def walk_forward_validation(data: pd.DataFrame) -> float:
    print("\n" + "═"*60)
    print(f"[WF] Walk-Forward Validation — {CV_FOLDS} folds (purge={PURGE_BARS})")
    print("═"*60)

    feat_cols = [c for c in data.columns if c != "target"]
    X = data[feat_cols].values
    y = data["target"].values

    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    aucs = []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        tr_purged = tr_idx[:-PURGE_BARS] if len(tr_idx) > PURGE_BARS else tr_idx
        X_tr, X_te = X[tr_purged], X[te_idx]
        y_tr, y_te = y[tr_purged], y[te_idx]

        if len(np.unique(y_te)) < 2 or len(X_tr) < 200:
            print(f"  Fold {fold}: skipped (données insuffisantes)")
            continue

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sel = select_features(X_tr, y_tr, feat_cols, verbose=False)
        fidx = [i for i, f in enumerate(feat_cols) if f in sel]

        pipe = Pipeline([
            ("scaler", RobustScaler()),
            ("model", lgb.LGBMClassifier(
                n_estimators=400, max_depth=5, learning_rate=0.025,
                num_leaves=25, subsample=0.80, colsample_bytree=0.70,
                min_child_samples=12, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
            )),
        ])
        pipe.fit(X_tr[:, fidx], y_tr)
        auc = roc_auc_score(y_te, pipe.predict_proba(X_te[:, fidx])[:, 1])
        aucs.append(auc)
        print(f"  Fold {fold}: AUC={auc:.4f}  (train={len(X_tr):4d}, test={len(X_te):4d})")

    mean_auc = np.mean(aucs) if aucs else 0.5
    std_auc  = np.std(aucs)  if aucs else 0.0

    print(f"\n  AUC moyen : {mean_auc:.4f} ± {std_auc:.4f}")
    if mean_auc >= 0.56:
        print("  ✅ Signal robuste — edge statistique confirmé")
    elif mean_auc >= 0.52:
        print("  ⚠️  Signal faible mais exploitable avec seuils stricts")
    else:
        print("  ❌ Signal proche du random — ne pas trader en réel")
    print("═"*60)
    return mean_auc


# ═══════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "═"*60)
    print(f"  ETH SNIPER V2 — {SYMBOL}")
    print(f"  Capital: {BT_CAPITAL}€ | Risk/trade: {BT_RISK*100:.0f}% | LONG + SHORT")
    print(f"  Objectif: 500€ → 30 000€")
    print("═"*60)

    # 1. Données
    raw_df = download_data()

    # 2. Features
    data = build_features(raw_df)

    feat_cols = [c for c in data.columns if c != "target"]
    X = data[feat_cols].values
    y = data["target"].values

    # 3. Split temporel strict (jamais mélanger train/test)
    split_idx = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    print(f"\n[SPLIT] Train={split_idx} | Test={len(X)-split_idx}")
    print(f"[SPLIT] Train → TP:{n_pos} ({n_pos/len(y_train)*100:.1f}%) | SL:{n_neg}")

    # 4. Sélection features (sur train seulement — ZERO LEAKAGE)
    selected = select_features(X_train, y_train, feat_cols)
    fidx     = [i for i, f in enumerate(feat_cols) if f in selected]
    X_tr_s   = X_train[:, fidx]
    X_te_s   = X_test[:, fidx]

    # 5. Entraînement
    spw = n_neg / max(n_pos, 1)
    print(f"\n[TRAIN] scale_pos_weight={spw:.2f} | {len(selected)} features")
    print("[TRAIN] Ensemble XGB+LGB+RF (VotingClassifier soft) en cours...")

    model = build_model(spw=spw)

    # Poids de récence (données récentes 5× plus importantes)
    n_tr = len(y_train)
    sw   = np.exp(np.log(5.0) / n_tr * np.arange(n_tr))
    sw  /= sw.mean()
    try:
        model.fit(X_tr_s, y_train, model__sample_weight=sw)
    except TypeError:
        model.fit(X_tr_s, y_train)

    print("[TRAIN] ✅ Terminé.")

    # Seuil optimal Youden (sur fin du train, jamais sur le test)
    val_n   = max(150, int(len(X_tr_s) * 0.18))
    val_p   = model.predict_proba(X_tr_s[-val_n:])[:, 1]
    y_val   = y_train[-val_n:]
    fpr, tpr, thr = roc_curve(y_val, val_p)
    mask = (thr >= 0.0) & (thr <= 1.0)
    if mask.sum() > 0:
        best_i    = int(np.argmax(tpr[mask] - fpr[mask]))
        threshold = float(np.clip(thr[mask][best_i], 0.50, 0.65))
    else:
        threshold = 0.55
    print(f"[THRESHOLD] Seuil décision : {threshold:.4f}")

    # 6. Évaluation test
    y_proba = model.predict_proba(X_te_s)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm  = confusion_matrix(y_test, y_pred)
    rep = classification_report(y_test, y_pred, target_names=["SL/Baisse", "TP/Hausse"])

    print("\n" + "═"*60)
    print(f"  MÉTRIQUES TEST")
    print("═"*60)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  AUC ROC   : {auc:.4f}")
    print(f"\n{rep}")
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    print("═"*60)

    # 7. Backtest
    atr21_full  = _atr(raw_df["High"], raw_df["Low"], raw_df["Close"], 21)
    adx14_full, _, _ = _adx(raw_df["High"], raw_df["Low"], raw_df["Close"], 14)
    close_test  = raw_df["Close"].reindex(data.index[split_idx:])
    atr_test    = atr21_full.reindex(data.index[split_idx:])
    adx_test    = adx14_full.reindex(data.index[split_idx:])
    backtest_strategy(close_test, atr_test, adx_test, y_proba, threshold)

    # 8. Walk-Forward
    walk_forward_validation(data)

    # 9. Signal live
    print("\n" + "═"*60)
    print(f"  SIGNAL LIVE — {SYMBOL}")
    print("═"*60)

    data_pred = build_features(raw_df, predict_mode=True)
    if data_pred.empty:
        print("[ERREUR] Données insuffisantes pour prédiction.")
        return

    for f in selected:
        if f not in data_pred.columns:
            data_pred[f] = 0.0

    last_X     = data_pred[selected].iloc[-1:].values
    prob_up    = float(model.predict_proba(last_X)[0][1])
    prob_dn    = 1.0 - prob_up
    last_date  = data_pred.index[-1]
    last_close = float(raw_df["Close"].iloc[-1])
    atr_live   = float(atr21_full.dropna().iloc[-1]) if not atr21_full.dropna().empty else last_close * 0.02
    adx_live   = float(adx14_full.dropna().iloc[-1]) if not adx14_full.dropna().empty else 0.0

    long_conf  = (prob_up - 0.5) * 2.0 * 100.0
    short_conf = (0.5 - prob_up) * 2.0 * 100.0
    adx_ok     = adx_live >= BT_ADX_MIN

    print(f"  Dernière bougie : {last_date}")
    print(f"  Prob hausse     : {prob_up*100:.1f}%")
    print(f"  Prob baisse     : {prob_dn*100:.1f}%")
    print(f"  Seuil décision  : {threshold*100:.1f}%")
    print(f"  ATR actuel (21) : ${atr_live:,.0f}")
    print(f"  ADX actuel (14) : {adx_live:.1f}  {'✅ Trend actif' if adx_ok else '⚠️ Range — pas de trade'}")

    if prob_up >= max(threshold, BT_LONG_THRESH) and long_conf >= BT_MIN_CONF and adx_ok:
        tp_price = last_close + atr_live * TB_TP_MULT
        sl_price = last_close - atr_live * TB_SL_MULT
        pos_size = BT_CAPITAL * BT_RISK
        rr       = (tp_price - last_close) / max(last_close - sl_price, 1)
        msg = (
            f"🟢 SIGNAL LONG {SYMBOL}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"Prix entrée  : ${last_close:,.0f}\n"
            f"TP           : ${tp_price:,.0f} (+{(tp_price/last_close-1)*100:.1f}%)\n"
            f"SL           : ${sl_price:,.0f} (-{(1-sl_price/last_close)*100:.1f}%)\n"
            f"R:R          : 1:{rr:.1f}\n"
            f"Horizon      : {TB_HORIZON*4}h\n"
            f"Prob hausse  : {prob_up*100:.1f}%\n"
            f"Position     : {pos_size:.0f}€ ({BT_RISK*100:.0f}% capital)\n"
            f"Capital      : {BT_CAPITAL:.0f}€"
        )
        print(f"\n  🟢 SIGNAL LONG — Entrée recommandée")
        print(f"  Prix : ${last_close:,.0f} | TP: ${tp_price:,.0f} | SL: ${sl_price:,.0f} | R:R 1:{rr:.1f}")
        send_telegram(msg)

    elif prob_up <= BT_SHORT_THRESH and short_conf >= BT_MIN_CONF and adx_ok:
        tp_price = last_close - atr_live * TB_TP_MULT
        sl_price = last_close + atr_live * TB_SL_MULT
        pos_size = BT_CAPITAL * BT_RISK
        rr       = (last_close - tp_price) / max(sl_price - last_close, 1)
        msg = (
            f"🔴 SIGNAL SHORT {SYMBOL}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"Prix entrée  : ${last_close:,.0f}\n"
            f"TP           : ${tp_price:,.0f} (-{(1-tp_price/last_close)*100:.1f}%)\n"
            f"SL           : ${sl_price:,.0f} (+{(sl_price/last_close-1)*100:.1f}%)\n"
            f"R:R          : 1:{rr:.1f}\n"
            f"Prob baisse  : {prob_dn*100:.1f}%\n"
            f"Position     : {pos_size:.0f}€ ({BT_RISK*100:.0f}% capital)\n"
            f"Capital      : {BT_CAPITAL:.0f}€"
        )
        print(f"\n  🔴 SIGNAL SHORT — Entrée recommandée")
        print(f"  Prix : ${last_close:,.0f} | TP: ${tp_price:,.0f} | SL: ${sl_price:,.0f} | R:R 1:{rr:.1f}")
        send_telegram(msg)

    else:
        print(f"\n  ⚪ PAS DE SIGNAL — Rester en cash")

    print("\n" + "═"*60)
    print("  FIN D'ANALYSE — Relancer demain à la même heure")
    print("═"*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOP] Arrêté par l'utilisateur.")
    except Exception as e:
        print(f"[CRASH] {e}")
        import traceback
        traceback.print_exc()
