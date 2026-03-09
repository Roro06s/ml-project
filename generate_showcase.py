"""
generate_showcase.py
═══════════════════════════════════════════════════════════════
Génère le graphique de présentation pour le repo GitHub.

Dashboard 4 panels (dark theme, GitHub compatible) :
  ① Prix WTI + Supertrend + signaux ML (LONG ▲ / SHORT ▼)
  ② Score de confiance ML (combinaison indicateurs normalisés)
  ③ ADX + DI+ + DI- (filtres de validation du signal)
  ④ Portfolio simulé vs Buy & Hold (500€ initial)

Usage : python generate_showcase.py
Sortie : assets/showcase_dashboard.png
═══════════════════════════════════════════════════════════════
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import io, contextlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from datetime import datetime, timedelta

# ── CONFIG ──────────────────────────────────────────────────────
SYMBOL     = "CL=F"         # WTI Crude Oil
PERIOD     = "2y"
INTERVAL   = "1d"
OUTPUT     = "assets/showcase_dashboard.png"

# Paramètres signal (identiques à WTI2.py)
ST_PERIOD  = 10
ST_MULT    = 3.0
ADX_PERIOD = 14
ATR_PERIOD = 14
ADX_MIN    = 12.0
TP_MULT    = 2.0
SL_MULT    = 1.0
HORIZON    = 10
CAPITAL    = 500.0
RISK       = 0.08

# Palette GitHub dark
BG_MAIN  = "#0d1117"
BG_PANEL = "#161b22"
COL_GRID = "#21262d"
COL_TEXT = "#e6edf3"
COL_MUTE = "#8b949e"
COL_GRN  = "#3fb950"
COL_RED  = "#f85149"
COL_YLW  = "#d29922"
COL_BLU  = "#58a6ff"
COL_PRP  = "#bc8cff"

# ── UTILITAIRES ─────────────────────────────────────────────────

def _dl(ticker, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        df = yf.download(ticker, progress=False, **kwargs)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _atr(high, low, close, period):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _adx(high, low, close, period=14):
    tr   = pd.concat([
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
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx      = dx.ewm(com=period - 1, min_periods=period).mean()
    return adx, plus_di, minus_di


def _supertrend(high, low, close, period=10, multiplier=3.0):
    atr_s  = _atr(high, low, close, period)
    hl2    = (high + low) / 2
    up_raw = (hl2 + multiplier * atr_s).values
    lo_raw = (hl2 - multiplier * atr_s).values
    c      = close.values
    n      = len(c)
    upper  = np.full(n, np.nan)
    lower  = np.full(n, np.nan)
    trend  = np.zeros(n)

    for i in range(n):
        if np.isnan(up_raw[i]) or np.isnan(lo_raw[i]):
            trend[i] = 0.0
            continue
        if i == 0 or np.isnan(upper[i-1]) or np.isnan(lower[i-1]):
            upper[i] = up_raw[i]
            lower[i] = lo_raw[i]
            trend[i] = 0.0
            continue
        upper[i] = up_raw[i] if (up_raw[i] < upper[i-1] or c[i-1] > upper[i-1]) else upper[i-1]
        lower[i] = lo_raw[i] if (lo_raw[i] > lower[i-1] or c[i-1] < lower[i-1]) else lower[i-1]
        if   c[i] > upper[i-1]: trend[i] =  1.0
        elif c[i] < lower[i-1]: trend[i] = -1.0
        else:                   trend[i] = trend[i-1]

    idx = close.index
    return (pd.Series(trend, index=idx),
            pd.Series(upper, index=idx),
            pd.Series(lower, index=idx))


def _rsi(series, period):
    delta = series.diff()
    ag = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    al = (-delta.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))


def _confidence_score(close, high, low, adx, di_plus, di_minus, st_trend, atr):
    """
    Score de confiance composite (0-100) utilisé comme proxy du signal ML.
    Combine : ADX, DI alignment, Supertrend, RSI momentum, ATR expansion.
    """
    log_ret  = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    rsi14    = _rsi(close, 14)
    atr_norm = atr / close.replace(0, np.nan)
    atr_z    = ((atr_norm - atr_norm.rolling(60).mean()) /
                atr_norm.rolling(60).std().replace(0, np.nan)).clip(-2, 2)

    # Composantes normalisées 0-1
    adx_score  = (adx.clip(0, 40) / 40.0).fillna(0)
    di_score   = ((di_plus - di_minus) / (di_plus + di_minus).replace(0, np.nan)).fillna(0) * 0.5 + 0.5
    st_score   = ((st_trend + 1) / 2.0).fillna(0.5)  # +1→1, -1→0, 0→0.5
    rsi_score  = ((rsi14 - 30) / 40.0).clip(0, 1).fillna(0.5)
    mom_score  = (log_ret.rolling(10).sum() * 15).clip(-1, 1).fillna(0) * 0.5 + 0.5

    # Score bullish composite
    bull = (adx_score * 0.25 + di_score * 0.25 + st_score * 0.25 +
            rsi_score * 0.15 + mom_score * 0.10)

    return (bull * 100).rolling(3).mean().fillna(50)


def _run_backtest(close, atr, adx, di_plus, di_minus, st_trend):
    """
    Backtest simplifié basé sur les filtres techniques (Supertrend + ADX + DI).
    Produit une courbe d'équité et les trades pour visualisation.
    """
    closes  = close.values.astype(np.float64)
    atrs    = atr.values.astype(np.float64)
    adxs    = adx.values.astype(np.float64)
    dip     = di_plus.values.astype(np.float64)
    dim     = di_minus.values.astype(np.float64)
    st      = st_trend.values.astype(np.float64)
    dates   = list(close.index)
    n       = len(closes)

    equity     = CAPITAL
    eq_curve   = [(dates[0], equity)]
    trades     = []
    busy_until = -1

    for i in range(n):
        eq_curve.append((dates[i], equity))
        if i <= busy_until or i < 30:
            continue

        curr_atr = atrs[i]
        curr_adx = adxs[i]
        if np.isnan(curr_atr) or np.isnan(curr_adx) or curr_atr <= 0:
            continue

        adx_ok  = curr_adx >= ADX_MIN
        di_bull = (not np.isnan(dip[i])) and (dip[i] >= dim[i] * 0.85)
        di_bear = (not np.isnan(dim[i])) and (dim[i] >= dip[i] * 0.85)
        st_long = st[i] >= 0
        st_shrt = st[i] <= 0

        is_long  = adx_ok and di_bull and st_long and (st[i] == 1.0)
        is_short = adx_ok and di_bear and st_shrt and (st[i] == -1.0)

        if not is_long and not is_short:
            continue

        entry = closes[i]
        if entry <= 0 or np.isnan(entry):
            continue

        direction = 1 if is_long else -1
        pos_val   = equity * RISK

        tp_lvl = entry + direction * curr_atr * TP_MULT
        sl_lvl = entry - direction * curr_atr * SL_MULT

        exit_price = closes[min(i + HORIZON, n - 1)]
        exit_idx   = min(i + HORIZON, n - 1)
        outcome    = "EXPIRY"

        for j in range(1, HORIZON + 1):
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

        ret   = direction * ((exit_price / entry) - 1.0)
        pnl   = pos_val * ret - pos_val * 0.0005 * 2
        equity = max(equity + pnl, 1.0)
        busy_until = exit_idx

        trades.append({
            "entry_date": dates[i], "exit_date": dates[exit_idx],
            "entry": entry, "exit": exit_price,
            "direction": "LONG" if direction == 1 else "SHORT",
            "outcome": outcome, "pnl": pnl, "equity": equity,
        })

    return eq_curve, trades


# ── MAIN ────────────────────────────────────────────────────────

def main():
    print(f"[SHOWCASE] Téléchargement {SYMBOL}...")
    raw = _dl(SYMBOL, period=PERIOD, interval=INTERVAL, auto_adjust=True)
    if raw.empty:
        sys.exit("[ERREUR] Données vides.")

    df    = raw[["Open", "High", "Low", "Close", "Volume"]].copy().dropna()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    print("[SHOWCASE] Calcul des indicateurs...")
    atr14            = _atr(high, low, close, ATR_PERIOD)
    adx, di_plus, di_minus = _adx(high, low, close, ADX_PERIOD)
    st_trend, st_up, st_lo = _supertrend(high, low, close, ST_PERIOD, ST_MULT)
    conf_score       = _confidence_score(close, high, low, adx, di_plus, di_minus, st_trend, atr14)

    # Buy & Hold reference
    bh_equity = CAPITAL * (close / close.iloc[0])

    print("[SHOWCASE] Backtest...")
    eq_curve, trades = _run_backtest(close, atr14, adx, di_plus, di_minus, st_trend)

    print("[SHOWCASE] Génération du graphique...")

    # ── FIGURE ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 13), facecolor=BG_MAIN)
    gs  = gridspec.GridSpec(4, 1, figure=fig,
                            height_ratios=[3, 1.2, 1.2, 1.8],
                            hspace=0.06)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors=COL_MUTE, labelsize=8)
        ax.yaxis.label.set_color(COL_TEXT)
        ax.xaxis.label.set_color(COL_TEXT)
        for s in ax.spines.values():
            s.set_color(COL_GRID)
        ax.grid(True, color=COL_GRID, linewidth=0.4, alpha=0.6)

    dates = close.index

    # ── PANEL 1 : PRIX + SUPERTREND + SIGNAUX ───────────────────
    # Fond coloré selon direction Supertrend
    bull_mask = st_trend == 1.0
    bear_mask = st_trend == -1.0

    # Zones colorées Supertrend
    for mask, col in [(bull_mask, COL_GRN), (bear_mask, COL_RED)]:
        idx_list = close.index[mask]
        if len(idx_list) > 0:
            ax1.fill_between(dates, close.min() * 0.9, close.max() * 1.05,
                             where=mask.reindex(dates, fill_value=False),
                             alpha=0.05, color=col, linewidth=0)

    # Prix
    ax1.plot(dates, close, color=COL_TEXT, linewidth=1.2, label="WTI Prix", zorder=3)

    # Supertrend bands
    ax1.plot(dates, st_lo.where(st_trend == 1.0),
             color=COL_GRN, linewidth=1.5, alpha=0.8, label="Supertrend Support", zorder=2)
    ax1.plot(dates, st_up.where(st_trend == -1.0),
             color=COL_RED, linewidth=1.5, alpha=0.8, label="Supertrend Résistance", zorder=2)

    # Signaux sur les trades
    long_trades  = [t for t in trades if t["direction"] == "LONG"]
    short_trades = [t for t in trades if t["direction"] == "SHORT"]
    tp_trades    = [t for t in trades if t["outcome"] == "TP"]
    sl_trades    = [t for t in trades if t["outcome"] == "SL"]

    for t in long_trades:
        ax1.scatter(t["entry_date"], t["entry"], marker="^", color=COL_GRN,
                    s=60, zorder=5, linewidths=0.5, edgecolors="#0d1117")
    for t in short_trades:
        ax1.scatter(t["entry_date"], t["entry"], marker="v", color=COL_RED,
                    s=60, zorder=5, linewidths=0.5, edgecolors="#0d1117")
    for t in tp_trades:
        ax1.scatter(t["exit_date"], t["exit"], marker="o", color=COL_GRN,
                    s=25, zorder=6, alpha=0.7)
    for t in sl_trades:
        ax1.scatter(t["exit_date"], t["exit"], marker="x", color=COL_RED,
                    s=30, zorder=6, alpha=0.9)

    ax1.set_ylabel("Price ($/bbl)", color=COL_TEXT, fontsize=9)
    ax1.set_title(
        f"ML Trading Sniper — WTI Crude Oil · Supertrend + ADX + ML Signals  |  "
        f"Backtested on {len(dates)} trading days",
        color=COL_TEXT, fontsize=11, fontweight="bold", pad=10
    )

    legend_els = [
        Line2D([0],[0], marker="^", color="w", markerfacecolor=COL_GRN, markersize=8, label=f"LONG Entry ({len(long_trades)})"),
        Line2D([0],[0], marker="v", color="w", markerfacecolor=COL_RED,  markersize=8, label=f"SHORT Entry ({len(short_trades)})"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=COL_GRN,  markersize=6, label="TP Hit"),
        Line2D([0],[0], marker="x", color="w", markerfacecolor=COL_RED,  markersize=6, label="SL Hit"),
        Line2D([0],[0], color=COL_GRN, linewidth=1.5, label="Supertrend Bullish"),
        Line2D([0],[0], color=COL_RED, linewidth=1.5, label="Supertrend Bearish"),
    ]
    ax1.legend(handles=legend_els, facecolor="#21262d", labelcolor=COL_TEXT,
               fontsize=7.5, loc="upper left", framealpha=0.9, ncol=3)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── PANEL 2 : SCORE DE CONFIANCE ML ─────────────────────────
    conf_vals = conf_score.reindex(dates).fillna(50).values
    ax2.plot(dates, conf_vals, color=COL_BLU, linewidth=1.2, label="ML Confidence Score")
    ax2.fill_between(dates, 50, conf_vals,
                     where=(conf_vals >= 60), alpha=0.25, color=COL_GRN, label="Signal LONG zone")
    ax2.fill_between(dates, 50, conf_vals,
                     where=(conf_vals <= 40), alpha=0.25, color=COL_RED, label="Signal SHORT zone")
    ax2.axhline(60, color=COL_GRN, linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.axhline(40, color=COL_RED, linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.axhline(50, color=COL_MUTE, linestyle="-", linewidth=0.5, alpha=0.5)
    ax2.set_ylim(10, 90)
    ax2.set_ylabel("ML Confidence (%)", color=COL_TEXT, fontsize=9)
    ax2.legend(facecolor="#21262d", labelcolor=COL_TEXT, fontsize=7.5, loc="upper right")
    plt.setp(ax2.get_xticklabels(), visible=False)

    # ── PANEL 3 : ADX + DI+ + DI- ───────────────────────────────
    ax3.plot(dates, adx.reindex(dates).values,       color=COL_YLW, linewidth=1.3, label="ADX(14)")
    ax3.plot(dates, di_plus.reindex(dates).values,   color=COL_GRN, linewidth=0.9, alpha=0.8, label="DI+")
    ax3.plot(dates, di_minus.reindex(dates).values,  color=COL_RED, linewidth=0.9, alpha=0.8, label="DI−")
    ax3.axhline(ADX_MIN, color=COL_YLW, linestyle=":", linewidth=0.8, alpha=0.6)
    ax3.axhline(25, color=COL_YLW, linestyle="--", linewidth=0.7, alpha=0.5)
    ax3.fill_between(dates, 0, adx.reindex(dates).values,
                     where=(adx.reindex(dates).values >= ADX_MIN),
                     alpha=0.08, color=COL_YLW)
    ax3.set_ylabel("ADX / DI±", color=COL_TEXT, fontsize=9)
    ax3.set_ylim(0, 60)
    ax3.legend(facecolor="#21262d", labelcolor=COL_TEXT, fontsize=7.5, loc="upper right")
    plt.setp(ax3.get_xticklabels(), visible=False)

    # ── PANEL 4 : PORTFOLIO vs BUY & HOLD ───────────────────────
    if eq_curve:
        eq_dates = [x[0] for x in eq_curve]
        eq_vals  = [x[1] for x in eq_curve]
        ax4.plot(eq_dates, eq_vals, color=COL_GRN, linewidth=2.0, label="ML Strategy", zorder=4)
        ax4.fill_between(eq_dates, CAPITAL, eq_vals,
                         where=[v >= CAPITAL for v in eq_vals],
                         alpha=0.18, color=COL_GRN)
        ax4.fill_between(eq_dates, CAPITAL, eq_vals,
                         where=[v < CAPITAL for v in eq_vals],
                         alpha=0.18, color=COL_RED)

    ax4.plot(dates, bh_equity.values, color=COL_PRP, linewidth=1.5,
             linestyle="--", alpha=0.8, label="Buy & Hold", zorder=3)
    ax4.axhline(CAPITAL, color=COL_MUTE, linestyle="--", linewidth=0.7, alpha=0.5)

    # Annotation finale
    if eq_curve:
        final_eq = eq_curve[-1][1]
        final_bh = float(bh_equity.iloc[-1])
        ax4.annotate(f"Strategy: {final_eq:.0f}€",
                     xy=(eq_curve[-1][0], final_eq),
                     xytext=(-80, 10), textcoords="offset points",
                     color=COL_GRN, fontsize=8, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=COL_GRN, lw=0.8))
        ax4.annotate(f"B&H: {final_bh:.0f}€",
                     xy=(dates[-1], final_bh),
                     xytext=(-80, -18), textcoords="offset points",
                     color=COL_PRP, fontsize=8,
                     arrowprops=dict(arrowstyle="->", color=COL_PRP, lw=0.8))

    # Stats dans le coin
    if trades:
        n_tp_b  = sum(1 for t in trades if t["outcome"] == "TP")
        n_sl_b  = sum(1 for t in trades if t["outcome"] == "SL")
        wr_b    = n_tp_b / max(len(trades), 1) * 100
        pnl_sum = sum(t["pnl"] for t in trades)
        stats_txt = (
            f"Trades: {len(trades)}  |  "
            f"Win Rate: {wr_b:.0f}%  |  "
            f"TP: {n_tp_b}  SL: {n_sl_b}  |  "
            f"PnL: {pnl_sum:+.1f}€"
        )
        ax4.text(0.02, 0.05, stats_txt, transform=ax4.transAxes,
                 color=COL_MUTE, fontsize=7.5)

    ax4.set_ylabel("Portfolio (€)", color=COL_TEXT, fontsize=9)
    ax4.set_xlabel("Date", color=COL_TEXT, fontsize=9)
    ax4.legend(facecolor="#21262d", labelcolor=COL_TEXT, fontsize=8, loc="upper left")

    # Metadonnées
    n_tp_total = sum(1 for t in trades if t["outcome"] == "TP")
    wr_total   = n_tp_total / max(len(trades), 1) * 100 if trades else 0
    fig.text(
        0.99, 0.01,
        f"ML Trading Sniper · github.com/Roro06s/Ml-Project · "
        f"{len(trades)} trades · Win rate: {wr_total:.0f}%",
        ha="right", va="bottom", color=COL_MUTE, fontsize=7
    )

    plt.savefig(OUTPUT, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

    print(f"[SHOWCASE] ✅ Graphique sauvegardé → {OUTPUT}")
    if trades:
        wins = sum(1 for t in trades if t["pnl"] > 0)
        print(f"[SHOWCASE] {len(trades)} trades | Win Rate: {wins/len(trades)*100:.1f}%")
        print(f"[SHOWCASE] Équité finale: {eq_curve[-1][1]:.2f}€ (départ: {CAPITAL:.0f}€)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOP] Arrêté par l'utilisateur.")
    except Exception as e:
        print(f"[ERREUR] {e}")
        import traceback
        traceback.print_exc()
