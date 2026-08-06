from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator

from src import factors

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "performance"

STRATEGY, BENCHMARK = "#2a78d6", "#eb6834"
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8a85"
GRID = "#e4e3df"

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
}

PCT = FuncFormatter(lambda v, _: f"{v:.0%}")


def _style(ax, ylabel=None, title=None, subtitle=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if subtitle:
        ax.text(0, 1.10, title, transform=ax.transAxes, color=INK, fontsize=13,
                fontweight="bold", va="bottom")
        ax.text(0, 1.025, subtitle, transform=ax.transAxes, color=INK3, fontsize=9.5, va="bottom")
    elif title:
        ax.text(0, 1.02, title, transform=ax.transAxes, color=INK, fontsize=13,
                fontweight="bold", va="bottom")


def _log_axis(ax, ticks):
    ax.set_yscale("log")
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}x" for t in ticks])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())


def _save(fig, name, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def equity_series(portfolio) -> pd.Series:
    eq = pd.DataFrame(portfolio.equity_curve, columns=["date", "equity"])
    eq = eq.drop_duplicates("date", keep="last").set_index("date")["equity"].sort_index()
    return eq / eq.iloc[0]


def benchmark_series(crsp: pd.DataFrame, index: pd.DatetimeIndex, total_return: bool = True) -> pd.Series:
    col = "ret" if total_return else "retx"
    df = crsp[["permno", "date", col, "mktcap"]].rename(columns={col: "ret"})
    r = factors.market_proxy_return(df).reindex(index).fillna(0.0)
    curve = (1 + r).cumprod()
    return curve / curve.iloc[0]


def plot_equity_curve(strat, cagr, out_dir=OUT_DIR):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(strat.index, strat.values, color=STRATEGY, linewidth=2.0, solid_capstyle="round")
    _log_axis(ax, [1, 2, 3, 4, 5])
    ax.set_ylim(0.72, max(5.4, strat.max() * 1.12))
    yrs = (strat.index[-1] - strat.index[0]).days / 365.25
    _style(ax, "growth of $1 (log scale)", "Multi-factor model: equity curve",
           f"Rank-IC composite, 25 names, 15% vol target  |  {cagr:.2%} CAGR over {yrs:.0f} years, "
           f"net of 7.5bps per trade")
    ax.annotate(f"{strat.iloc[-1]:.2f}x", xy=(strat.index[-1], strat.iloc[-1]), xytext=(8, 0),
                textcoords="offset points", color=INK, fontsize=11, fontweight="bold", va="center")
    _save(fig, "equity_curve", out_dir)


def plot_drawdown(strat, bench, out_dir=OUT_DIR):
    dd = strat / strat.cummax() - 1
    bdd = (bench / bench.cummax() - 1).min()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.fill_between(dd.index, dd.values, 0, color=STRATEGY, alpha=0.16, linewidth=0)
    ax.plot(dd.index, dd.values, color=STRATEGY, linewidth=1.4)
    ax.axhline(0, color=GRID, linewidth=1.0)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(min(-0.35, dd.min() * 1.08), 0.012)
    ax.annotate(f"worst {dd.min():.1%}", xy=(dd.idxmin(), dd.min()), xytext=(10, 12),
                textcoords="offset points", color=INK, fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=INK3, linewidth=0.9))
    _style(ax, "drawdown from peak", "Multi-factor model: drawdown",
           f"Peak-to-trough decline, daily  |  worst {dd.min():.1%} vs {bdd:.1%} for the S&P 500")
    _save(fig, "drawdown", out_dir)


def plot_vs_benchmark(strat, bench, out_dir=OUT_DIR):
    yrs = (strat.index[-1] - strat.index[0]).days / 365.25
    cagr = lambda s: s.iloc[-1] ** (1 / yrs) - 1
    s_dd = (strat / strat.cummax() - 1).min()
    b_dd = (bench / bench.cummax() - 1).min()

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(strat.index, strat.values, color=STRATEGY, linewidth=2.0,
            label="Multi-factor model", solid_capstyle="round")
    ax.plot(bench.index, bench.values, color=BENCHMARK, linewidth=2.0,
            label="S&P 500", solid_capstyle="round")
    labelled = [(strat, "Multi-factor model"), (bench, "S&P 500")]

    _log_axis(ax, [1, 2, 3, 5, 8])
    ax.set_ylim(0.55, 11.5)
    for series, name in labelled:
        ax.annotate(f"{name}  {series.iloc[-1]:.2f}x", xy=(series.index[-1], series.iloc[-1]),
                    xytext=(8, 0), textcoords="offset points", color=INK, fontsize=9.5,
                    fontweight="bold", va="center")
    ax.legend(frameon=False, loc="upper left", fontsize=9.5, labelcolor=INK2)
    _style(ax, "growth of $1 (log scale)", "Multi-factor model vs S&P 500",
           f"{cagr(strat):.2%} vs {cagr(bench):.2%} CAGR  |  "
           f"max drawdown {s_dd:.1%} vs {b_dd:.1%}")
    ax.set_xlim(strat.index[0], strat.index[-1] + pd.Timedelta(days=1150))
    _save(fig, "strategy_vs_sp500", out_dir)


def plot_crisis(strat, bench, start, end, title, subtitle, fname, out_dir=OUT_DIR):
    w = (strat.index >= pd.Timestamp(start)) & (strat.index <= pd.Timestamp(end))
    s = strat[w] / strat[w].iloc[0] * 100
    b = bench[w] / bench[w].iloc[0] * 100
    sd = (s / s.cummax() - 1).min()
    bd = (b / b.cummax() - 1).min()

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(s.index, s.values, color=STRATEGY, linewidth=2.0, label="Multi-factor model", solid_capstyle="round")
    ax.plot(b.index, b.values, color=BENCHMARK, linewidth=2.0, label="S&P 500", solid_capstyle="round")
    ax.axhline(100, color=GRID, linewidth=1.0, zorder=0)
    ax.legend(frameon=False, loc="lower left", fontsize=9.5, labelcolor=INK2)
    _style(ax, "total return, indexed to 100 at start", title,
           f"{subtitle}  |  worst drawdown {sd:.1%} vs {bd:.1%}")
    ax.set_xlim(s.index[0], s.index[-1])
    _save(fig, fname, out_dir)


def generate_all(portfolio, crsp, out_dir=OUT_DIR):
    with plt.rc_context(STYLE):
        strat = equity_series(portfolio)
        bench = benchmark_series(crsp, strat.index)
        yrs = (strat.index[-1] - strat.index[0]).days / 365.25
        cagr = strat.iloc[-1] ** (1 / yrs) - 1

        plot_equity_curve(strat, cagr, out_dir)
        plot_drawdown(strat, bench, out_dir)
        plot_vs_benchmark(strat, bench, out_dir)
        plot_crisis(strat, bench, "2007-10-01", "2009-06-30", "Global financial crisis",
                    "Oct 2007 market peak through Jun 2009", "crisis_2008", out_dir)
        plot_crisis(strat, bench, "2020-02-01", "2020-06-30", "COVID-19 crash",
                    "Feb 2020 peak through Jun 2020", "crisis_covid", out_dir)
    return strat, bench


if __name__ == "__main__":
    from src.optimize import Optimizer

    summary = json.load(open(CACHE_DIR / "ic12_final_summary.json"))
    sizing = {k: ([tuple(t) for t in v] if k == "drawdown_tiers" else v)
              for k, v in summary["sizing_params"].items()}

    opt = Optimizer(start="2001-01-01", end="2025-12-31", holdout_last_fold=False)
    portfolio = opt.run_backtest(opt.rescore(), opt.start, opt.end, sizing)
    crsp = opt._crsp.loc[opt._crsp["date"].between(opt.start, opt.end)]
    strat, bench = generate_all(portfolio, crsp)
    print(f"wrote 5 plots to {OUT_DIR}")
    print(f"model {strat.iloc[-1]:.2f}x | S&P 500 {bench.iloc[-1]:.2f}x")
