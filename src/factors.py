from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import settings

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def sector_neutral_percentile(
    df: pd.DataFrame,
    factor: str,
    sign: int = 1,
    date_col: str = "date",
    sector_col: str = "gsector",
) -> pd.Series:
    x = df[factor].astype("float64") * sign
    return x.groupby([df[date_col], df[sector_col]]).rank(pct=True)


def cross_sectional_percentile(df: pd.DataFrame, col: str, date_col: str = "date") -> pd.Series:
    return df[col].astype("float64").groupby(df[date_col]).rank(pct=True)


def weighted_combine(df: pd.DataFrame, col_weights: dict[str, float]) -> pd.Series:
    cols = list(col_weights.keys())
    weights = pd.Series(col_weights, dtype="float64")
    values = df[cols].astype("float64")
    mask = values.notna()
    effective_weights = mask.mul(weights, axis=1)
    row_weight_sum = effective_weights.sum(axis=1)
    weighted_sum = (values.fillna(0.0) * effective_weights).sum(axis=1)
    return (weighted_sum / row_weight_sum).where(row_weight_sum > 0)


def weighted_combine_dynamic(df: pd.DataFrame, value_cols: list[str], weight_cols: list[str]) -> pd.Series:
    values = df[value_cols].astype("float64")
    weights = df[weight_cols].astype("float64")
    weights.columns = value_cols
    mask = values.notna()
    effective_weights = mask * weights
    row_weight_sum = effective_weights.sum(axis=1)
    weighted_sum = (values.fillna(0.0) * effective_weights).sum(axis=1)
    return (weighted_sum / row_weight_sum).where(row_weight_sum > 0)


def rolling_rank_ic(
    panel: pd.DataFrame,
    daily_returns: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    factor_list: list[str] | None = None,
    horizon_days: int | None = None,
) -> pd.DataFrame:
    factor_list = factor_list if factor_list is not None else ALL_FACTORS
    horizon_days = horizon_days if horizon_days is not None else settings.LABEL_HORIZON_DAYS

    rets = daily_returns[["permno", "date", "ret"]].copy()
    rets["date"] = pd.to_datetime(rets["date"])
    rets["ret"] = rets["ret"].astype("float64")
    rets = rets.sort_values(["permno", "date"])
    cum = rets.groupby("permno")["ret"].transform(lambda s: np.log1p(s.fillna(0.0)).cumsum())
    rets["fwd_ret"] = np.exp(cum.groupby(rets["permno"]).shift(-horizon_days) - cum) - 1

    cols = [f"pct_{f}" for f in factor_list]
    df = panel.loc[panel["date"].isin(rebal_dates), ["permno", "date"] + cols]
    df = df.merge(rets[["permno", "date", "fwd_ret"]], on=["permno", "date"], how="left")

    out = {}
    for date, g in df.groupby("date", sort=True):
        y = g["fwd_ret"].rank()
        x = g[cols].rank()
        valid = y.notna()
        if valid.sum() < 30:
            continue
        xv, yv = x.loc[valid], y.loc[valid]
        usable = xv.columns[(xv.notna().sum() >= 30) & (xv.nunique() > 1)]
        out[date] = xv[usable].corrwith(yv)
    ic = pd.DataFrame(out).T
    ic.columns = [c.removeprefix("pct_") for c in ic.columns]
    ic.index = pd.DatetimeIndex(ic.index, name="date")
    return ic.sort_index()


# Water-fill: clip-then-renormalize does not converge.
def _constrain_family_weights(weights: pd.Series, lo: float, hi: float) -> pd.Series:
    idx = weights.index
    w = weights.astype("float64").clip(lower=0.0)
    w = w / w.sum() if w.sum() > 0 else pd.Series(1.0 / len(idx), index=idx)

    pinned: dict = {}
    for _ in range(len(idx) + 1):
        free = [k for k in idx if k not in pinned]
        if not free:
            break
        remaining = 1.0 - sum(pinned.values())
        sub = w[free]
        sub = sub / sub.sum() * remaining if sub.sum() > 0 else pd.Series(remaining / len(free), index=free)
        # caps first, then floors
        over = {k: hi for k, v in sub.items() if v > hi}
        if over:
            pinned.update(over)
            continue
        under = {k: lo for k, v in sub.items() if v < lo}
        if under:
            pinned.update(under)
            continue
        return pd.Series({**pinned, **sub.to_dict()}).reindex(idx)
    return pd.Series(pinned).reindex(idx)


def ic_weight_schedule(
    ic: pd.DataFrame,
    window_months: int | None = None,
    horizon_days: int | None = None,
    rebalance_interval_days: int | None = None,
    family_bounds: tuple[float, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    window_months = window_months if window_months is not None else settings.RANK_IC_WINDOW_MONTHS
    horizon_days = horizon_days if horizon_days is not None else settings.LABEL_HORIZON_DAYS
    step = rebalance_interval_days if rebalance_interval_days is not None else settings.REBALANCE_INTERVAL_DAYS
    lo, hi = family_bounds if family_bounds is not None else settings.FAMILY_WEIGHT_BOUNDS

    # lag past the label horizon: no look-ahead
    lag = int(np.ceil(horizon_days / step))
    window = max(1, int(round(window_months * 21 / step)))
    trailing = ic.shift(lag).rolling(window, min_periods=max(2, window // 3)).mean()

    positive = trailing.clip(lower=0.0)

    within: dict[str, pd.DataFrame] = {}
    family_raw = {}
    for family, (flist, _) in FAMILIES.items():
        cols = [f for f in flist if f in positive.columns]
        block = positive[cols]
        total = block.sum(axis=1)
        fallback = pd.Series(
            {f: settings.WITHIN_FACTOR_WEIGHT_OVERRIDES.get(family, {}).get(f, 1.0) for f in cols}
        )
        fallback = fallback / fallback.sum()
        w = block.div(total, axis=0)
        w.loc[total <= 0, :] = fallback.values
        within[family] = w
        family_raw[family] = block.mean(axis=1)

    fam = pd.DataFrame(family_raw)
    fam_fallback = pd.Series(settings.FACTOR_WEIGHTS)
    fam_fallback = fam_fallback / fam_fallback.sum()
    rows = {}
    for date, row in fam.iterrows():
        if not np.isfinite(row).all() or row.sum() <= 0:
            rows[date] = fam_fallback
        else:
            rows[date] = _constrain_family_weights(row, lo, hi)
    fam_sched = pd.DataFrame(rows).T
    fam_sched.index = pd.DatetimeIndex(fam_sched.index, name="date")
    return fam_sched.sort_index(), within


def apply_gics_exclusions(
    df: pd.DataFrame, exclusions: dict[str, list[str]], sector_col: str = "gsector"
) -> None:
    for code, factors in exclusions.items():
        mask = df[sector_col] == code
        for f in factors:
            if f in df.columns:
                df.loc[mask, f] = np.nan


def _live(factors: list[str]) -> list[str]:
    return [f for f in factors if f not in settings.DISABLED_FACTORS]


def load_fundamentals_panel() -> pd.DataFrame:
    return pd.read_parquet(CACHE_DIR / "fundamentals_panel.parquet")


def load_momentum_panel() -> pd.DataFrame:
    return pd.read_parquet(CACHE_DIR / "momentum_panel.parquet")


def load_volatility_panel() -> pd.DataFrame:
    return pd.read_parquet(CACHE_DIR / "volatility_panel.parquet")


VALUE_FACTORS_ALL = [
    "book_to_market",
    "earnings_yield",
    "sales_yield",
    "ebitda_to_ev",
    "ocf_yield",
    "fcf_yield",
    "shareholder_yield",
    "tangible_book_to_market",
]
VALUE_FACTORS = _live(VALUE_FACTORS_ALL)
VALUE_SIGNS = {f: 1 for f in VALUE_FACTORS_ALL}


def compute_value(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    pref_stock = panel["pstkrv"].fillna(panel["pstkl"]).fillna(panel["pstk"]).fillna(0)
    panel["book_equity"] = panel["seq"] + panel["txditc"].fillna(0) - pref_stock
    panel["enterprise_value"] = (
        panel["mktcap"] + panel["dltt"].fillna(0) + panel["dlc"].fillna(0) - panel["che"].fillna(0)
    )

    panel["book_to_market"] = panel["book_equity"] / panel["mktcap"]
    panel["earnings_yield"] = panel["ni"] / panel["mktcap"]
    panel["sales_yield"] = panel["sale"] / panel["mktcap"]
    panel["ebitda_to_ev"] = panel["ebitda"] / panel["enterprise_value"]
    panel["ocf_yield"] = panel["oancf"] / panel["mktcap"]
    panel["fcf_yield"] = (panel["oancf"] - panel["capx"]) / panel["mktcap"]
    panel["shareholder_yield"] = (
        panel["dvc"].fillna(0) + panel["prstkc"].fillna(0) - panel["sstk"].fillna(0)
    ) / panel["mktcap"]

    panel["tangible_book_to_market"] = (
        panel["book_equity"] - panel["gdwl"].fillna(0) - panel["intan"].fillna(0)
    ) / panel["mktcap"]
    panel.loc[panel["gsector"] != "40", "tangible_book_to_market"] = np.nan
    return panel


QUALITY_FACTORS_ALL = [
    "gross_profitability",
    "roe",
    "roa",
    "operating_profitability",
    "accruals",
    "leverage",
    "interest_coverage",
    "asset_turnover",
    "net_debt_issuance",
    "days_sales_outstanding",
    "days_inventory",
    "days_payable",
    "cash_conversion_cycle",
    "goodwill_intangibles_pct",
    "efficiency_ratio",
    "capital_adequacy",
    "reserve_adequacy",
]
QUALITY_FACTORS = _live(QUALITY_FACTORS_ALL)

QUALITY_SIGNS = {f: 1 for f in QUALITY_FACTORS_ALL}
for _f in (
    "accruals",
    "leverage",
    "days_sales_outstanding",
    "days_inventory",
    "cash_conversion_cycle",
    "goodwill_intangibles_pct",
    "efficiency_ratio",
    "net_debt_issuance",
):
    QUALITY_SIGNS[_f] = -1

QUALITY_GICS_EXCLUSIONS = {
    "40": [
        "leverage",
        "asset_turnover",
        "operating_profitability",
        "gross_profitability",
        "days_sales_outstanding",
        "days_inventory",
        "days_payable",
        "cash_conversion_cycle",
    ],
}


def compute_quality(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel if "book_equity" in panel.columns else compute_value(panel)
    panel = panel.copy()

    panel["gross_profitability"] = panel["gp"] / panel["at"]
    panel["roe"] = panel["ni"] / panel["ceq"]
    panel["roa"] = panel["ni"] / panel["at"]
    panel["operating_profitability"] = (
        panel["revt"] - panel["cogs"] - panel["xsga"].fillna(0) - panel["xint"].fillna(0)
    ) / panel["book_equity"]
    panel["accruals"] = (panel["ni"] - panel["oancf"]) / panel["at"]
    panel["leverage"] = (panel["dltt"].fillna(0) + panel["dlc"].fillna(0)) / panel["at"]
    panel["interest_coverage"] = panel["ebit"] / panel["xint"]
    panel["asset_turnover"] = panel["sale"] / panel["at"]
    panel["net_debt_issuance"] = (panel["dltis"].fillna(0) - panel["dltr"].fillna(0)) / panel["at"]

    panel["days_sales_outstanding"] = panel["rect"] / panel["sale"] * 365
    panel["days_inventory"] = panel["invt"] / panel["cogs"] * 365
    panel["days_payable"] = panel["ap"] / panel["cogs"] * 365
    panel["cash_conversion_cycle"] = (
        panel["days_sales_outstanding"] + panel["days_inventory"] - panel["days_payable"]
    )
    panel["goodwill_intangibles_pct"] = (panel["gdwl"].fillna(0) + panel["intan"].fillna(0)) / panel["at"]

    panel["efficiency_ratio"] = panel["xnitb"] / (panel["niint"] + panel["initb"])
    panel["capital_adequacy"] = panel["capr1"]
    panel["reserve_adequacy"] = panel["rvupi"] / panel["ipti"]

    apply_gics_exclusions(panel, QUALITY_GICS_EXCLUSIONS)
    return panel


GROWTH_FACTORS_ALL = [
    "sales_growth_yoy",
    "sales_growth_3y_cagr",
    "ni_growth_yoy",
    "ebitda_growth_yoy",
    "gp_growth_yoy",
    "oancf_growth_yoy",
    "fcf_growth_yoy",
    "book_equity_growth_yoy",
    "capx_growth_yoy",
    "niint_growth_yoy",
    "dptc_growth_yoy",
    "initb_growth_yoy",
    "premium_growth_yoy",
]
GROWTH_FACTORS = _live(GROWTH_FACTORS_ALL)

GROWTH_SIGNS = {f: 1 for f in GROWTH_FACTORS_ALL}
for _f in (
    "capx_growth_yoy",
    "book_equity_growth_yoy",
):
    GROWTH_SIGNS[_f] = -1

GROWTH_GICS_EXCLUSIONS = {"40": ["capx_growth_yoy"]}


def _yoy_growth(
    df: pd.DataFrame, col: str, shift: int = 1, min_gap: int = 300, max_gap: int = 430
) -> pd.Series:
    prev = df.groupby("gvkey")[col].shift(shift)
    prev_date = df.groupby("gvkey")["datadate"].shift(shift)
    gap = (df["datadate"] - prev_date).dt.days
    valid = gap.between(min_gap, max_gap) & (prev > 0)
    return (df[col] / prev - 1).where(valid)


# Runs on raw annual funda, before the merge_asof.
def compute_growth(funda: pd.DataFrame) -> pd.DataFrame:
    df = funda.sort_values(["gvkey", "datadate"]).reset_index(drop=True).copy()
    pref_stock = df["pstkrv"].fillna(df["pstkl"]).fillna(df["pstk"]).fillna(0)
    df["book_equity"] = df["seq"] + df["txditc"].fillna(0) - pref_stock
    df["fcf"] = df["oancf"] - df["capx"]

    df["sales_growth_yoy"] = _yoy_growth(df, "sale")
    cagr_total = _yoy_growth(df, "sale", shift=3, min_gap=950, max_gap=1160)
    df["sales_growth_3y_cagr"] = (1 + cagr_total) ** (1 / 3) - 1
    df["ni_growth_yoy"] = _yoy_growth(df, "ni")
    df["ebitda_growth_yoy"] = _yoy_growth(df, "ebitda")
    df["gp_growth_yoy"] = _yoy_growth(df, "gp")
    df["oancf_growth_yoy"] = _yoy_growth(df, "oancf")
    df["fcf_growth_yoy"] = _yoy_growth(df, "fcf")
    df["book_equity_growth_yoy"] = _yoy_growth(df, "book_equity")
    df["capx_growth_yoy"] = _yoy_growth(df, "capx")

    df["niint_growth_yoy"] = _yoy_growth(df, "niint")
    df["dptc_growth_yoy"] = _yoy_growth(df, "dptc")
    df["initb_growth_yoy"] = _yoy_growth(df, "initb")
    df["premium_growth_yoy"] = _yoy_growth(df, "ipti")
    return df


MOMENTUM_FACTORS_ALL = (
    [f"mom_{m}_1" for m in settings.LOOKBACKS["mom_skip_months"]]
    + [f"mom_{n}d" for n in settings.LOOKBACKS["mom_short_days"]]
)
MOMENTUM_FACTORS = _live(MOMENTUM_FACTORS_ALL)
MOMENTUM_SIGNS = {f: 1 for f in MOMENTUM_FACTORS_ALL}


# `prices` must be unfiltered: momentum shifts by row count.
def compute_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.sort_values(["permno", "date"]).reset_index(drop=True).copy()
    df["log_ret"] = np.log1p(df["ret"].astype("float64").fillna(0.0))
    df["cum_log_ret"] = df.groupby("permno")["log_ret"].cumsum()

    def _add(name, lookback, skip=0):
        grp = df.groupby("permno")["cum_log_ret"]
        end = grp.shift(skip) if skip else df["cum_log_ret"]
        start = grp.shift(skip + lookback)
        df[name] = np.exp(end - start) - 1

    month_days = settings.LOOKBACKS["mom_month_days"]
    skip = settings.LOOKBACKS["mom_skip_days"]
    for m in settings.LOOKBACKS["mom_skip_months"]:
        _add(f"mom_{m}_1", m * month_days, skip)
    for n in settings.LOOKBACKS["mom_short_days"]:
        _add(f"mom_{n}d", n)

    return df.drop(columns=["log_ret", "cum_log_ret"])


VOLATILITY_FACTORS_ALL = ["realised_vol_short", "realised_vol_long", "low_vol_annual", "beta", "max_dd"]
VOLATILITY_FACTORS = _live(VOLATILITY_FACTORS_ALL)
VOLATILITY_SIGNS = {
    "realised_vol_short": -1,
    "realised_vol_long": -1,
    "low_vol_annual": -1,
    "beta": -1,
    "max_dd": 1,
}


def market_proxy_return(point_in_time_panel: pd.DataFrame) -> pd.Series:
    df = point_in_time_panel.sort_values(["permno", "date"])
    # prior-day cap: same-day embeds that day's return
    w = df.groupby("permno")["mktcap"].shift(1).astype("float64")
    r = df["ret"].astype("float64")
    valid = w.notna() & r.notna()
    w, r, dates = w[valid], r[valid], df["date"][valid]
    return (((w * r).groupby(dates).sum()) / w.groupby(dates).sum()).rename("mkt_ret")


def compute_volatility(prices: pd.DataFrame, market_ret: pd.Series | None = None) -> pd.DataFrame:
    if market_ret is None:
        panel = pd.read_parquet(CACHE_DIR / "crsp_panel.parquet", columns=["permno", "date", "ret", "mktcap"])
        market_ret = market_proxy_return(panel)

    df = prices.sort_values(["permno", "date"]).reset_index(drop=True).copy()
    df["ret"] = df["ret"].astype("float64")
    df["close"] = df["close"].abs().astype("float64")
    df = df.merge(market_ret, on="date", how="left")

    grp_ret = df.groupby("permno")["ret"]
    for name in ("realised_vol_short", "realised_vol_long", "low_vol_annual"):
        df[name] = grp_ret.rolling(settings.LOOKBACKS[name]).std().reset_index(level=0, drop=True) * np.sqrt(252)

    n = settings.LOOKBACKS["beta"]
    df["xy"] = df["ret"] * df["mkt_ret"]
    df["y2"] = df["mkt_ret"] ** 2
    roll = lambda col: df.groupby("permno")[col].rolling(n).sum().reset_index(level=0, drop=True)
    sum_x, sum_y, sum_xy, sum_y2 = roll("ret"), roll("mkt_ret"), roll("xy"), roll("y2")
    cov = sum_xy / n - (sum_x / n) * (sum_y / n)
    var_mkt = sum_y2 / n - (sum_y / n) ** 2
    df["beta"] = cov / var_mkt
    df = df.drop(columns=["xy", "y2"])

    n = settings.LOOKBACKS["max_dd"]
    rolling_high = df.groupby("permno")["close"].rolling(n).max().reset_index(level=0, drop=True)
    df["max_dd"] = df["close"] / rolling_high - 1

    return df


FAMILIES = {
    "value": (VALUE_FACTORS, VALUE_SIGNS),
    "quality": (QUALITY_FACTORS, QUALITY_SIGNS),
    "growth": (GROWTH_FACTORS, GROWTH_SIGNS),
    "momentum": (MOMENTUM_FACTORS, MOMENTUM_SIGNS),
    "volatility": (VOLATILITY_FACTORS, VOLATILITY_SIGNS),
}

ALL_FACTORS = [f for flist, _ in FAMILIES.values() for f in flist]


def _validate_weights() -> None:
    if abs(sum(settings.FACTOR_WEIGHTS.values()) - 1.0) > 1e-6:
        raise ValueError(
            f"settings.FACTOR_WEIGHTS must sum to ~1.0, got {sum(settings.FACTOR_WEIGHTS.values())}"
        )
    if set(settings.FACTOR_WEIGHTS) != set(FAMILIES):
        raise ValueError(
            f"settings.FACTOR_WEIGHTS keys {set(settings.FACTOR_WEIGHTS)} must exactly match "
            f"the 5 built families {set(FAMILIES)}"
        )


def _daily_align_fundamentals(
    funda: pd.DataFrame, daily_spine: pd.DataFrame, funda_factors: list[str]
) -> pd.DataFrame:
    events = (
        funda.dropna(subset=["available_date"])
        .sort_values("available_date")
        .drop_duplicates(subset=["permno", "available_date"], keep="last")
    )
    events_sorted = events[["permno", "available_date"] + funda_factors].sort_values("available_date")
    spine_sorted = daily_spine.sort_values("date")

    merged = pd.merge_asof(
        spine_sorted,
        events_sorted,
        by="permno",
        left_on="date",
        right_on="available_date",
        direction="backward",
    )
    staleness_days = (merged["date"] - merged["available_date"]).dt.days
    merged.loc[staleness_days > settings.MAX_FUNDAMENTALS_STALENESS_DAYS, funda_factors] = np.nan
    return merged


def _expand_schedule(schedule: pd.DataFrame, dates) -> pd.DataFrame:
    return schedule.sort_index().reindex(pd.DatetimeIndex(sorted(pd.unique(dates))), method="ffill")


def _validate_schedule(schedule: pd.DataFrame, expected_cols, label: str) -> None:
    if not isinstance(schedule.index, pd.DatetimeIndex):
        raise ValueError(f"{label} must be indexed by date (DatetimeIndex)")
    extra = set(schedule.columns) - set(expected_cols)
    if extra:
        raise ValueError(f"{label} has unexpected columns {extra}, expected a subset of {set(expected_cols)}")


def compute_combined_score(
    funda: pd.DataFrame | None = None,
    mom: pd.DataFrame | None = None,
    vol: pd.DataFrame | None = None,
    crsp: pd.DataFrame | None = None,
    factor_weight_schedule: pd.DataFrame | None = None,
    within_factor_weight_schedule: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    _validate_weights()

    funda = funda if funda is not None else pd.read_parquet(CACHE_DIR / "fundamentals_panel.parquet")
    mom = mom if mom is not None else pd.read_parquet(CACHE_DIR / "momentum_panel.parquet")
    vol = vol if vol is not None else pd.read_parquet(CACHE_DIR / "volatility_panel.parquet")
    crsp = crsp if crsp is not None else pd.read_parquet(CACHE_DIR / "crsp_panel.parquet")

    funda_factors = VALUE_FACTORS + QUALITY_FACTORS + GROWTH_FACTORS
    funda_daily = _daily_align_fundamentals(funda, mom[["permno", "date"]], funda_factors)

    panel = mom[["permno", "date", "gvkey", "gsector", "sector_name", "ticker"] + MOMENTUM_FACTORS].copy()
    panel = panel.merge(vol[["permno", "date"] + VOLATILITY_FACTORS], on=["permno", "date"], how="left")
    panel = panel.merge(crsp[["permno", "date", "mktcap"]], on=["permno", "date"], how="left")
    panel = panel.merge(funda_daily[["permno", "date"] + funda_factors], on=["permno", "date"], how="left")

    for family, (flist, signs) in FAMILIES.items():
        for f in flist:
            panel[f"pct_{f}"] = sector_neutral_percentile(panel, f, sign=signs[f])

    for family, (flist, _) in FAMILIES.items():
        within_schedule = None if within_factor_weight_schedule is None else within_factor_weight_schedule.get(family)
        if within_schedule is None:
            within_weights = {
                f"pct_{f}": settings.WITHIN_FACTOR_WEIGHT_OVERRIDES.get(family, {}).get(f, 1.0) for f in flist
            }
            panel[f"{family}_score"] = weighted_combine(panel, within_weights)
        else:
            _validate_schedule(within_schedule, flist, f"within_factor_weight_schedule['{family}']")
            wcols = [f"_wf_{f}" for f in flist]
            within_schedule = _expand_schedule(within_schedule, panel["date"])
            panel = panel.merge(
                within_schedule.rename(columns={f: f"_wf_{f}" for f in flist}),
                left_on="date",
                right_index=True,
                how="left",
            )
            for f, wcol in zip(flist, wcols):
                baseline = settings.WITHIN_FACTOR_WEIGHT_OVERRIDES.get(family, {}).get(f, 1.0)
                panel[wcol] = panel[wcol].fillna(baseline) if wcol in panel else baseline
            panel[f"{family}_score"] = weighted_combine_dynamic(panel, [f"pct_{f}" for f in flist], wcols)
            panel = panel.drop(columns=wcols)

        # same [0,1] scale before families are weighted together
        panel[f"{family}_score"] = cross_sectional_percentile(panel, f"{family}_score")

    if factor_weight_schedule is None:
        family_score_weights = {f"{family}_score": w for family, w in settings.FACTOR_WEIGHTS.items()}
        panel["combined_score"] = weighted_combine(panel, family_score_weights)
    else:
        _validate_schedule(factor_weight_schedule, FAMILIES.keys(), "factor_weight_schedule")
        wcols = [f"_wfam_{fam}" for fam in FAMILIES]
        factor_weight_schedule = _expand_schedule(factor_weight_schedule, panel["date"])
        panel = panel.merge(
            factor_weight_schedule.rename(columns={fam: f"_wfam_{fam}" for fam in FAMILIES}),
            left_on="date",
            right_index=True,
            how="left",
        )
        for fam, wcol in zip(FAMILIES, wcols):
            panel[wcol] = (
                panel[wcol].fillna(settings.FACTOR_WEIGHTS[fam]) if wcol in panel else settings.FACTOR_WEIGHTS[fam]
            )
        panel["combined_score"] = weighted_combine_dynamic(panel, [f"{fam}_score" for fam in FAMILIES], wcols)
        panel = panel.drop(columns=wcols)

    return panel


if __name__ == "__main__":
    print(f"Live factors: {len(ALL_FACTORS)} (disabled: {sorted(settings.DISABLED_FACTORS) or 'none'})")
    for fam, (flist, signs) in FAMILIES.items():
        neg = [f for f in flist if signs[f] == -1]
        print(f"  {fam:11s} {len(flist):2d} factors, sign -1: {neg or '-'}")

    panel = compute_combined_score()
    print(f"\ncombined_score coverage: {panel['combined_score'].notna().mean() * 100:.1f}%")
    print(f"panel: {len(panel):,} rows, {panel['date'].nunique():,} dates")
    for fam in FAMILIES:
        s = panel[f"{fam}_score"]
        print(f"  {fam:11s} score coverage {s.notna().mean() * 100:5.1f}%  mean {s.mean():.4f}")
