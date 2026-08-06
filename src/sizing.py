from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from src.backtest.backtester import MarketEvent, OrderType, SignalEvent
from src.backtest.portfolio import PortfolioTracker
from src import settings


class StockPicker:

    def __init__(
        self,
        n: int | None = None,
        buffer_rank: int | None = None,
        buffer_days: int | None = None,
        rebalance_interval: int | None = None,
        entry_trigger_rank: int | None = None,
    ):
        self.n = n if n is not None else settings.SELECTION_N
        self.buffer_rank = buffer_rank if buffer_rank is not None else settings.SELECTION_BUFFER_RANK
        self.buffer_days = buffer_days if buffer_days is not None else settings.SELECTION_BUFFER_DAYS
        self.rebalance_interval = (
            rebalance_interval if rebalance_interval is not None else settings.REBALANCE_INTERVAL_DAYS
        )
        self.entry_trigger_rank = (
            entry_trigger_rank if entry_trigger_rank is not None else settings.ENTRY_TRIGGER_RANK
        )
        self.holdings: set = set()
        self._days_outside: dict = {}
        self._days_inside: dict = {}
        self._days_since_rebalance = 0
        self.rebalance_dates: list = []

    def step(self, date, cross_section: pd.DataFrame) -> pd.DataFrame:
        df = cross_section.dropna(subset=["combined_score"]).sort_values("permno").copy()
        df["rank"] = df["combined_score"].rank(ascending=False, method="first")

        scores = dict(zip(df["permno"], df["combined_score"]))
        rank = dict(zip(df["permno"], df["rank"]))

        # missing score = forced exit (survivorship guard)
        delisted = {p for p in self.holdings if p not in rank}
        if delisted:
            self.holdings -= delisted
            for p in delisted:
                self._days_outside.pop(p, None)

        exit_breach = False
        for p in self.holdings:
            if rank[p] <= self.buffer_rank:
                self._days_outside[p] = 0
            else:
                self._days_outside[p] = self._days_outside.get(p, 0) + 1
                if self._days_outside[p] >= self.buffer_days:
                    exit_breach = True

        entry_breach = False
        for p, r in rank.items():
            if p in self.holdings:
                self._days_inside.pop(p, None)
                continue
            if r <= self.entry_trigger_rank:
                self._days_inside[p] = self._days_inside.get(p, 0) + 1
                if self._days_inside[p] >= self.buffer_days:
                    entry_breach = True
            else:
                self._days_inside.pop(p, None)

        # fixed clock, interrupted by either breach
        due = (
            not self.holdings
            or self._days_since_rebalance >= self.rebalance_interval
            or exit_breach
            or entry_breach
            or bool(delisted)
        )

        if due:
            pool = {p for p in self.holdings if self._days_outside.get(p, 0) < self.buffer_days}
            pool |= {p for p, d in self._days_inside.items() if d >= self.buffer_days}
            for p in df.sort_values("rank")["permno"]:
                if len(pool) >= self.n:
                    break
                pool.add(p)
            if len(pool) > self.n:
                pool = set(sorted(pool, key=lambda p: rank[p])[: self.n])
            self.holdings = pool
            self._days_outside = {p: d for p, d in self._days_outside.items() if p in pool}
            self._days_inside = {}
            self._days_since_rebalance = 0
            self.rebalance_dates.append(date)
        else:
            self._days_since_rebalance += 1

        rows = [
            {"permno": p, "combined_score": scores[p], "rank": rank[p]}
            for p in self.holdings
            if p in rank
        ]
        return pd.DataFrame(rows, columns=["permno", "combined_score", "rank"])


def run_selection(
    combined_panel: pd.DataFrame, start=None, end=None, picker: StockPicker | None = None
) -> pd.DataFrame:
    df = combined_panel[["permno", "date", "combined_score"]]
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]

    picker = picker if picker is not None else StockPicker()
    frames = []
    for date, cross_section in df.sort_values("date").groupby("date", sort=True):
        holdings = picker.step(date, cross_section[["permno", "combined_score"]])
        if holdings.empty:
            continue
        holdings["date"] = date
        frames.append(holdings)

    if not frames:
        return pd.DataFrame(columns=["permno", "combined_score", "rank", "date"])
    return pd.concat(frames, ignore_index=True)


class RollingVolatilityTracker:

    def __init__(self, window: int | None = None):
        self.window = window if window is not None else settings.ROLLING_VOL_WINDOW
        self._returns: dict = {}

    def vol(self, key) -> float:
        rets = self._returns.get(key)
        if rets is None or len(rets) < self.window:
            return float("nan")
        return float(np.std(rets, ddof=1))

    def observe(self, key, ret: float) -> None:
        window = self._returns.setdefault(key, deque(maxlen=self.window))
        window.append(ret)


def compute_weights(holdings: pd.DataFrame, vol_tracker: RollingVolatilityTracker) -> pd.DataFrame:
    df = holdings.copy()
    df["vol"] = df["permno"].astype(str).map(vol_tracker.vol)
    df["raw_weight"] = df["combined_score"] / df["vol"]

    weight = pd.Series(0.0, index=df.index, dtype="float64")
    mask = df["raw_weight"].notna() & (df["raw_weight"] > 0)
    raw = df.loc[mask, "raw_weight"]
    if not raw.empty:
        denom = raw.sum()
        if denom != 0 and np.isfinite(denom):
            weight.loc[mask] = raw / denom

    df["weight"] = weight
    return df


def apply_weight_cap(
    weight: pd.Series, cap: float | None = None, multiple: float | None = None
) -> pd.Series:
    if cap is None:
        multiple = multiple if multiple is not None else settings.MAX_WEIGHT_MULTIPLE_OF_EQUAL
        n = len(weight)
        cap = multiple / n if n else 1.0
    w = weight.astype("float64").copy()
    total = w.sum()
    if total <= 0 or cap <= 0:
        return w

    capped = pd.Series(False, index=w.index)
    for _ in range(len(w)):
        over = (w > cap) & ~capped
        if not over.any():
            break
        w.loc[over] = cap
        capped.loc[over] = True
        remaining_mask = ~capped
        remaining_total = w.loc[remaining_mask].sum()
        target_remaining = total - cap * capped.sum()
        if remaining_total > 0 and target_remaining > 0:
            w.loc[remaining_mask] = w.loc[remaining_mask] / remaining_total * target_remaining
        else:
            break
    return w


def assert_no_lookahead() -> None:
    rng = np.random.default_rng(7)
    n_days, t = 30, 25
    permnos = ["AAA", "BBB"]
    returns = {p: rng.normal(0, 0.02, n_days) for p in permnos}

    def weights_on_day_t(rets: dict) -> pd.Series:
        tracker = RollingVolatilityTracker()
        for day in range(t):
            for p in permnos:
                tracker.observe(p, rets[p][day])
        holdings = pd.DataFrame({"permno": permnos, "combined_score": [0.95, 0.80]})
        return compute_weights(holdings, tracker).set_index("permno")["weight"].sort_index()

    baseline = weights_on_day_t(returns)

    perturbed = {p: r.copy() for p, r in returns.items()}
    for p in permnos:
        perturbed[p][t:] = perturbed[p][t:] * 50 + 10

    pd.testing.assert_series_equal(baseline, weights_on_day_t(perturbed))
    print("PASS: compute_weights on day t is unaffected by day-t-and-later returns.")


def warm_up_vol_tracker(tracker: RollingVolatilityTracker, raw_daily: pd.DataFrame, permnos, end_date) -> None:
    df = raw_daily[raw_daily["permno"].isin(set(permnos)) & (raw_daily["date"] < pd.Timestamp(end_date))]
    df = df.sort_values(["permno", "date"])
    for permno, ret in zip(df["permno"], df["ret"].astype("float64")):
        if ret == ret:
            tracker.observe(str(permno), ret)


class PortfolioVolTracker:

    _KEY = "portfolio"

    def __init__(self, window: int | None = None):
        self._tracker = RollingVolatilityTracker(window=window)

    def current_vol(self) -> float:
        return self._tracker.vol(self._KEY)

    def observe_return(self, ret: float) -> None:
        self._tracker.observe(self._KEY, ret)


# De-risk only: capped at 1.0, never levers up.
class VolatilityTargetOverlay:

    def __init__(
        self,
        target_vol: float | None = None,
        bounds: tuple[float, float] | None = None,
        window: int | None = None,
    ):
        self.target_vol = target_vol if target_vol is not None else settings.TARGET_VOL
        self.bounds = bounds if bounds is not None else settings.VOL_TARGET_BOUNDS
        self.vol_tracker = PortfolioVolTracker(
            window=window if window is not None else settings.VOL_TARGET_WINDOW_DAYS
        )
        self.current_multiplier = 1.0

    def step(self, ret_t: float) -> float:
        self.vol_tracker.observe_return(ret_t)
        daily_vol = self.vol_tracker.current_vol()
        if daily_vol is None or not np.isfinite(daily_vol) or daily_vol <= 0:
            self.current_multiplier = 1.0
        else:
            annualized = daily_vol * np.sqrt(252.0)
            lo, hi = self.bounds
            self.current_multiplier = float(np.clip(self.target_vol / annualized, lo, hi))
        return self.current_multiplier


class DrawdownOverlay:

    def __init__(
        self,
        tiers: list[tuple[float, float]] | None = None,
        confirmation_days: int | None = None,
        reentry_vol_ratio: float | None = None,
    ):
        self.tiers = list(tiers) if tiers is not None else list(settings.DRAWDOWN_TIERS)
        # floored at 1: 0 would fire every tier daily
        self.confirmation_days = max(
            1, confirmation_days if confirmation_days is not None else settings.DRAWDOWN_CONFIRMATION_DAYS
        )
        self.reentry_vol_ratio = (
            reentry_vol_ratio if reentry_vol_ratio is not None else settings.DRAWDOWN_REENTRY_VOL_RATIO
        )

        self._peak: float | None = None
        self.active_tier: int | None = None
        self._breach_days = [0] * len(self.tiers)
        self._reentry_days = 0
        self._fire_vol: dict[int, float] = {}
        self._deepest_reached: int | None = None

    @property
    def current_multiplier(self) -> float:
        return 1.0 if self.active_tier is None else self.tiers[self.active_tier][1]

    def step(self, nav_t: float, vol_t: float) -> float:
        self._peak = nav_t if self._peak is None else max(self._peak, nav_t)
        drawdown = nav_t / self._peak - 1

        deepest_confirmed = None
        for i, (threshold, _mult) in enumerate(self.tiers):
            if drawdown <= threshold:
                self._breach_days[i] += 1
            else:
                self._breach_days[i] = 0
            if self._breach_days[i] >= self.confirmation_days:
                deepest_confirmed = i

        escalated = False
        if deepest_confirmed is not None and (self._deepest_reached is None or deepest_confirmed > self._deepest_reached):
            self.active_tier = deepest_confirmed
            self._deepest_reached = deepest_confirmed
            self._fire_vol[self.active_tier] = vol_t
            self._reentry_days = 0
            escalated = True

        if not escalated and self.active_tier is not None:
            baseline = self._fire_vol.get(self.active_tier, vol_t)
            # x == x is a NaN guard
            if vol_t == vol_t and baseline == baseline and vol_t <= self.reentry_vol_ratio * baseline:
                self._reentry_days += 1
            else:
                self._reentry_days = 0

            if self._reentry_days >= self.confirmation_days:
                self.active_tier = self.active_tier - 1 if self.active_tier > 0 else None
                self._reentry_days = 0
                if self.active_tier is not None:
                    self._fire_vol[self.active_tier] = vol_t
                else:
                    self._deepest_reached = None

        return self.current_multiplier


class SizedLongOnlyStrategy:

    def __init__(
        self,
        combined_panel: pd.DataFrame,
        enabled: bool = True,
        vol_tracker: RollingVolatilityTracker | None = None,
        picker: StockPicker | None = None,
        drawdown_overlay: DrawdownOverlay | None = None,
        portfolio_vol: PortfolioVolTracker | None = None,
        vol_target_overlay: VolatilityTargetOverlay | None = None,
        position_no_trade_band: float | None = None,
        max_weight_multiple: float | None = None,
    ):
        self.enabled = enabled
        self.picker = picker if picker is not None else StockPicker()
        self.vol_tracker = vol_tracker if vol_tracker is not None else RollingVolatilityTracker()
        self.drawdown_overlay = drawdown_overlay if drawdown_overlay is not None else DrawdownOverlay()
        self.portfolio_vol = portfolio_vol if portfolio_vol is not None else PortfolioVolTracker()
        self.vol_target_overlay = (
            vol_target_overlay if vol_target_overlay is not None else VolatilityTargetOverlay()
        )
        self.overlay_history: list[tuple] = []
        self.position_no_trade_band = (
            position_no_trade_band if position_no_trade_band is not None else settings.POSITION_NO_TRADE_BAND
        )
        self.max_weight_multiple = (
            max_weight_multiple if max_weight_multiple is not None else settings.MAX_WEIGHT_MULTIPLE_OF_EQUAL
        )

        self._combined_by_date = {ts: df for ts, df in combined_panel.groupby("date")}
        self._last_date = None
        self._last_close: dict[str, float] = {}
        self._equity_ptr = 0

    def generate_signals(
        self, bar: MarketEvent, latest_bars: dict[str, MarketEvent], portfolio: PortfolioTracker
    ) -> list[SignalEvent]:
        if bar.timestamp == self._last_date:
            return []
        today = bar.timestamp
        self._last_date = today

        self._process_new_equity_entries(portfolio)
        signals = self._rebalance(today, latest_bars, portfolio)
        self._update_vol_tracker(today, latest_bars)
        return signals

    def _process_new_equity_entries(self, portfolio: PortfolioTracker) -> None:
        curve = portfolio.equity_curve
        while self._equity_ptr < len(curve):
            idx = self._equity_ptr
            self._equity_ptr += 1
            if idx == 0:
                continue
            prev_equity = curve[idx - 1][1]
            _, curr_equity = curve[idx]
            if not self.enabled or prev_equity == 0:
                continue
            port_ret = curr_equity / prev_equity - 1
            self.portfolio_vol.observe_return(port_ret)
            self.drawdown_overlay.step(curr_equity, self.portfolio_vol.current_vol())
            self.vol_target_overlay.step(port_ret)

    def _rebalance(self, today, latest_bars: dict[str, MarketEvent], portfolio: PortfolioTracker) -> list[SignalEvent]:
        cross_section = self._combined_by_date.get(today)
        if cross_section is None:
            return []

        holdings = self.picker.step(today, cross_section[["permno", "combined_score"]])
        if holdings.empty:
            return []

        if self.enabled:
            sized = compute_weights(holdings, self.vol_tracker)
            drawdown_mult = self.drawdown_overlay.current_multiplier
            vol_target_mult = self.vol_target_overlay.current_multiplier
        else:
            sized = holdings.copy()
            n = len(sized)
            sized["weight"] = 1.0 / n if n else 0.0
            drawdown_mult = 1.0
            vol_target_mult = 1.0

        self.overlay_history.append((today, drawdown_mult, vol_target_mult))
        sized["weight"] = apply_weight_cap(sized["weight"], multiple=self.max_weight_multiple)
        sized["final_weight"] = sized["weight"] * drawdown_mult * vol_target_mult
        nav = portfolio.equity_curve[-1][1] if portfolio.equity_curve else portfolio.starting_cash

        signals: list[SignalEvent] = []
        today_symbols = {str(p) for p in sized["permno"]}
        for symbol in set(portfolio.positions.keys()) - today_symbols:
            if portfolio.position_quantity(symbol) != 0:
                signals.append(SignalEvent(timestamp=today, symbol=symbol, target_position=0, order_type=OrderType.MARKET))

        for row in sized.itertuples(index=False):
            symbol = str(row.permno)
            price_bar = latest_bars.get(symbol)
            if price_bar is None or price_bar.close <= 0:
                continue
            if self.position_no_trade_band > 0 and nav > 0:
                current_weight = portfolio.position_quantity(symbol) * price_bar.close / nav
                if current_weight != 0 and abs(row.final_weight - current_weight) < self.position_no_trade_band:
                    continue
            target_shares = round(nav * row.final_weight / price_bar.close)
            signals.append(
                SignalEvent(timestamp=today, symbol=symbol, target_position=target_shares, order_type=OrderType.MARKET)
            )
        return signals

    def _update_vol_tracker(self, today, latest_bars: dict[str, MarketEvent]) -> None:
        for symbol, b in latest_bars.items():
            if b.timestamp != today:
                continue
            prev_close = self._last_close.get(symbol)
            if self.enabled and prev_close is not None and prev_close > 0:
                self.vol_tracker.observe(symbol, b.close / prev_close - 1)
            self._last_close[symbol] = b.close


if __name__ == "__main__":
    assert_no_lookahead()
