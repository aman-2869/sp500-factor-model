from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Protocol

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.backtest.portfolio import PortfolioTracker
from src import settings

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    TWAP = "TWAP"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(slots=True)
class MarketEvent:

    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float | None = None
    ask: float | None = None
    rolling_volatility_20d: float | None = None
    rolling_adv_20d: float | None = None
    dividend_per_share: float = 0.0


@dataclass(slots=True)
class SignalEvent:

    timestamp: datetime
    symbol: str
    target_position: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    strategy_id: str = "default"
    strength: float = 1.0


@dataclass(slots=True)
class OrderEvent:
    order_id: int
    created_at: datetime
    eligible_at: datetime
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    remaining_quantity: float
    decision_price: float
    limit_price: float | None = None
    twap_slices_remaining: int = 1


@dataclass(slots=True)
class FillEvent:
    timestamp: datetime
    order_id: int
    symbol: str
    side: OrderSide
    quantity: float
    fill_price: float
    decision_price: float
    cost: float

    @property
    def notional(self) -> float:
        return self.quantity * self.fill_price

    @property
    def implementation_shortfall_bps(self) -> float:
        sign = 1 if self.side == OrderSide.BUY else -1
        if self.decision_price == 0:
            return float("nan")
        return sign * (self.fill_price - self.decision_price) / self.decision_price * 10_000


def load_market_frame(
    permnos: list[int] | None = None,
    start: str | None = None,
    end: str | None = None,
    vol_window: int = 20,
    adv_window: int = 20,
) -> pd.DataFrame:
    df = pd.read_parquet(CACHE_DIR / "crsp_panel.parquet")
    if permnos is not None:
        df = df[df["permno"].isin(permnos)]
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]

    df = df.sort_values(["permno", "date"]).reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume", "ret", "retx"]:
        df[col] = df[col].astype("float64")

    prev_close = df["close"].abs() / (1.0 + df["retx"])
    df["dividend_per_share"] = (prev_close * (df["ret"] - df["retx"])).fillna(0.0)
    df.loc[~np.isfinite(df["dividend_per_share"]), "dividend_per_share"] = 0.0

    grp = df.groupby("permno")
    df["rolling_volatility_20d"] = grp["ret"].transform(lambda s: s.rolling(vol_window).std())
    df["rolling_adv_20d"] = grp["volume"].transform(lambda s: s.rolling(adv_window).mean())

    return df.sort_values(["date", "permno"]).reset_index(drop=True)


def iter_market_events(df: pd.DataFrame):
    for row in df.itertuples(index=False):
        yield MarketEvent(
            timestamp=row.date,
            symbol=str(row.permno),
            open=row.open,
            high=row.high,
            low=row.low,
            close=row.close,
            volume=row.volume,
            bid=None,
            ask=None,
            rolling_volatility_20d=row.rolling_volatility_20d if not np.isnan(row.rolling_volatility_20d) else None,
            rolling_adv_20d=row.rolling_adv_20d if not np.isnan(row.rolling_adv_20d) else None,
            dividend_per_share=getattr(row, "dividend_per_share", 0.0),
        )


def load_market_events(
    permnos: list[int] | None = None,
    start: str | None = None,
    end: str | None = None,
    vol_window: int = 20,
    adv_window: int = 20,
):
    df = load_market_frame(permnos, start, end, vol_window, adv_window)
    return iter_market_events(df)


def load_trading_calendar(
    permnos: list[int] | None = None, start: str | None = None, end: str | None = None
) -> list:
    df = pd.read_parquet(CACHE_DIR / "crsp_panel.parquet", columns=["permno", "date"])
    if permnos is not None:
        df = df[df["permno"].isin(permnos)]
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df["date"].drop_duplicates().sort_values().tolist()


@dataclass
class FlatCostConfig:
    cost_bps: float | None = None

    def resolved_bps(self) -> float:
        return self.cost_bps if self.cost_bps is not None else settings.FLAT_COST_BPS


class FlatCostExecutionSimulator:
    def __init__(self, config: FlatCostConfig | None = None, cost_bps: float | None = None):
        self.config = config or FlatCostConfig(cost_bps=cost_bps)

    def attempt_fill(self, order: OrderEvent, bar: MarketEvent) -> FillEvent | None:
        if bar.symbol != order.symbol or bar.timestamp < order.eligible_at:
            return None
        if order.remaining_quantity <= 0 or bar.close <= 0:
            return None

        fill_qty = order.remaining_quantity
        fill_price = bar.close
        cost = fill_qty * fill_price * self.config.resolved_bps() / 10_000
        order.remaining_quantity = 0.0

        return FillEvent(
            timestamp=bar.timestamp,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_qty,
            fill_price=fill_price,
            decision_price=order.decision_price,
            cost=cost,
        )


class Strategy(Protocol):
    def generate_signals(
        self, bar: MarketEvent, latest_bars: dict[str, MarketEvent], portfolio: PortfolioTracker
    ) -> list[SignalEvent]: ...


class ExecutionEngine:
    def __init__(
        self,
        tca: FlatCostExecutionSimulator,
        portfolio: PortfolioTracker,
        strategy: Strategy,
        latency_bars: int = 1,
        twap_slices: int = 5,
    ):
        self.tca = tca
        self.portfolio = portfolio
        self.strategy = strategy
        self.latency_bars = max(latency_bars, 1)
        self.twap_slices = max(twap_slices, 1)

        self._order_id_counter = itertools.count(1)
        self.open_orders: dict[str, list[OrderEvent]] = {}
        self.fill_log: list = []
        self.order_log: list[OrderEvent] = []
        self._latest_bar: dict[str, MarketEvent] = {}
        self._bar_timestamps: list = []
        self._timestamp_index: dict = {}

    def run(self, bars: Iterable[MarketEvent], all_timestamps: Iterable | None = None) -> None:
        if all_timestamps is not None:
            self._bar_timestamps = list(all_timestamps)
        else:
            bars = list(bars)
            self._bar_timestamps = sorted({b.timestamp for b in bars})
        self._timestamp_index = {t: i for i, t in enumerate(self._bar_timestamps)}

        last_ts = None
        for ts, todays_bars_iter in itertools.groupby(bars, key=lambda b: b.timestamp):
            if last_ts is not None and ts < last_ts:
                raise ValueError(
                    f"ExecutionEngine.run() requires `bars` sorted by timestamp - got {ts} after "
                    f"{last_ts}. Use load_market_frame()/iter_market_events(), which guarantee this."
                )
            last_ts = ts
            todays_bars = list(todays_bars_iter)

            for bar in todays_bars:
                if bar.dividend_per_share:
                    qty = self.portfolio.position_quantity(bar.symbol)
                    if qty:
                        self.portfolio.cash += qty * bar.dividend_per_share
            for bar in todays_bars:
                self._latest_bar[bar.symbol] = bar
                self._try_fill_open_orders(bar)

            signals: list[SignalEvent] = []
            for bar in todays_bars:
                signals.extend(self.strategy.generate_signals(bar, self._latest_bar, self.portfolio))
            for sig in signals:
                order = self._signal_to_order(sig)
                if order is not None:
                    self.open_orders.setdefault(order.symbol, []).append(order)
                    self.order_log.append(order)

            prices = {s: bar.close for s, bar in self._latest_bar.items()}
            self.portfolio.mark_to_market(ts, prices)

    def _eligible_at(self, signal_ts):
        idx = self._timestamp_index.get(signal_ts)
        if idx is None:
            return signal_ts
        target_idx = min(idx + self.latency_bars, len(self._bar_timestamps) - 1)
        return self._bar_timestamps[target_idx]

    def _signal_to_order(self, signal: SignalEvent) -> OrderEvent | None:
        bar = self._latest_bar.get(signal.symbol)
        if bar is None:
            return None
        current_qty = self.portfolio.position_quantity(signal.symbol)
        pending = sum(
            (o.remaining_quantity if o.side == OrderSide.BUY else -o.remaining_quantity)
            for o in self.open_orders.get(signal.symbol, [])
        )
        delta = signal.target_position - (current_qty + pending)
        if abs(delta) < 1e-9:
            return None

        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        return OrderEvent(
            order_id=next(self._order_id_counter),
            created_at=signal.timestamp,
            eligible_at=self._eligible_at(signal.timestamp),
            symbol=signal.symbol,
            side=side,
            order_type=signal.order_type,
            quantity=abs(delta),
            remaining_quantity=abs(delta),
            decision_price=bar.close,
            limit_price=signal.limit_price,
            twap_slices_remaining=self.twap_slices if signal.order_type == OrderType.TWAP else 1,
        )

    def _try_fill_open_orders(self, bar: MarketEvent) -> None:
        orders = self.open_orders.get(bar.symbol)
        if not orders:
            return
        still_open = []
        for order in orders:
            if bar.timestamp < order.eligible_at:
                still_open.append(order)
                continue
            fill = self.tca.attempt_fill(order, bar)
            if fill is not None:
                self.portfolio.on_fill(fill)
                self.fill_log.append(fill)
            if order.remaining_quantity > 1e-9:
                still_open.append(order)
        if still_open:
            self.open_orders[bar.symbol] = still_open
        else:
            del self.open_orders[bar.symbol]


@dataclass
class PerformanceReport:
    gross_sharpe: float
    net_sharpe: float
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_loss_ratio: float
    profit_factor: float
    turnover: float
    total_costs: float
    total_traded_notional: float
    n_fills: int


class TCAReporter:
    def __init__(self, periods_per_year: int = 252, annual_risk_free_rate: float = 0.0):
        self.periods_per_year = periods_per_year
        self.annual_risk_free_rate = annual_risk_free_rate

    def _equity_series(self, equity_curve: list[tuple]) -> pd.Series:
        if not equity_curve:
            return pd.Series(dtype=float)
        df = pd.DataFrame(equity_curve, columns=["timestamp", "equity"]).drop_duplicates(
            "timestamp", keep="last"
        )
        return df.set_index("timestamp")["equity"].sort_index()

    def _daily_risk_free(self) -> float:
        return (1 + self.annual_risk_free_rate) ** (1 / self.periods_per_year) - 1

    def sharpe(self, returns: pd.Series) -> float:
        if len(returns) < 2 or returns.std(ddof=0) == 0:
            return float("nan")
        excess = returns - self._daily_risk_free()
        return excess.mean() / returns.std(ddof=0) * np.sqrt(self.periods_per_year)

    def volatility(self, returns: pd.Series) -> float:
        if len(returns) < 2:
            return float("nan")
        return returns.std(ddof=0) * np.sqrt(self.periods_per_year)

    def sortino(self, returns: pd.Series) -> float:
        if len(returns) < 2:
            return float("nan")
        excess = returns - self._daily_risk_free()
        downside = excess.clip(upper=0.0)
        downside_deviation = np.sqrt((downside**2).sum() / len(excess))
        if downside_deviation == 0:
            return float("nan")
        return excess.mean() / downside_deviation * np.sqrt(self.periods_per_year)

    def max_drawdown(self, equity: pd.Series) -> float:
        if equity.empty:
            return float("nan")
        running_max = equity.cummax()
        return (equity / running_max - 1).min()

    def generate(self, portfolio: PortfolioTracker) -> PerformanceReport:
        equity = self._equity_series(portfolio.equity_curve)
        if equity.empty:
            return PerformanceReport(*([float("nan")] * 13), n_fills=0)
        net_returns = equity.pct_change().dropna()

        trade_log = pd.DataFrame(portfolio.trade_log)
        if not trade_log.empty:
            daily_cost_total = trade_log.groupby("timestamp")["cost"].sum().reindex(equity.index, fill_value=0.0)
        else:
            daily_cost_total = pd.Series(0.0, index=equity.index)

        gross_equity = equity + daily_cost_total.cumsum()
        gross_returns = gross_equity.pct_change().dropna()

        total_return = equity.iloc[-1] / portfolio.starting_cash - 1
        n_periods = max(len(equity) - 1, 1)
        n_years = max(n_periods / self.periods_per_year, 1e-9)
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
        annualized_volatility = self.volatility(net_returns)
        mdd = self.max_drawdown(equity)
        calmar = annualized_return / abs(mdd) if mdd and not np.isnan(mdd) and mdd != 0 else float("nan")

        realized = trade_log["realized_pnl"] if not trade_log.empty else pd.Series(dtype=float)
        realized = realized[realized != 0]
        wins = realized[realized > 0]
        losses = realized[realized < 0]
        win_loss_ratio = (wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else float("nan")
        profit_factor = (wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float("nan")

        total_traded_notional = (
            (trade_log["quantity"] * trade_log["fill_price"]).sum() if not trade_log.empty else 0.0
        )
        turnover = total_traded_notional / max(equity.mean(), 1e-9)

        return PerformanceReport(
            gross_sharpe=self.sharpe(gross_returns),
            net_sharpe=self.sharpe(net_returns),
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            sortino_ratio=self.sortino(net_returns),
            max_drawdown=mdd,
            calmar_ratio=calmar,
            win_loss_ratio=win_loss_ratio,
            profit_factor=profit_factor,
            turnover=turnover,
            total_costs=float(daily_cost_total.sum()),
            total_traded_notional=total_traded_notional,
            n_fills=len(trade_log),
        )

    def cost_stress_test(
        self,
        run_at_scalar: Callable[[float], PerformanceReport],
        cost_scalars: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0),
    ) -> tuple[pd.DataFrame, float | None]:
        rows = []
        reference = None
        for s in cost_scalars:
            report = run_at_scalar(s)
            total_costs = report.total_costs
            rows.append({
                "cost_scalar": s,
                "total_return": report.total_return,
                "net_sharpe": report.net_sharpe,
                "total_costs": total_costs,
            })
            if s == 1.0:
                reference = (total_costs, report.total_traded_notional)

        df = pd.DataFrame(rows).sort_values("cost_scalar").reset_index(drop=True)
        breakeven_scalar = self._interpolate_breakeven(df)

        breakeven_cost_bps = None
        if breakeven_scalar is not None and reference is not None:
            ref_cost, ref_notional = reference
            if ref_notional:
                breakeven_cost_bps = (ref_cost * breakeven_scalar) / ref_notional * 10_000

        return df, breakeven_cost_bps

    @staticmethod
    def _interpolate_breakeven(df: pd.DataFrame) -> float | None:
        for i in range(len(df) - 1):
            r0, r1 = df.iloc[i], df.iloc[i + 1]
            if (r0["total_return"] >= 0) != (r1["total_return"] >= 0):
                span = r1["total_return"] - r0["total_return"]
                if span == 0:
                    continue
                frac = -r0["total_return"] / span
                return r0["cost_scalar"] + frac * (r1["cost_scalar"] - r0["cost_scalar"])
        return None
