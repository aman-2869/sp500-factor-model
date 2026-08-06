from __future__ import annotations

from dataclasses import dataclass

from src.backtest.backtester import FillEvent, OrderSide


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def unrealized_pnl(self, price: float) -> float:
        return self.quantity * (price - self.avg_cost)


class PortfolioTracker:
    def __init__(self, starting_cash: float = 1_000_000.0):
        self.cash = starting_cash
        self.starting_cash = starting_cash
        self.positions: dict[str, Position] = {}
        self.fills: list[FillEvent] = []
        self.trade_log: list[dict] = []
        self.equity_curve: list[tuple] = []

    def position_quantity(self, symbol: str) -> float:
        pos = self.positions.get(symbol)
        return pos.quantity if pos is not None else 0.0

    def _position(self, symbol: str) -> Position:
        return self.positions.setdefault(symbol, Position(symbol=symbol))

    def on_fill(self, fill: FillEvent) -> None:
        pos = self._position(fill.symbol)
        signed_qty = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
        cost = fill.cost

        realized_delta = 0.0
        if pos.quantity == 0 or (signed_qty > 0) == (pos.quantity > 0):
            new_qty = pos.quantity + signed_qty
            if new_qty != 0:
                pos.avg_cost = (pos.avg_cost * pos.quantity + fill.fill_price * signed_qty) / new_qty
            pos.quantity = new_qty
        else:
            closing_qty = min(abs(signed_qty), abs(pos.quantity))
            direction = 1 if pos.quantity > 0 else -1
            realized_delta = closing_qty * direction * (fill.fill_price - pos.avg_cost)
            pos.realized_pnl += realized_delta
            pos.quantity += signed_qty
            if abs(signed_qty) > closing_qty:
                pos.avg_cost = fill.fill_price
            elif pos.quantity == 0:
                pos.avg_cost = 0.0

        if fill.side == OrderSide.BUY:
            self.cash -= fill.notional + cost
        else:
            self.cash += fill.notional - cost

        self.fills.append(fill)
        self.trade_log.append({
            "timestamp": fill.timestamp,
            "symbol": fill.symbol,
            "side": fill.side.value,
            "quantity": fill.quantity,
            "fill_price": fill.fill_price,
            "realized_pnl": realized_delta,
            "cost": cost,
            "implementation_shortfall_bps": fill.implementation_shortfall_bps,
        })

    def mark_to_market(self, timestamp, prices: dict[str, float]) -> float:
        equity = self.cash
        for symbol, pos in self.positions.items():
            price = prices.get(symbol)
            if price is not None and pos.quantity != 0:
                equity += pos.market_value(price)
        self.equity_curve.append((timestamp, equity))
        return equity

    @property
    def realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.positions.values())

    def unrealized_pnl(self, prices: dict[str, float]) -> float:
        return sum(
            p.unrealized_pnl(prices[s]) for s, p in self.positions.items() if s in prices and p.quantity != 0
        )

    def gross_exposure(self, prices: dict[str, float]) -> float:
        return sum(abs(p.market_value(prices[s])) for s, p in self.positions.items() if s in prices)

    def net_exposure(self, prices: dict[str, float]) -> float:
        return sum(p.market_value(prices[s]) for s, p in self.positions.items() if s in prices)

    def leverage(self, prices: dict[str, float]) -> float:
        equity = self.cash + self.net_exposure(prices)
        if equity == 0:
            return float("nan")
        return self.gross_exposure(prices) / equity
