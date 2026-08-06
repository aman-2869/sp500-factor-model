from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from src import factors, settings
from src.backtest.backtester import (
    ExecutionEngine, FlatCostExecutionSimulator, TCAReporter, iter_market_events,
    load_market_frame, load_trading_calendar,
)
from src.backtest.portfolio import PortfolioTracker
from src.cross_sectional import (
    build_training_frame, fit_predict_fold, make_walk_forward_folds,
    month_end_rebalance_dates, predictions_to_daily_score,
)
from src.sizing import (
    DrawdownOverlay, RollingVolatilityTracker, SizedLongOnlyStrategy, StockPicker,
    VolatilityTargetOverlay, warm_up_vol_tracker,
)

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
STUDY_DIR = Path(__file__).resolve().parent.parent / "results" / "optimization"


# -mean(Sharpe) + lambda*mean(|maxDD|); Sharpe alone ignores drawdown.
def _objective_value(sharpes: list[float], max_drawdowns: list[float]) -> float:
    lam = settings.DRAWDOWN_PENALTY_LAMBDA
    return -float(np.mean(sharpes)) + lam * float(np.mean(np.abs(max_drawdowns)))


class Optimizer:

    def __init__(
        self,
        start: str | None = None,
        end: str | None = None,
        holdout_last_fold: bool = True,
        starting_cash: float = 10_000_000.0,
    ):
        self.start = pd.Timestamp(start or "2001-01-01")
        self.end = pd.Timestamp(end or "2025-12-31")
        self.starting_cash = starting_cash
        self.optimization_results: dict = {}

        logger.info("Optimizer: loading cached panels and computing factor percentiles...")
        self._funda = pd.read_parquet(CACHE_DIR / "fundamentals_panel.parquet")
        self._mom = pd.read_parquet(CACHE_DIR / "momentum_panel.parquet")
        self._vol = pd.read_parquet(CACHE_DIR / "volatility_panel.parquet")
        self._crsp = pd.read_parquet(CACHE_DIR / "crsp_panel.parquet")
        panel = factors.compute_combined_score(self._funda, self._mom, self._vol, self._crsp)
        self.combined_panel = panel[(panel["date"] >= self.start) & (panel["date"] <= self.end)].reset_index(drop=True)

        self.universe_permnos = sorted(self.combined_panel["permno"].unique())
        self.rebal_dates = month_end_rebalance_dates(self.combined_panel["date"])

        trading_days = pd.DatetimeIndex(sorted(self.combined_panel["date"].unique()))
        self.weight_grid = trading_days[:: settings.REBALANCE_INTERVAL_DAYS]

        logger.info("Computing rolling Rank IC on the rebalance grid (once)...")
        self.factor_ic = factors.rolling_rank_ic(self.combined_panel, self._crsp, self.weight_grid)
        self.ic_schedule = factors.ic_weight_schedule(self.factor_ic)

        logger.info("Building the cross-sectional training frame (once)...")
        self.training_frame = build_training_frame(self.combined_panel, self._crsp)

        self.folds = make_walk_forward_folds(pd.DatetimeIndex(sorted(self.training_frame["date"].unique())))
        self.holdout_last_fold = holdout_last_fold
        # the holdout fold is never seen by any objective
        self.search_folds = self.folds[:-1] if holdout_last_fold else self.folds
        self.holdout_fold = self.folds[-1] if holdout_last_fold else None

        logger.info("Loading market frame/calendar (once)...")
        permnos = [int(p) for p in self.universe_permnos]
        self.market_frame = self._repair(load_market_frame(permnos=permnos, start=self.start, end=self.end))
        self.calendar = load_trading_calendar(permnos=permnos, start=self.start, end=self.end)
        self.raw_daily = pd.read_parquet(RAW_DIR / "crsp_dsf_v2.parquet", columns=["permno", "date", "ret"])

        logger.info(
            "Optimizer ready: %d month-ends, %d folds (%d search + %d holdout), %d training rows",
            len(self.rebal_dates), len(self.folds), len(self.search_folds),
            1 if holdout_last_fold else 0, len(self.training_frame),
        )

    @staticmethod
    def _repair(market_frame: pd.DataFrame) -> pd.DataFrame:
        mf = market_frame.sort_values(["permno", "date"])
        ffilled = mf.groupby("permno")["close"].ffill()
        still_nan = ffilled.isna()
        fix = mf["close"].isna() & ~still_nan
        for col in ["open", "high", "low", "close"]:
            mf.loc[fix, col] = ffilled[fix]
        mf.loc[fix, "volume"] = 0.0
        return mf[~still_nan].sort_values(["date", "permno"]).reset_index(drop=True)


    def rescore(self) -> pd.DataFrame:
        panel = self.combined_panel
        fam_sched, within_sched = self.ic_schedule
        out = panel[["permno", "date", "combined_score"]].copy()

        for family, (flist, _) in factors.FAMILIES.items():
            cols = [f for f in flist if f in within_sched[family].columns]
            wide = factors._expand_schedule(within_sched[family][cols], panel["date"])
            wcols = [f"_wf_{f}" for f in cols]
            merged = panel[["date"] + [f"pct_{f}" for f in cols]].merge(
                wide.rename(columns={f: f"_wf_{f}" for f in cols}),
                left_on="date", right_index=True, how="left",
            )
            fallback = settings.WITHIN_FACTOR_WEIGHT_OVERRIDES.get(family, {})
            for f, wcol in zip(cols, wcols):
                merged[wcol] = merged[wcol].fillna(fallback.get(f, 1.0))
            score = factors.weighted_combine_dynamic(merged, [f"pct_{f}" for f in cols], wcols)
            out[f"{family}_score"] = score.to_numpy()
            out[f"{family}_score"] = out[f"{family}_score"].groupby(panel["date"]).rank(pct=True)

        fams = list(factors.FAMILIES)
        fam_wide = factors._expand_schedule(fam_sched, panel["date"])
        fam_cols = [f"_wfam_{fam}" for fam in fams]
        merged = out[["date"] + [f"{fam}_score" for fam in fams]].merge(
            fam_wide.rename(columns={fam: f"_wfam_{fam}" for fam in fams}),
            left_on="date", right_index=True, how="left",
        )
        for fam, c in zip(fams, fam_cols):
            merged[c] = merged[c].fillna(settings.FACTOR_WEIGHTS[fam])
        out["combined_score"] = factors.weighted_combine_dynamic(
            merged, [f"{fam}_score" for fam in fams], fam_cols
        ).to_numpy()
        return out


    def run_backtest(self, scored_panel: pd.DataFrame, start, end, sizing_params: dict) -> PortfolioTracker:
        vol_tracker = RollingVolatilityTracker(window=sizing_params.get("rolling_vol_window"))
        warm_up_vol_tracker(vol_tracker, self.raw_daily, self.universe_permnos, start)

        portfolio = PortfolioTracker(starting_cash=self.starting_cash)
        strategy = SizedLongOnlyStrategy(
            scored_panel,
            enabled=True,
            vol_tracker=vol_tracker,
            max_weight_multiple=sizing_params.get("max_weight_multiple"),
            picker=StockPicker(
                n=sizing_params.get("selection_n"),
                buffer_rank=sizing_params.get("selection_buffer_rank"),
                buffer_days=sizing_params.get("selection_buffer_days"),
                rebalance_interval=sizing_params.get("rebalance_interval"),
                entry_trigger_rank=sizing_params.get("entry_trigger_rank"),
            ),
            drawdown_overlay=DrawdownOverlay(
                tiers=sizing_params.get("drawdown_tiers"),
                confirmation_days=sizing_params.get("drawdown_confirmation_days"),
                reentry_vol_ratio=sizing_params.get("drawdown_reentry_vol_ratio"),
            ),
            vol_target_overlay=VolatilityTargetOverlay(
                target_vol=sizing_params.get("target_vol"),
                bounds=sizing_params.get("vol_target_bounds"),
                window=sizing_params.get("vol_target_window"),
            ),
            portfolio_vol=None,
            position_no_trade_band=sizing_params.get("position_no_trade_band"),
        )
        engine = ExecutionEngine(
            FlatCostExecutionSimulator(cost_bps=sizing_params.get("cost_bps")),
            portfolio, strategy, latency_bars=1, twap_slices=1,
        )

        mask = (self.market_frame["date"] >= pd.Timestamp(start)) & (self.market_frame["date"] <= pd.Timestamp(end))
        calendar = [d for d in self.calendar if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
        engine.run(iter_market_events(self.market_frame.loc[mask].reset_index(drop=True)), all_timestamps=calendar)
        return portfolio

    @staticmethod
    def metrics(portfolio: PortfolioTracker, risk_free_rate: float = 0.03) -> dict:
        report = TCAReporter(periods_per_year=252, annual_risk_free_rate=risk_free_rate).generate(portfolio)
        return {f: getattr(report, f) for f in report.__dataclass_fields__}

    def _fold_scores(
        self, scored_panel: pd.DataFrame, folds, sizing_params: dict, trial=None
    ) -> tuple[list[float], list[float]]:
        sharpes, drawdowns, last = [], [], None
        for i, fold in enumerate(folds):
            last = self.metrics(self.run_backtest(scored_panel, fold.test_start, fold.test_end, sizing_params))
            s, dd = last["net_sharpe"], last["max_drawdown"]
            sharpes.append(s if s == s else -10.0)
            drawdowns.append(dd if dd == dd else -1.0)
            if trial is not None:
                trial.report(_objective_value(sharpes, drawdowns), step=i)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        if trial is not None and last is not None:
            trial.set_user_attr("last_fold_cagr", float(last["annualized_return"]))
            trial.set_user_attr("last_fold_turnover", float(last["turnover"]))
            trial.set_user_attr("last_fold_vol", float(last["annualized_volatility"]))
            trial.set_user_attr("mean_fold_sharpe", float(np.mean(sharpes)))
            trial.set_user_attr("mean_fold_max_drawdown", float(np.mean(drawdowns)))
        return sharpes, drawdowns

    def _ml_scored_panel(self, folds, gbm_params: dict) -> pd.DataFrame:
        preds = [fit_predict_fold(self.training_frame, f, gbm_params) for f in folds]
        preds = [p for p in preds if not p.empty]
        if not preds:
            return self.combined_panel[["permno", "date", "combined_score"]].copy()
        merged = pd.concat(preds, ignore_index=True)
        return predictions_to_daily_score(merged, self.combined_panel[["permno", "date", "combined_score"]])


    def _run_study(self, key: str, objective, seed_params: dict, n_trials: int) -> optuna.Study:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        )
        if seed_params:
            study.enqueue_trial(seed_params)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        self.optimization_results[key] = {
            "study": study,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_user_attrs": study.best_trial.user_attrs,
        }
        logger.info("%s: best value %.4f", key, study.best_value)
        return study

    def optimize_cross_sectional(self, n_trials: int | None = None) -> optuna.Study:
        n_trials = n_trials if n_trials is not None else settings.OPTUNA_N_TRIALS_STAGE1

        def objective(trial: optuna.Trial) -> float:
            gbm_params = {
                "num_leaves": trial.suggest_int("num_leaves", 7, 31),
                "max_depth": trial.suggest_int("max_depth", 2, 5),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 50, 250),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 50, 500, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            }
            scored = self._ml_scored_panel(self.search_folds, gbm_params)
            sharpes, drawdowns = self._fold_scores(scored, self.search_folds, {}, trial)
            return _objective_value(sharpes, drawdowns)

        return self._run_study("cross_sectional", objective, dict(settings.GBM_DEFAULT_PARAMS), n_trials)

    def optimize_sizing_overlay(
        self, scored_panel: pd.DataFrame, n_trials: int | None = None, key: str = "sizing_overlay"
    ) -> optuna.Study:
        n_trials = n_trials if n_trials is not None else settings.OPTUNA_N_TRIALS_STAGE2

        n = settings.SELECTION_N

        def objective(trial: optuna.Trial) -> float:
            sizing_params = {
                "selection_n": n,
                "max_weight_multiple": settings.MAX_WEIGHT_MULTIPLE_OF_EQUAL,
                "selection_buffer_days": settings.SELECTION_BUFFER_DAYS,
                "selection_buffer_rank": settings.SELECTION_BUFFER_RANK,
                "rebalance_interval": settings.REBALANCE_INTERVAL_DAYS,
                "entry_trigger_rank": settings.ENTRY_TRIGGER_RANK,
                "drawdown_confirmation_days": settings.DRAWDOWN_CONFIRMATION_DAYS,
                "position_no_trade_band": trial.suggest_float("position_no_trade_band", 0.0, 0.02),
                "rolling_vol_window": trial.suggest_int("rolling_vol_window", 10, 60),
                "drawdown_reentry_vol_ratio": trial.suggest_float("drawdown_reentry_vol_ratio", 0.6, 0.95),
            }
            t1 = trial.suggest_float("drawdown_tier1_threshold", -0.15, -0.03)
            t2 = trial.suggest_float("drawdown_tier2_threshold", -0.25, t1 - 0.02)
            t3 = trial.suggest_float("drawdown_tier3_threshold", -0.35, t2 - 0.02)
            sizing_params["drawdown_tiers"] = [
                (t1, trial.suggest_float("drawdown_tier1_mult", 0.5, 0.9)),
                (t2, trial.suggest_float("drawdown_tier2_mult", 0.2, 0.6)),
                (t3, trial.suggest_float("drawdown_tier3_mult", 0.0, 0.3)),
            ]
            trial.set_user_attr("sizing_params", {k: v for k, v in sizing_params.items() if k != "drawdown_tiers"})

            sharpes, drawdowns = self._fold_scores(scored_panel, self.search_folds, sizing_params, trial)
            return _objective_value(sharpes, drawdowns)

        seed = {
            "position_no_trade_band": settings.POSITION_NO_TRADE_BAND,
            "rolling_vol_window": settings.ROLLING_VOL_WINDOW,
            "drawdown_reentry_vol_ratio": settings.DRAWDOWN_REENTRY_VOL_RATIO,
            "drawdown_tier1_threshold": settings.DRAWDOWN_TIERS[0][0],
            "drawdown_tier2_threshold": settings.DRAWDOWN_TIERS[1][0],
            "drawdown_tier3_threshold": settings.DRAWDOWN_TIERS[2][0],
            "drawdown_tier1_mult": settings.DRAWDOWN_TIERS[0][1],
            "drawdown_tier2_mult": settings.DRAWDOWN_TIERS[1][1],
            "drawdown_tier3_mult": settings.DRAWDOWN_TIERS[2][1],
        }
        return self._run_study(key, objective, seed, n_trials)


    def get_optimization_summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {k: {"best_value": r["best_value"], "n_trials": len(r["study"].trials)}
             for k, r in self.optimization_results.items()}
        ).T

    def plot_optimization_history(self, key: str, save_path: str | None = None):
        import matplotlib.pyplot as plt

        study = self.optimization_results[key]["study"]
        values = [t.value for t in study.trials if t.value is not None]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(values, linewidth=1.3)
        ax.plot(np.minimum.accumulate(values), linewidth=1.6, linestyle="--", label="best so far")
        ax.set_title(f"Optuna optimization history: {key}")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Objective (-mean fold Sharpe)")
        ax.legend(frameon=False)
        ax.grid(True, linewidth=0.5, alpha=0.4)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.show()

    def export_optimization_results(self, key: str, out_dir: str | None = None) -> None:
        out = Path(out_dir) if out_dir is not None else STUDY_DIR
        out.mkdir(parents=True, exist_ok=True)
        results = self.optimization_results[key]

        with open(out / f"{key}_params.txt", "w") as f:
            f.write(f"{key.replace('_', ' ').title()}\n" + "=" * 50 + "\n")
            f.write(f"Best value: {results['best_value']:.6f}\n")
            f.write(f"n_trials: {len(results['study'].trials)}\n\nBest parameters (raw search space):\n")
            for p, v in results["best_params"].items():
                f.write(f"  {p}: {v}\n")
            for name in ("family_weights", "within_weights", "sizing_params"):
                block = results["best_user_attrs"].get(name)
                if not block:
                    continue
                f.write(f"\n{name} (normalized, sums to 1.0 where applicable):\n")
                if isinstance(next(iter(block.values()), None), dict):
                    for fam, d in block.items():
                        f.write(f"  {fam}: (sum {sum(d.values()):.4f})\n")
                        for k, v in sorted(d.items(), key=lambda kv: -kv[1]):
                            f.write(f"    {k}: {v:.4f}\n")
                else:
                    total = sum(v for v in block.values() if isinstance(v, (int, float)))
                    f.write(f"  (sum {total:.4f})\n")
                    for k, v in block.items():
                        f.write(f"    {k}: {v}\n")

        with open(out / f"{key}_study.pkl", "wb") as f:
            pickle.dump(results["study"], f)
        logger.info("Exported %s to %s", key, out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("optimize.py smoke test: 2 trials each, truncated 2001-2008 window")
    opt = Optimizer(start="2001-01-01", end="2008-01-01", holdout_last_fold=True)

    fam_sched, _ = opt.ic_schedule
    sums = fam_sched.sum(axis=1)
    lo, hi = settings.FAMILY_WEIGHT_BOUNDS
    assert np.allclose(sums, 1.0), f"IC family weights must sum to 1.0, got {sums.min()}..{sums.max()}"
    assert fam_sched.min().min() >= lo - 1e-9 and fam_sched.max().max() <= hi + 1e-9, "family bounds breached"
    print(f"IC schedule: {len(fam_sched)} rebalances, family weights in "
          f"[{fam_sched.min().min():.3f}, {fam_sched.max().max():.3f}]")

    opt.optimize_cross_sectional(n_trials=2)
    print(opt.get_optimization_summary())
    print("PASS: Optimizer smoke test completed; IC weights sum to 1.0 and respect bounds.")
