from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import factors, settings

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb

    lgb.LGBMRegressor(n_estimators=1, verbose=-1).fit([[0.0], [1.0]], [0.0, 1.0])
    GBM_LIBRARY = "lightgbm"
except Exception:
    GBM_LIBRARY = "sklearn_hgbr"
logger.info("ml.cross_sectional: GBM backend = %s", GBM_LIBRARY)

__all__ = [
    "GBM_LIBRARY",
    "WalkForwardFold",
    "build_training_frame",
    "feature_columns",
    "fit_predict_fold",
    "make_walk_forward_folds",
    "model_factor_weights",
    "month_end_rebalance_dates",
    "predict_expanding_monthly",
    "predict_walk_forward",
    "assert_no_lookahead_cross_sectional",
]


def get_gbm_regressor(params: dict, random_state: int = 42):
    if GBM_LIBRARY == "lightgbm":
        return lgb.LGBMRegressor(
            num_leaves=params.get("num_leaves", 15),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.05),
            n_estimators=params.get("n_estimators", 150),
            min_child_samples=params.get("min_samples_leaf", 20),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_alpha=params.get("reg_alpha", 0.01),
            reg_lambda=params.get("reg_lambda", 0.01),
            random_state=random_state,
            verbose=-1,
        )
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_depth=params.get("max_depth", 4),
        learning_rate=params.get("learning_rate", 0.05),
        max_iter=params.get("n_estimators", 150),
        min_samples_leaf=params.get("min_samples_leaf", 20),
        l2_regularization=params.get("reg_lambda", 0.01),
        random_state=random_state,
    )


def month_end_rebalance_dates(dates) -> pd.DatetimeIndex:
    s = pd.Series(pd.DatetimeIndex(pd.unique(pd.DatetimeIndex(dates))).sort_values())
    return pd.DatetimeIndex(s.groupby(s.dt.to_period("M")).max())


@dataclass
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_walk_forward_folds(
    rebal_dates: pd.DatetimeIndex,
    n_folds: int | None = None,
    min_train_months: int | None = None,
    embargo_months: int | None = None,
) -> list[WalkForwardFold]:
    n_folds = n_folds if n_folds is not None else settings.WALK_FORWARD_N_FOLDS
    min_train_months = min_train_months if min_train_months is not None else settings.WALK_FORWARD_MIN_TRAIN_MONTHS
    embargo_months = embargo_months if embargo_months is not None else settings.WALK_FORWARD_EMBARGO_MONTHS

    dates = pd.DatetimeIndex(sorted(pd.unique(rebal_dates)))
    n = len(dates)
    if n <= min_train_months + embargo_months + n_folds:
        raise ValueError(
            f"Not enough history ({n} periods) for {n_folds} folds with a {min_train_months}-month minimum train window"
        )

    fold_size = (n - min_train_months) // n_folds
    folds = []
    for k in range(n_folds):
        test_start_idx = min_train_months + k * fold_size
        test_end_idx = min_train_months + (k + 1) * fold_size - 1 if k < n_folds - 1 else n - 1
        # embargo: labels look forward and would overlap the test window
        train_end_idx = test_start_idx - embargo_months - 1
        if train_end_idx < 0:
            raise ValueError("embargo_months too large relative to min_train_months")
        folds.append(WalkForwardFold(
            fold_id=k,
            train_start=dates[0],
            train_end=dates[train_end_idx],
            test_start=dates[test_start_idx],
            test_end=dates[test_end_idx],
        ))
    return folds


def feature_columns(panel: pd.DataFrame) -> list[str]:
    return [f"pct_{f}" for f in factors.ALL_FACTORS if f"pct_{f}" in panel.columns] + ["sector_code"]


def build_training_frame(
    panel: pd.DataFrame,
    daily_returns: pd.DataFrame,
    horizon_days: int | None = None,
) -> pd.DataFrame:
    horizon_days = horizon_days if horizon_days is not None else settings.LABEL_HORIZON_DAYS

    rets = daily_returns[["permno", "date", "ret"]].copy()
    rets["date"] = pd.to_datetime(rets["date"])
    rets["ret"] = rets["ret"].astype("float64")
    rets = rets.sort_values(["permno", "date"])
    cum = rets.groupby("permno")["ret"].transform(lambda s: np.log1p(s.fillna(0.0)).cumsum())
    rets["fwd_ret"] = np.exp(cum.groupby(rets["permno"]).shift(-horizon_days) - cum) - 1

    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    month_ends = set(month_end_rebalance_dates(df["date"]))
    df = df[df["date"].isin(month_ends)]

    df = df.merge(rets[["permno", "date", "fwd_ret"]], on=["permno", "date"], how="left")
    df["sector_code"] = pd.to_numeric(df["gsector"], errors="coerce")
    # demeaned per date: learn relative rank, not market direction
    df["label"] = df["fwd_ret"] - df.groupby("date")["fwd_ret"].transform("mean")

    feats = feature_columns(df)
    keep = ["permno", "date", "label", "fwd_ret"] + feats
    df = df[keep].dropna(subset=["label"])
    return df[df[feats].notna().any(axis=1)].reset_index(drop=True)


def fit_predict_fold(
    frame: pd.DataFrame,
    fold: WalkForwardFold,
    gbm_params: dict | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    gbm_params = gbm_params if gbm_params is not None else settings.GBM_DEFAULT_PARAMS

    train = frame[(frame["date"] >= fold.train_start) & (frame["date"] <= fold.train_end)]
    test = frame[(frame["date"] >= fold.test_start) & (frame["date"] <= fold.test_end)]
    if train.empty or test.empty:
        return pd.DataFrame(columns=["permno", "date", "prediction"])

    feats = [c for c in feature_columns(frame) if train[c].nunique(dropna=True) >= 2]
    if not feats:
        return pd.DataFrame(columns=["permno", "date", "prediction"])

    model = get_gbm_regressor(gbm_params, random_state=random_state)
    model.fit(train[feats].to_numpy(dtype="float64"), train["label"].to_numpy(dtype="float64"))
    pred = model.predict(test[feats].to_numpy(dtype="float64"))

    return pd.DataFrame({"permno": test["permno"].to_numpy(), "date": test["date"].to_numpy(), "prediction": pred})


def predict_expanding_monthly(
    frame: pd.DataFrame,
    min_train_months: int | None = None,
    embargo_months: int | None = None,
    gbm_params: dict | None = None,
    random_state: int = 42,
    refit_every: int = 1,
    return_weights: bool = False,
):
    min_train_months = min_train_months if min_train_months is not None else settings.WALK_FORWARD_MIN_TRAIN_MONTHS
    embargo_months = embargo_months if embargo_months is not None else settings.WALK_FORWARD_EMBARGO_MONTHS
    gbm_params = gbm_params if gbm_params is not None else settings.GBM_DEFAULT_PARAMS

    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    all_feats = feature_columns(frame)
    out, weights, model, feats = [], [], None, None

    for i in range(min_train_months + embargo_months, len(dates)):
        test_date = dates[i]
        train_end = dates[i - embargo_months - 1]
        refitted = False

        if model is None or (i - (min_train_months + embargo_months)) % refit_every == 0:
            train = frame[frame["date"] <= train_end]
            feats = [c for c in all_feats if train[c].nunique(dropna=True) >= 2]
            if not feats:
                continue
            model = get_gbm_regressor(gbm_params, random_state=random_state)
            model.fit(train[feats].to_numpy(dtype="float64"), train["label"].to_numpy(dtype="float64"))
            refitted = True

        test = frame[frame["date"] == test_date]
        if test.empty:
            continue
        X_test = test[feats].to_numpy(dtype="float64")
        out.append(pd.DataFrame({
            "permno": test["permno"].to_numpy(),
            "date": test["date"].to_numpy(),
            "prediction": model.predict(X_test),
        }))

        if return_weights and refitted:
            w = model_factor_weights(model, X_test, feats, random_state=random_state)
            weights.append(pd.DataFrame({
                "date": test_date,
                "factor": [f.removeprefix("pct_") for f in w],
                "weight": list(w.values()),
            }))

    preds = (pd.concat(out, ignore_index=True).sort_values(["date", "permno"]).reset_index(drop=True)
             if out else pd.DataFrame(columns=["permno", "date", "prediction"]))
    if not return_weights:
        return preds
    wdf = (pd.concat(weights, ignore_index=True) if weights
           else pd.DataFrame(columns=["date", "factor", "weight"]))
    return preds, wdf


def model_factor_weights(model, X: np.ndarray, feats: list[str], n_repeats: int = 3,
                         random_state: int = 42) -> dict[str, float]:
    if X.size == 0 or not feats:
        return {}
    rng = np.random.default_rng(random_state)
    base = model.predict(X)
    scores = {}
    for j, name in enumerate(feats):
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            deltas.append(np.abs(model.predict(Xp) - base).mean())
        scores[name] = float(np.mean(deltas))
    total = sum(scores.values())
    return {k: v / total for k, v in scores.items()} if total > 0 else scores


def predict_walk_forward(
    frame: pd.DataFrame,
    folds: list[WalkForwardFold],
    gbm_params: dict | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    parts = [fit_predict_fold(frame, f, gbm_params, random_state) for f in folds]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["permno", "date", "prediction"])
    return pd.concat(parts, ignore_index=True).sort_values(["date", "permno"]).reset_index(drop=True)


def predictions_to_daily_score(
    predictions: pd.DataFrame, panel: pd.DataFrame, score_col: str = "combined_score"
) -> pd.DataFrame:
    preds = predictions.sort_values("date")[["permno", "date", "prediction"]].copy()
    preds["permno"] = preds["permno"].astype("int64")
    out = panel.copy()
    out["permno"] = out["permno"].astype("int64")
    out["date"] = pd.to_datetime(out["date"])

    merged = pd.merge_asof(
        out.sort_values("date"),
        preds.rename(columns={"date": "pred_date"}).sort_values("pred_date"),
        by="permno",
        left_on="date",
        right_on="pred_date",
        direction="backward",
    )
    ranked = merged.groupby("date")["prediction"].rank(pct=True)
    merged[score_col] = ranked.where(ranked.notna(), merged[score_col])
    return merged.drop(columns=["pred_date", "prediction"])


def assert_no_lookahead_cross_sectional() -> None:
    rng = np.random.default_rng(11)
    permnos = [10001, 10002, 10003, 10004, 10005]
    dates = pd.bdate_range("2010-01-01", periods=260)

    panel = pd.DataFrame(
        [{"permno": p, "date": d} for p in permnos for d in dates]
    )
    panel["gsector"] = "40"
    for f in factors.ALL_FACTORS:
        panel[f"pct_{f}"] = rng.random(len(panel))
    panel["combined_score"] = rng.random(len(panel))

    rets = panel[["permno", "date"]].copy()
    rets["ret"] = rng.normal(0, 0.015, len(rets))

    cut = dates[len(dates) // 2]
    base = build_training_frame(panel, rets)

    perturbed = rets.copy()
    mask = perturbed["date"] > cut
    perturbed.loc[mask, "ret"] = (perturbed.loc[mask, "ret"] * 40 + 0.5).clip(lower=-0.9)
    after = build_training_frame(panel, perturbed)

    feats = feature_columns(base)
    past = base["date"] <= cut - pd.Timedelta(f"{settings.LABEL_HORIZON_DAYS * 2}D")
    pd.testing.assert_frame_equal(
        base.loc[past, ["permno", "date"] + feats].reset_index(drop=True),
        after.loc[past.values, ["permno", "date"] + feats].reset_index(drop=True),
    )

    touched = base["date"] > cut - pd.Timedelta(f"{settings.LABEL_HORIZON_DAYS}D")
    assert not np.allclose(
        base.loc[touched, "label"].to_numpy(), after.loc[touched.values, "label"].to_numpy()
    ), "labels did not react to future data - the test is not exercising anything"

    print("PASS: features at date d are unaffected by data after d; labels do react.")


if __name__ == "__main__":
    print(f"GBM backend: {GBM_LIBRARY}")
    assert_no_lookahead_cross_sectional()

    panel = factors.compute_combined_score()
    crsp = pd.read_parquet(factors.CACHE_DIR / "crsp_panel.parquet", columns=["permno", "date", "ret"])
    frame = build_training_frame(panel, crsp)
    feats = feature_columns(frame)
    print(f"\ntraining frame: {len(frame):,} rows x {len(feats)} features, "
          f"{frame['date'].nunique()} month-ends "
          f"({frame['date'].min().date()} -> {frame['date'].max().date()})")

    folds = make_walk_forward_folds(pd.DatetimeIndex(sorted(frame["date"].unique())))
    for f in folds:
        print(f"  fold {f.fold_id}: train -> {f.train_end.date()}   test {f.test_start.date()} -> {f.test_end.date()}")

    preds = predict_walk_forward(frame, folds[:-1])
    merged = preds.merge(frame[["permno", "date", "label"]], on=["permno", "date"])
    ic = merged.groupby("date").apply(
        lambda d: d["prediction"].corr(d["label"], method="spearman"), include_groups=False
    ).dropna()
    print(f"\nOOS monthly IC over search folds: mean {ic.mean():+.4f}  "
          f"t {ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic)):+.2f}  ({len(ic)} months)")
