DATA_START = "2000-01-01"
DATA_END = None

DISABLED_FACTORS: set[str] = {
    "max_dd",
    "capital_adequacy",
    "efficiency_ratio",
    "reserve_adequacy",
    "premium_growth_yoy",
}

# Warm-up fallback only; live weights come from trailing Rank IC.
FACTOR_WEIGHTS: dict[str, float] = {
    "value":       0.20,
    "quality":     0.20,
    "growth":      0.20,
    "momentum":    0.20,
    "volatility":  0.20,
}


# value
book_to_market = 0.125
earnings_yield = 0.125
sales_yield = 0.125
ebitda_to_ev = 0.125
ocf_yield = 0.125
fcf_yield = 0.125
shareholder_yield = 0.125
tangible_book_to_market = 0.125   # financials only

# quality
gross_profitability = 0.0714
roe = 0.0714
roa = 0.0714
operating_profitability = 0.0714
accruals = 0.0714
leverage = 0.0714
interest_coverage = 0.0714
asset_turnover = 0.0714
net_debt_issuance = 0.0714
days_sales_outstanding = 0.0714
days_inventory = 0.0714
days_payable = 0.0714
cash_conversion_cycle = 0.0714
goodwill_intangibles_pct = 0.0714

# growth
sales_growth_yoy = 0.0833
sales_growth_3y_cagr = 0.0833
ni_growth_yoy = 0.0833
ebitda_growth_yoy = 0.0833
gp_growth_yoy = 0.0833
oancf_growth_yoy = 0.0833
fcf_growth_yoy = 0.0833
book_equity_growth_yoy = 0.0833
capx_growth_yoy = 0.0833
niint_growth_yoy = 0.0833         # banks
dptc_growth_yoy = 0.0833          # banks
initb_growth_yoy = 0.0833         # banks

# momentum: skip-month J-T at 70%, raw short-horizon at 30%
mom_12_1 = 0.40
mom_8_1 = 0.20
mom_7_1 = 0.0333
mom_9_1 = 0.0333
mom_10_1 = 0.0333
mom_24d = 0.06
mom_36d = 0.06
mom_50d = 0.06
mom_62d = 0.06
mom_126d = 0.06

# volatility
realised_vol_short = 0.25
realised_vol_long = 0.25
low_vol_annual = 0.25
beta = 0.25


WITHIN_FACTOR_WEIGHT_OVERRIDES: dict[str, dict[str, float]] = {
    "value": {
        "book_to_market": book_to_market,
        "earnings_yield": earnings_yield,
        "sales_yield": sales_yield,
        "ebitda_to_ev": ebitda_to_ev,
        "ocf_yield": ocf_yield,
        "fcf_yield": fcf_yield,
        "shareholder_yield": shareholder_yield,
        "tangible_book_to_market": tangible_book_to_market,
    },
    "quality": {
        "gross_profitability": gross_profitability,
        "roe": roe,
        "roa": roa,
        "operating_profitability": operating_profitability,
        "accruals": accruals,
        "leverage": leverage,
        "interest_coverage": interest_coverage,
        "asset_turnover": asset_turnover,
        "net_debt_issuance": net_debt_issuance,
        "days_sales_outstanding": days_sales_outstanding,
        "days_inventory": days_inventory,
        "days_payable": days_payable,
        "cash_conversion_cycle": cash_conversion_cycle,
        "goodwill_intangibles_pct": goodwill_intangibles_pct,
    },
    "growth": {
        "sales_growth_yoy": sales_growth_yoy,
        "sales_growth_3y_cagr": sales_growth_3y_cagr,
        "ni_growth_yoy": ni_growth_yoy,
        "ebitda_growth_yoy": ebitda_growth_yoy,
        "gp_growth_yoy": gp_growth_yoy,
        "oancf_growth_yoy": oancf_growth_yoy,
        "fcf_growth_yoy": fcf_growth_yoy,
        "book_equity_growth_yoy": book_equity_growth_yoy,
        "capx_growth_yoy": capx_growth_yoy,
        "niint_growth_yoy": niint_growth_yoy,
        "dptc_growth_yoy": dptc_growth_yoy,
        "initb_growth_yoy": initb_growth_yoy,
    },
    "momentum": {
        "mom_12_1": mom_12_1,
        "mom_8_1": mom_8_1,
        "mom_7_1": mom_7_1,
        "mom_9_1": mom_9_1,
        "mom_10_1": mom_10_1,
        "mom_24d": mom_24d,
        "mom_36d": mom_36d,
        "mom_50d": mom_50d,
        "mom_62d": mom_62d,
        "mom_126d": mom_126d,
    },
    "volatility": {
        "realised_vol_short": realised_vol_short,
        "realised_vol_long": realised_vol_long,
        "low_vol_annual": low_vol_annual,
        "beta": beta,
    },
}


MAX_FUNDAMENTALS_STALENESS_DAYS: int = 450


LOOKBACKS = {
    "mom_skip_months": (12, 10, 9, 8, 7),
    "mom_month_days": 21,
    "mom_skip_days": 21,
    "mom_short_days": (24, 36, 50, 62, 126),

    "realised_vol_short": 20,
    "realised_vol_long": 60,
    "low_vol_annual": 252,
    "beta": 60,
    "max_dd": 60,
}

SELECTION_N: int = 25
SELECTION_BUFFER_RANK: int = 75
SELECTION_BUFFER_DAYS: int = 3

# Fixed clock; breaches below interrupt it.
REBALANCE_INTERVAL_DAYS: int = 25
ENTRY_TRIGGER_RANK: int = 10

# 1.25x equal weight = 5% cap at N=25.
MAX_WEIGHT_MULTIPLE_OF_EQUAL: float = 1.25

POSITION_NO_TRADE_BAND: float = 0.005

ROLLING_VOL_WINDOW: int = 20

DRAWDOWN_TIERS: list[tuple[float, float]] = [
    (-0.10, 0.75),
    (-0.15, 0.50),
    (-0.20, 0.00),
]
# 1 = next-day exit; 0 would fire every tier daily.
DRAWDOWN_CONFIRMATION_DAYS: int = 1
DRAWDOWN_REENTRY_VOL_RATIO: float = 0.85

# De-risk only; capped at 1.0 (unlevered).
TARGET_VOL: float = 0.15
VOL_TARGET_WINDOW_DAYS: int = 60
VOL_TARGET_BOUNDS: tuple[float, float] = (0.3, 1.0)

FLAT_COST_BPS: float = 7.5


GBM_DEFAULT_PARAMS: dict = {
    "num_leaves": 15, "max_depth": 4, "learning_rate": 0.05, "n_estimators": 150,
    "min_samples_leaf": 20, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.01, "reg_lambda": 0.01,
}

LABEL_HORIZON_DAYS: int = 21

WALK_FORWARD_N_FOLDS: int = 5
WALK_FORWARD_MIN_TRAIN_MONTHS: int = 36
WALK_FORWARD_EMBARGO_MONTHS: int = 1

OPTUNA_N_TRIALS_STAGE1: int = 150
OPTUNA_N_TRIALS_STAGE2: int = 150

# objective = -mean(fold Sharpe) + LAMBDA * mean(fold |maxDD|)
DRAWDOWN_PENALTY_LAMBDA: float = 1.0

# 12 = best cell of a noisy sweep; carries selection bias.
RANK_IC_WINDOW_MONTHS: int = 12

FAMILY_WEIGHT_BOUNDS: tuple[float, float] = (0.05, 0.60)
