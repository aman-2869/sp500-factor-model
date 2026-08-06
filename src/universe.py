import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
_MEMBERSHIP_PATH = RAW_DIR / "sp500_membership.parquet"
_NAMES_PATH = RAW_DIR / "crsp_names.parquet"

_membership_history_cache: pd.DataFrame | None = None


def _load_membership() -> pd.DataFrame:
    return pd.read_parquet(_MEMBERSHIP_PATH)


def _load_names() -> pd.DataFrame:
    df = pd.read_parquet(_NAMES_PATH)
    df = df.dropna(subset=["ticker"]).copy()
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    return df


def get_membership_history(force_refresh: bool = False) -> pd.DataFrame:
    global _membership_history_cache
    if _membership_history_cache is not None and not force_refresh:
        return _membership_history_cache

    membership = _load_membership()
    names = _load_names()

    merged = membership.merge(names, on="permno", how="inner")
    overlap = (merged["start"] <= merged["nameendt"]) & (merged["ending"] >= merged["namedt"])
    merged = merged.loc[overlap].copy()

    merged["spell_start"] = merged[["start", "namedt"]].max(axis=1)
    merged["spell_end"] = merged[["ending", "nameendt"]].min(axis=1)

    history = (
        merged[["permno", "ticker", "comnam", "spell_start", "spell_end"]]
        .sort_values(["permno", "spell_start"])
        .reset_index(drop=True)
    )
    _membership_history_cache = history
    return history


def get_point_in_time_universe(as_of_date: str | pd.Timestamp | None = None) -> list[str]:
    history = get_membership_history()
    max_known = _load_membership()["ending"].max()

    as_of_date = pd.Timestamp.today().normalize() if as_of_date is None else pd.Timestamp(as_of_date)
    if as_of_date > max_known:
        logger.warning(
            "as_of_date %s is beyond the membership data's vintage (%s) — clamping. "
            "Re-run the WRDS pull in datasaver.ipynb to refresh sp500_membership.parquet "
            "/ crsp_names.parquet for a more current snapshot.",
            as_of_date.date(), max_known.date(),
        )
        as_of_date = max_known

    active = history[(history["spell_start"] <= as_of_date) & (history["spell_end"] >= as_of_date)]
    tickers = sorted(active["ticker"].unique())
    logger.info("Point-in-time universe on %s: %d tickers", as_of_date.date(), len(tickers))
    return tickers


def get_historical_tickers(start: str | pd.Timestamp, end: str | pd.Timestamp | None = None) -> list[str]:
    history = get_membership_history()
    start = pd.Timestamp(start)
    end = pd.Timestamp.today().normalize() if end is None else pd.Timestamp(end)

    overlapping = history[(history["spell_start"] <= end) & (history["spell_end"] >= start)]
    tickers = sorted(overlapping["ticker"].unique())
    logger.info("Historical universe %s to %s: %d tickers", start.date(), end.date(), len(tickers))
    return tickers


def get_universe() -> list[str]:
    return get_point_in_time_universe()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    from src import settings

    today_tickers = get_point_in_time_universe()
    print(f"\nUniverse today       : {len(today_tickers)} tickers")
    print(f"First 5               : {today_tickers[:5]}")

    old_tickers = get_point_in_time_universe("2008-01-01")
    print(f"\nUniverse on 2008-01-01: {len(old_tickers)} tickers")
    print(f"First 5               : {old_tickers[:5]}")

    historical = get_historical_tickers(settings.DATA_START, None)
    print(f"\nFull historical union since {settings.DATA_START}: {len(historical)} tickers")
    print(f"(vs. {len(today_tickers)} active today — the gap is what avoids survivorship bias)")
