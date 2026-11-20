"""Data layer: download NYC TLC yellow-taxi trip records and build an hourly demand series.

Pipeline (each step is idempotent and leaves a trace on disk):

1. ``download_month``: stream one monthly parquet file into ``data/raw/``, compute its
   SHA-256 and append it to ``configs/data_manifest.csv``. If the manifest already holds a hash
   for that month, a re-download must reproduce it, otherwise we fail loudly.
2. ``aggregate_month``: read only the pickup timestamp column, floor to the hour, count trips,
   and keep only the hours that belong to the file's own month (TLC files contain a few
   records with absurd dates such as 2002 or 2088). The monthly counts go to ``data/interim/``.
   The raw file can then be deleted: the interim file plus the manifest are enough to audit.
3. ``build_hourly_series``: concatenate the months, reindex on a complete hourly grid and
   repair the two daylight-saving artefacts of naive local timestamps (see ``repair_dst``).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

log = logging.getLogger(__name__)

MANIFEST_COLUMNS = ["month", "url", "sha256", "bytes", "n_records", "n_kept"]


def month_range(first: str, last: str) -> list[str]:
    """Inclusive list of 'YYYY-MM' strings."""
    return [p.strftime("%Y-%m") for p in pd.period_range(first, last, freq="M")]


def _sha256(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype={"month": str})
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def download_month(ym: str, url_template: str, raw_dir: Path, timeout: int = 600) -> Path:
    """Stream one monthly file to disk (skipped if already there). Returns the local path."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"yellow_tripdata_{ym}.parquet"
    if dest.exists():
        return dest
    url = url_template.format(ym=ym)
    tmp = dest.with_suffix(".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    return dest


def count_pickups_per_hour(parquet_path: Path, ym: str, ts_col: str) -> tuple[pd.Series, int]:
    """Hourly trip counts restricted to month ``ym``. Returns (counts, n_records_in_file)."""
    table = pq.read_table(parquet_path, columns=[ts_col])
    ts = table.column(ts_col).to_pandas()
    n_records = len(ts)
    start = pd.Timestamp(f"{ym}-01")
    end = start + pd.offsets.MonthBegin(1)
    ts = ts[(ts >= start) & (ts < end)]
    counts = ts.dt.floor("h").value_counts().sort_index()
    counts.index.name = "timestamp"
    counts.name = "trips"
    return counts, n_records


def aggregate_month(
    ym: str,
    url_template: str,
    ts_col: str,
    raw_dir: Path,
    interim_dir: Path,
    manifest_path: Path,
    keep_raw: bool = False,
) -> Path:
    """Download (if needed), hash, aggregate one month; returns the interim CSV path."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    out = interim_dir / f"yellow_hourly_{ym}.csv"
    if out.exists():
        return out
    path = download_month(ym, url_template, raw_dir)
    digest = _sha256(path)
    counts, n_records = count_pickups_per_hour(path, ym, ts_col)

    manifest = read_manifest(manifest_path)
    known = manifest.loc[manifest["month"] == ym, "sha256"]
    if len(known) and known.iloc[0] != digest:
        raise RuntimeError(
            f"{ym}: sha256 {digest} differs from manifest {known.iloc[0]} - upstream file changed"
        )
    if not len(known):
        row = {
            "month": ym,
            "url": url_template.format(ym=ym),
            "sha256": digest,
            "bytes": path.stat().st_size,
            "n_records": n_records,
            "n_kept": int(counts.sum()),
        }
        manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
        manifest.sort_values("month").to_csv(manifest_path, index=False)

    counts.to_csv(out)
    if not keep_raw:
        path.unlink()
    log.info("%s: %d records, %d kept, sha256=%s", ym, n_records, counts.sum(), digest[:12])
    return out


def repair_dst(s: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Repair daylight-saving artefacts of naive New-York local timestamps.

    * Spring forward: the 02:00 local hour does not exist; it holds ~0 trips (only a few
      mis-stamped records). We replace it by the mean of the two neighbouring hours.
    * Fall back: the 01:00 local hour happens twice and holds two real hours of trips.
      We halve it.

    Returns the repaired series and a log of every change (for the implementation notes).
    """
    s = s.copy()
    changes = []
    years = sorted(set(s.index.year))
    for y in years:
        # US rule since 2007: 2nd Sunday of March, 1st Sunday of November.
        march = pd.date_range(f"{y}-03-01", f"{y}-03-31", freq="D")
        spring = march[march.dayofweek == 6][1] + pd.Timedelta(hours=2)
        nov = pd.date_range(f"{y}-11-01", f"{y}-11-30", freq="D")
        fall = nov[nov.dayofweek == 6][0] + pd.Timedelta(hours=1)
        if spring in s.index and spring - pd.Timedelta(hours=1) in s.index:
            before = s.loc[spring]
            s.loc[spring] = round(
                (s.loc[spring - pd.Timedelta(hours=1)] + s.loc[spring + pd.Timedelta(hours=1)]) / 2
            )
            changes.append(("spring_forward", spring, before, s.loc[spring]))
        if fall in s.index:
            before = s.loc[fall]
            s.loc[fall] = round(before / 2)
            changes.append(("fall_back", fall, before, s.loc[fall]))
    log_df = pd.DataFrame(changes, columns=["kind", "timestamp", "before", "after"])
    return s, log_df


def build_hourly_series(interim_dir: Path, first: str, last: str) -> tuple[pd.Series, pd.DataFrame]:
    """Concatenate monthly counts onto a complete hourly grid and repair DST hours."""
    parts = []
    for ym in month_range(first, last):
        f = interim_dir / f"yellow_hourly_{ym}.csv"
        if not f.exists():
            raise FileNotFoundError(f"missing interim file for {ym}: run `w47 data` first")
        parts.append(pd.read_csv(f, parse_dates=["timestamp"], index_col="timestamp")["trips"])
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated()]
    grid = pd.date_range(
        pd.Timestamp(f"{first}-01"),
        pd.Timestamp(f"{last}-01") + pd.offsets.MonthBegin(1) - pd.Timedelta(hours=1),
        freq="h",
    )
    missing = grid.difference(s.index)
    s = s.reindex(grid).fillna(0).astype(int)
    s.index.name = "timestamp"
    s.name = "trips"
    s, dst_log = repair_dst(s)
    dst_log["n_missing_hours_filled_with_0"] = len(missing)
    return s, dst_log


def load_hourly(path: str | Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    s = df["trips"].astype(float)
    s.index.freq = "h"
    return s


def fetch_all(cfg: dict, keep_raw: bool = False, workers: int = 4) -> None:
    """Download + aggregate every month of the protocol; downloads run in parallel threads,
    hashing/aggregation/manifest writes are serialised by a lock."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    d = cfg["data"]
    raw, interim, manifest = Path("data/raw"), Path("data/interim"), Path(d["manifest"])
    lock = threading.Lock()

    def job(ym: str) -> str:
        if (interim / f"yellow_hourly_{ym}.csv").exists():
            return ym
        for attempt in range(4):
            try:
                download_month(ym, d["url_template"], raw)
                break
            except requests.RequestException as e:  # network hiccup: retry with backoff
                log.warning("%s download failed (%s), retry %d", ym, e, attempt + 1)
                import time

                time.sleep(2 ** (attempt + 2))
        with lock:
            aggregate_month(ym, d["url_template"], d["timestamp_column"], raw, interim, manifest,
                            keep_raw=keep_raw)
        return ym

    months = month_range(d["first_month"], d["last_month"])
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ym in ex.map(job, months):
            log.info("done %s", ym)
