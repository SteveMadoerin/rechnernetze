"""Download multiple years of NVDA 1-minute intraday candles from the LSE API
and save them as a single CSV in "E:\\Golden Predict Agent\\lse API".

The LSE API caps every `candles` call at 5,000 rows, while a year of 1-minute
bars is roughly 100,000 rows. We therefore page through the range: fetch a
batch in ascending order, then move the start cursor just past the last bar we
received and fetch again, until no rows are left.

We ask for a generous look-back window (YEARS_BACK) and simply keep paging from
the oldest requested date toward now. The server returns the earliest data it
actually has, so requesting more years than exist is harmless — you just get
everything available. Override the defaults from the command line:

    python download_stock_from_lse.py --years 5 --symbol NVDA --timeframe 1m
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from lse import LSE, LSEError

# --- Configuration ----------------------------------------------------------
API_KEY = "lse_live_acc6a45e758b0098098771efad152644"
SYMBOL = "NVDA"
TIMEFRAME = "1m"
YEARS_BACK = 10            # how far back to try; the loop stops when data runs out
PAGE_LIMIT = 5000          # server hard cap per request
REQUEST_PAUSE = 0.7        # seconds between calls (API allows 100/min ≈ one per 0.6s)
RATE_LIMIT_WAIT = 60       # seconds to back off when the API returns HTTP 429
OUTPUT_DIR = Path(r"E:\Golden Predict Agent\lse API")

# CSV columns in the order they appear in each candle dict.
FIELDS = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]


def _fmt(dt: datetime) -> str:
    """Format a datetime as a UTC "Z" timestamp the API accepts.

    The API's query encoder turns a literal "+" offset into a space and then
    rejects it, so we never send "+HH:MM" offsets — we convert to UTC and use
    a trailing "Z" instead.
    """
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _candles_with_retry(client: LSE, symbol: str, timeframe: str,
                        start: str, end: str) -> list[dict]:
    """Call client.candles, backing off and retrying on HTTP 429 rate limits."""
    while True:
        try:
            return client.candles(symbol, timeframe, start=start, end=end,
                                   limit=PAGE_LIMIT, order="asc")
        except LSEError as e:
            if e.status == 429:
                print(f"  rate limited (429); waiting {RATE_LIMIT_WAIT}s ...")
                time.sleep(RATE_LIMIT_WAIT)
                continue
            raise


def fetch_all_candles(client: LSE, symbol: str, timeframe: str,
                      start: datetime, end: datetime) -> list[dict]:
    """Page through [start, end) and return all candles, oldest first.

    Bars are fetched ascending in chunks of PAGE_LIMIT. After each chunk the
    cursor advances to one second past the newest bar received so the next
    request never re-reads the boundary row. A timestamp set guards against
    any residual duplicates.
    """
    rows: list[dict] = []
    seen: set[str] = set()
    cursor = start

    while cursor < end:
        batch = _candles_with_retry(client, symbol, timeframe,
                                    start=_fmt(cursor), end=_fmt(end))
        if not batch:
            break

        new_in_batch = 0
        for row in batch:
            ts = row["timestamp"]
            if ts not in seen:
                seen.add(ts)
                rows.append(row)
                new_in_batch += 1

        last_ts = datetime.fromisoformat(batch[-1]["timestamp"])
        cursor = last_ts + timedelta(seconds=1)
        print(f"  fetched {len(batch):>5} bars (+{new_in_batch} new), "
              f"total {len(rows):>6}, up to {batch[-1]['timestamp']}")

        # If the server returned fewer than the cap, we have reached the end.
        if len(batch) < PAGE_LIMIT:
            break
        time.sleep(REQUEST_PAUSE)  # stay under the API's per-minute call limit

    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    """Write candles to `path` as CSV with a header row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default=SYMBOL, help=f"ticker (default {SYMBOL})")
    p.add_argument("--timeframe", default=TIMEFRAME,
                   help=f"1m/5m/15m/1h/4h/1d (default {TIMEFRAME})")
    p.add_argument("--years", type=float, default=YEARS_BACK,
                   help=f"how many years back to try (default {YEARS_BACK})")
    p.add_argument("--start", default=None,
                   help="explicit start date YYYY-MM-DD (overrides --years)")
    p.add_argument("--end", default=None,
                   help="explicit end date YYYY-MM-DD (defaults to now)")
    return p.parse_args()


def _parse_day(day: str) -> datetime:
    """Parse a YYYY-MM-DD string as a UTC midnight datetime."""
    return datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def main() -> None:
    args = parse_args()
    client = LSE(api_key=API_KEY)

    end = _parse_day(args.end) if args.end else datetime.now(timezone.utc)
    if args.start:
        start = _parse_day(args.start)
    else:
        start = end - timedelta(days=round(args.years * 365.25))

    print(f"Downloading {args.symbol} {args.timeframe} candles "
          f"from {start.date()} to {end.date()} ...")

    try:
        rows = fetch_all_candles(client, args.symbol, args.timeframe, start, end)
    except LSEError as e:
        raise SystemExit(f"LSE API error {e.status}: {e}")

    if not rows:
        print("No candles returned for the requested range.")
        return

    # Name the file after the data we actually got, not what we asked for.
    first_day = rows[0]["timestamp"][:10]
    last_day = rows[-1]["timestamp"][:10]
    out_path = OUTPUT_DIR / f"{args.symbol}_{args.timeframe}_{first_day}_{last_day}.csv"
    write_csv(rows, out_path)
    print(f"Saved {len(rows)} candles ({first_day} to {last_day}) to {out_path}")


if __name__ == "__main__":
    main()
