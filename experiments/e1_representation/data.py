"""wikitext-2 raw test, fetched without the `datasets` package.

The HF datasets-server returns rows as JSON, which avoids needing a parquet
engine. Cached to disk after the first fetch.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

CACHE = Path(__file__).parent / "wikitext2-test.txt"
API = (
    "https://datasets-server.huggingface.co/rows?dataset=Salesforce%2Fwikitext"
    "&config=wikitext-2-raw-v1&split=test&offset={off}&length={n}"
)


def wikitext2_test() -> str:
    if CACHE.exists():
        return CACHE.read_text(encoding="utf-8")
    rows, off, total = [], 0, None
    while total is None or off < total:
        d = None
        for attempt in range(6):  # WSL DNS is intermittently flaky here
            try:
                with urllib.request.urlopen(API.format(off=off, n=100), timeout=60) as r:
                    d = json.load(r)
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 * (attempt + 1))
        total = d["num_rows_total"] if total is None else total
        batch = d["rows"]
        if not batch:
            break
        rows.extend(x["row"]["text"] for x in batch)
        off += len(batch)
    text = "".join(rows)
    CACHE.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    t = wikitext2_test()
    print(f"{len(t):,} chars cached at {CACHE}")
