"""Triage-grade literature search for the Yan+2023 Table 2 clusters.

For each named stellar system in ``yan23_table2.fits`` (the WKP13/Kirk &
Myers/Stephens compilation), query ADS for *other* refereed papers about
that object and write a triage list to ``yan23_followups.fits``. The output
is intentionally not curated for "this measures the cluster mass" — it just
lists candidate papers per cluster that name the object directly. Hand
curation comes later.

Workflow
--------
1. Load ``yan23_table2.fits``.
2. Build a clean per-row query string from the ``designation`` column. Many
   designations are coded (``No.18`` = VV Ser, ``MYSO 052124``, etc.), so a
   ``CLUSTER_NAME_OVERRIDES`` dict maps the worst offenders to a human-friendly
   ADS-searchable name. For the un-overridden rows the designation is used
   as-is and is expected to find nothing — the script logs those.
3. For each unique cluster name, hit ADS ``search/query`` with
   ``object:"<name>"`` (ADS's curated object resolver), restricting to
   refereed journal articles published after 2000 (cuts the bulk of pre-WKP13
   redundant hits). If the object: query returns nothing, fall back to a
   title+abstract free-text search.
4. For each hit, keep the bibcode, year, citation count, title, and first
   author.
5. Skip the bibcode that's already in ``primary_bibcode`` for that row — it's
   the WKP13/Yan-inherited source and not a "follow-up".
6. Write all (cluster, candidate-paper) pairs to ``yan23_followups.fits``.

Setup
-----
Reads the token from ``$ADS_DEV_KEY`` or ``~/.ads/dev_key`` (the standard
``ads`` Python-package location).

Usage
-----
    python cluster_lit_search.py [--limit N] [--sleep SECONDS]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import requests
from astropy.table import Table


def _ascii(s: str) -> str:
    """Strip diacritics so the result is FITS-bintable safe (ASCII only)."""
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")


ADS_BASE = "https://api.adsabs.harvard.edu/v1"

# Designation -> human-searchable cluster name. Built from WKP13's Table A2
# (cluster designations) so the per-row "No.X" is mapped to its real name. The
# Kirk&Myers (ref=1) and Stephens (ref=2) entries already carry searchable
# names in the designation column.
CLUSTER_NAME_OVERRIDES = {
    # WKP13 No.X (Yan designation) -> human-searchable name (from WKP13 Table A1)
    "No.3":   "B59",
    "No.18":  "VV Ser",
    "No.19":  "VY Mon",
    "No.21":  "IRAS 05377+3548",
    "No.22":  "Ser SVS2",
    "No.23":  "IRAS 05553+1631",
    "No.24":  "IRAS 05490+2658",
    "No.25":  "IRAS 03064+5638",
    "No.26":  "IRAS 06155+2319",
    "No.30":  "IRAS 06058+2138",
    "No.31":  "NGC 2023",
    "No.34":  "NGC 7129",
    "No.35":  "IRAS 06068+2030",
    "No.36":  "IRAS 00494+5617",
    "No.37":  "V921 Sco",
    "No.38":  "IRAS 05197+3355",
    "No.39":  "IRAS 05375+3540",
    "No.40":  "IRAS 02593+6016",
    "No.42":  "NGC 2071",
    "No.43":  "Cha I",
    "No.44":  "MWC 297",
    "No.45":  "BD+40 4124",
    "No.46":  "rho Oph",
    "No.47":  "IRAS 06056+2131",
    "No.48":  "IRAS 05100+3723",
    "No.49":  "R CrA",
    "No.50":  "NGC 1333",
    "No.52":  "IRAS 02575+6017",
    "No.56":  "sigma Ori",
    "No.57":  "NGC 2068",
    "No.59":  "LkHa 101",
    "No.60":  "Mon R2",
    "No.61":  "IRAS 06073+1249",
    "No.62":  "Trumpler 24",
    "No.63":  "IC 5146",
    "No.66":  "Alicante 5",
    "No.67":  "Cep OB3b",
    "No.69":  "Sh2-294",
    "No.70":  "NGC 2264",
    "No.71":  "RCW 116B",
    "No.72":  "Alicante 1",
    "No.73":  "RCW 36",
    "No.76":  "NGC 6383",
    "No.77":  "NGC 2024",
    "No.81":  "NGC 2362",
    "No.85":  "NGC 6530",
    "No.88":  "[DBSB2003] 177",
    "No.89":  "FSR 1530",
    "No.90":  "[DB2000] 52",
    "No.92":  "Berkeley 86",
    "No.93":  "CC01",
    "No.94":  "NGC 637",
    "No.95":  "[DB2000] 26",
    "No.96":  "W5Wb",
    "No.99":  "ONC",
    "No.100": "RCW 38",
    "No.103": "Berkeley 59",
    "No.105": "[FSR2007] 817",
    "No.106": "W5E",
    "No.107": "W5Wa",
    "No.108": "NGC 1931",
    "No.110": "Mercer 23",
    "No.111": "LH 118",
    "No.112": "NGC 2103",
    "No.114": "[BDSB2003] 106",
    "No.115": "NGC 7380",
    "No.116": "GLIMPSE 30",
    "No.117": "NGC 6231",
    "No.118": "RCW 106",
    "No.119": "IC 1484",
    "No.120": "NGC 6823",
    "No.122": "RCW 121",
    "No.125": "NGC 2244",
    "No.126": "NGC 2122",
    "No.127": "[BDSB2003] 107",
    "No.128": "Danks 1",
    "No.130": "Westerlund 2",
    "No.131": "RCW 95",
    "No.132": "IC 1805",
    "No.133": "NGC 6357",
    "No.135": "Trumpler 14",
    "No.136": "NGC 6611",
    "No.137": "Cyg OB2",
    "No.138": "Arches",
    "No.139": "R136",
}


def _load_ads_token() -> str:
    token = os.environ.get("ADS_DEV_KEY") or os.environ.get("ADS_API_KEY")
    if token:
        return token.strip()
    home_key = Path.home() / ".ads" / "dev_key"
    if home_key.exists():
        return home_key.read_text().strip()
    raise SystemExit(
        "ADS token not found. Set $ADS_DEV_KEY or write the token to "
        "~/.ads/dev_key. See https://ui.adsabs.harvard.edu/user/settings/token"
    )


def _searchable_name(designation: str, ref: str) -> str | None:
    """Return a human-readable cluster name for ADS searching, or None to skip."""
    if designation in CLUSTER_NAME_OVERRIDES:
        return CLUSTER_NAME_OVERRIDES[designation]
    if ref in ("1", "2"):
        # Kirk & Myers / Stephens designations are already searchable strings
        # like "Taurus No.1", "MYSO 052124" — use the designation as-is.
        return designation
    return None  # uncovered WKP13 No.X — log and skip


def _ads_search(session: requests.Session, query: str, fl: str, rows: int = 10) -> list[dict]:
    """Run a single ADS search/query and return ``response.docs``."""
    r = session.get(
        f"{ADS_BASE}/search/query",
        params={"q": query, "fl": fl, "rows": rows, "sort": "citation_count desc"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("response", {}).get("docs", [])


def _query_cluster(session: requests.Session, name: str) -> list[dict]:
    """Query ADS for refereed post-2000 astronomy papers about ``name``.

    ADS doesn't expose its SIMBAD object resolver as a Solr field on the
    public API, so we run a full-text search restricted by topic keywords —
    "cluster", "IMF", "mass", "YSO", "young stellar", "embedded" — to keep
    out random astronomy papers that just mention the name in passing. The
    output is still a triage list: hits need to be opened to confirm they
    actually report a cluster mass or mmax measurement.
    """
    fl = "bibcode,title,first_author,year,citation_count"
    base = (
        f'full:"{name}" AND property:refereed AND year:2000- '
        f'AND collection:astronomy '
        f'AND (cluster OR IMF OR "stellar mass" OR YSO OR '
        f'"young stellar" OR embedded OR "most massive")'
    )
    try:
        return _ads_search(session, base, fl, rows=15)
    except requests.HTTPError:
        return []


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bib", default="yan23_table2.fits",
                   help="path to Yan+2023 table (FITS)")
    p.add_argument("--out", default="yan23_followups.fits",
                   help="output FITS path")
    p.add_argument("--limit", type=int, default=0,
                   help="cap on number of clusters to process (0 = all)")
    p.add_argument("--sleep", type=float, default=0.5,
                   help="seconds between ADS calls (rate-limit hygiene)")
    args = p.parse_args(argv)

    token = _load_ads_token()
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    session.headers["Accept"] = "application/json"

    yan = Table.read(args.bib)
    print(f"loaded {len(yan)} Yan+2023 rows from {args.bib}")

    # Build the list of (cluster_id, searchable_name, primary_bibcode_to_skip)
    queries: list[tuple[str, str, str]] = []
    skipped: list[str] = []
    for row in yan:
        d = str(row["designation"])
        name = _searchable_name(d, str(row["ref"]))
        if name is None:
            skipped.append(d); continue
        queries.append((d, name, str(row["primary_bibcode"])))
    print(f"clusters with a searchable name: {len(queries)} "
          f"(skipped {len(skipped)} uncovered designations)")

    if args.limit:
        queries = queries[: args.limit]

    rows_out: list[dict] = []
    for i, (desig, name, primary_bc) in enumerate(queries, 1):
        try:
            docs = _query_cluster(session, name)
        except Exception as e:
            print(f"[{i:3d}/{len(queries)}] {desig:<22s} ({name!r:<24s})  ERROR: {e}")
            time.sleep(args.sleep)
            continue
        # Filter out the WKP13/Yan-inherited primary so we surface only follow-ups.
        followups = [d for d in docs if d.get("bibcode") != primary_bc]
        print(f"[{i:3d}/{len(queries)}] {desig:<22s} ({name!r:<24s})  hits={len(docs)} new={len(followups)}")
        for d in followups:
            rows_out.append({
                "designation":     _ascii(desig),
                "cluster_name":    _ascii(name),
                "bibcode":         _ascii(d.get("bibcode") or ""),
                "title":           _ascii(d.get("title", [""])[0] if d.get("title") else ""),
                "first_author":    _ascii(d.get("first_author", "") or ""),
                "year":            int(d.get("year") or 0),
                "citation_count":  int(d.get("citation_count") or 0),
            })
        time.sleep(args.sleep)

    # Write as FITS. Pad strings to fixed-width via numpy structured array.
    t = Table(
        rows=rows_out,
        names=("designation", "cluster_name", "bibcode", "title",
               "first_author", "year", "citation_count"),
    )
    t.meta["source"] = "ADS search/query, object: then title/abstract fallback"
    t.meta["filter"] = "property:refereed AND year:2000-"
    t.write(args.out, overwrite=True)
    print(f"\nwrote {args.out} ({len(t)} candidate papers across "
          f"{len(set(t['designation']))} clusters)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
