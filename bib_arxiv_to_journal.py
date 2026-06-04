"""Replace arXiv-only bib entries with their published journal versions via ADS.

For every entry in ``master.bib`` whose ``journal`` field reads "arXiv e-prints"
(or similar), this script:
  1. Pulls the ``eprint`` (arXiv ID) from the entry.
  2. Queries ADS for the canonical bibcode of that paper.
  3. If the canonical bibcode is *not* an arXiv bibcode (i.e. ADS has the
     published version), pulls the published BibTeX from ADS and uses it to
     replace the local entry — preserving the original cite key so existing
     ``\\cite{...}`` calls in the manuscript keep working.
  4. Leaves the rest of the file byte-identical (entry-level text surgery,
     not a parser round-trip).

Defaults to dry-run; pass ``--write`` to actually modify the file (a
``master.bib.bak`` backup is created first).

Setup
-----
    export ADS_DEV_KEY=<your token>      # get one at
                                         # https://ui.adsabs.harvard.edu/user/settings/token

Usage
-----
    python bib_arxiv_to_journal.py            # dry-run, prints summary
    python bib_arxiv_to_journal.py --write    # apply changes in place
    python bib_arxiv_to_journal.py --bib other.bib --limit 5
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from typing import Iterable

import requests


ADS_BASE = "https://api.adsabs.harvard.edu/v1"
# ADS published bibcodes always carry a journal abbreviation in chars 5..9
# (e.g. "2018ApJ.."); arXiv-only bibcodes start with "arXiv" or end in the
# "arXiv" abbreviation. Match defensively against either.
ARXIV_BIBCODE_RE = re.compile(r"arxiv", re.IGNORECASE)
# A bib entry header: @TYPE{citekey,
ENTRY_HEADER_RE = re.compile(r"@([A-Za-z]+)\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)
# Field extractor — handles {value} or "value". Conservative: only used on a
# single entry slice, never on the whole file.
FIELD_RE = re.compile(
    r"(?P<name>\w+)\s*=\s*[{\"](?P<value>.*?)[}\"]\s*,?\s*$",
    re.MULTILINE,
)
ARXIV_JOURNAL_RE = re.compile(r"^\s*arxiv\b", re.IGNORECASE)
# Extract the ADS bibcode from an adsurl like "https://ui.adsabs.harvard.edu/abs/<bibcode>"
# or "https://ui.adsabs.harvard.edu/abs/<bibcode>/abstract".
ADSURL_BIBCODE_RE = re.compile(r"/abs/([^/\s}]+)")


# --------------------------------------------------------------------------- #
# Bib parsing helpers                                                         #
# --------------------------------------------------------------------------- #


def iter_entry_spans(text: str) -> Iterable[tuple[int, int, str, str]]:
    """Yield (start, end, entry_type, citekey) for every ``@type{key,...}`` block.

    ``end`` is exclusive and lands right after the entry's closing brace.
    Walks braces from each header so nested ``{...}`` in values are handled.
    """
    for m in ENTRY_HEADER_RE.finditer(text):
        entry_type = m.group(1)
        citekey = m.group(2)
        # Find the entry's opening brace (the one right after @TYPE).
        open_idx = text.index("{", m.start())
        depth = 1
        i = open_idx + 1
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
        if depth != 0:
            # Malformed — skip silently rather than crash.
            continue
        yield m.start(), i, entry_type, citekey


def parse_fields(entry_text: str) -> dict[str, str]:
    """Pull a flat {field_name_lower: value} dict from a single entry's text.

    Only used for *reading* fields (journal, eprint, etc.). Doesn't preserve
    formatting — for that we keep the raw entry slice and do text surgery.
    """
    fields: dict[str, str] = {}
    # Strip the @TYPE{key, header and the trailing }
    inner = entry_text[entry_text.index("{") + 1 : entry_text.rfind("}")]
    # Field finder that respects brace depth (so {Galv{\'a}n-Madrid} parses cleanly).
    depth = 0
    field_name: list[str] = []
    field_val_chars: list[str] = []
    state = "name"  # "name" -> "eq" -> "value" -> back to "name"
    quote_char: str | None = None
    # Skip past the cite key (already consumed): inner starts after key,
    # but iter_entry_spans gives us the full entry; let's redo:
    # Actually, parse from after the first comma at depth 0.
    first_comma = inner.find(",")
    if first_comma < 0:
        return fields
    body = inner[first_comma + 1 :]
    i = 0
    while i < len(body):
        ch = body[i]
        if state == "name":
            if ch.isspace() or ch == ",":
                i += 1
                continue
            # Read field name until '='
            j = i
            while j < len(body) and body[j] not in "=":
                j += 1
            field_name = [body[i:j].strip()]
            i = j + 1  # skip '='
            state = "value"
            field_val_chars = []
            quote_char = None
            depth = 0
            # Skip whitespace, then read opening delimiter
            while i < len(body) and body[i].isspace():
                i += 1
            if i < len(body) and body[i] in '{"':
                quote_char = "}" if body[i] == "{" else '"'
                depth = 1 if body[i] == "{" else 0
                i += 1
            else:
                # bare value (numeric); read until comma at top level
                j = i
                while j < len(body) and body[j] != ",":
                    j += 1
                fields[field_name[0].lower()] = body[i:j].strip()
                i = j
                state = "name"
                continue
            continue
        if state == "value":
            if quote_char == "}":
                if ch == "{":
                    depth += 1
                    field_val_chars.append(ch)
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        fields[field_name[0].lower()] = "".join(field_val_chars)
                        i += 1
                        state = "name"
                        continue
                    field_val_chars.append(ch)
                else:
                    field_val_chars.append(ch)
            else:  # quote_char == '"'
                if ch == '"':
                    fields[field_name[0].lower()] = "".join(field_val_chars)
                    i += 1
                    state = "name"
                    continue
                field_val_chars.append(ch)
            i += 1
            continue
    return fields


# --------------------------------------------------------------------------- #
# ADS                                                                         #
# --------------------------------------------------------------------------- #


def ads_canonical_bibcode(session: requests.Session, arxiv_id: str) -> str | None:
    """Return the canonical (published if available) bibcode for an arXiv ID.

    ADS's `identifier:` field accepts ``arXiv:<id>`` and returns the canonical
    paper. If a published version exists, ADS aliases the arXiv bibcode to it.
    """
    r = session.get(
        f"{ADS_BASE}/search/query",
        params={
            "q": f"identifier:arXiv:{arxiv_id}",
            "fl": "bibcode",
            "rows": 1,
        },
        timeout=30,
    )
    r.raise_for_status()
    docs = r.json().get("response", {}).get("docs", [])
    if not docs:
        return None
    return docs[0].get("bibcode")


def ads_bibtex(session: requests.Session, bibcode: str) -> str | None:
    """Return BibTeX text for ``bibcode`` from ADS's export endpoint."""
    r = session.post(
        f"{ADS_BASE}/export/bibtex",
        json={"bibcode": [bibcode]},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("export")


# --------------------------------------------------------------------------- #
# Surgery                                                                     #
# --------------------------------------------------------------------------- #


def patch_citekey(bibtex_block: str, new_citekey: str) -> str:
    """Replace the citekey in a one-entry BibTeX block with ``new_citekey``."""
    return ENTRY_HEADER_RE.sub(
        lambda m: f"@{m.group(1)}{{{new_citekey},", bibtex_block, count=1
    )


def is_arxiv_only(fields: dict[str, str]) -> bool:
    """True if the entry's journal field looks like an arXiv preprint marker."""
    journal = fields.get("journal", "")
    return bool(ARXIV_JOURNAL_RE.match(journal))


def local_bibcode(fields: dict[str, str]) -> str | None:
    """Best-effort extraction of the local entry's ADS bibcode.

    Looks at an explicit ``bibcode`` field first (some hand-maintained entries
    carry one); otherwise scrapes it from ``adsurl``. Returns ``None`` if no
    bibcode can be inferred — caller should treat that as "unknown, replace
    if ADS gives a publishable one".
    """
    if fields.get("bibcode"):
        return fields["bibcode"].strip()
    m = ADSURL_BIBCODE_RE.search(fields.get("adsurl", ""))
    if m:
        return m.group(1)
    return None


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bib", default="master.bib", help="path to bib file")
    p.add_argument("--write", action="store_true",
                   help="apply changes in place (backs up to <bib>.bak first); "
                        "default is dry-run")
    p.add_argument("--all-eprints", action="store_true",
                   help="process every entry with an `eprint` field, not just "
                        "ones whose journal looks like 'arXiv e-prints'. Entries "
                        "whose local ADS bibcode already matches ADS's canonical "
                        "one are still skipped, so this mainly catches "
                        "@INPROCEEDINGS / misc cases the narrow heuristic missed.")
    p.add_argument("--limit", type=int, default=0,
                   help="cap on number of entries to process (0 = all)")
    p.add_argument("--sleep", type=float, default=0.15,
                   help="seconds to sleep between ADS calls (rate limit hygiene)")
    args = p.parse_args(argv)

    token = os.environ.get("ADS_DEV_KEY") or os.environ.get("ADS_API_KEY")
    if not token:
        sys.stderr.write(
            "ERROR: ADS_DEV_KEY (or ADS_API_KEY) is not set. Get a token at\n"
            "       https://ui.adsabs.harvard.edu/user/settings/token\n"
        )
        return 2

    with open(args.bib, "r", encoding="utf-8") as f:
        text = f.read()

    # Walk entries once, collect (start, end, citekey, fields) for the ones we'll touch.
    # Default scope: journal field looks like arXiv. --all-eprints widens to
    # "any entry with an eprint" (so @INPROCEEDINGS, @MISC, etc. are included).
    targets: list[tuple[int, int, str, dict[str, str]]] = []
    for start, end, _entry_type, citekey in iter_entry_spans(text):
        block = text[start:end]
        fields = parse_fields(block)
        if not fields.get("eprint"):
            continue
        if args.all_eprints or is_arxiv_only(fields):
            targets.append((start, end, citekey, fields))

    if args.limit:
        targets = targets[: args.limit]
    scope = "entries with an eprint" if args.all_eprints else "arXiv-only entries with an eprint"
    print(f"Found {len(targets)} {scope} in {args.bib}")

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    session.headers["Accept"] = "application/json"

    # Process targets back-to-front so byte offsets of earlier targets remain valid
    # when we splice replacements in.
    replacements: list[tuple[int, int, str]] = []  # (start, end, new_block)
    n_replaced = n_still_preprint = n_not_found = n_failed = n_already_canonical = 0

    for idx, (start, end, citekey, fields) in enumerate(targets, 1):
        eprint = fields["eprint"].strip()
        prefix = f"[{idx}/{len(targets)}] {citekey} (arXiv:{eprint})"
        try:
            bibcode = ads_canonical_bibcode(session, eprint)
        except requests.HTTPError as e:
            print(f"{prefix}  ADS query failed: {e}")
            n_failed += 1
            time.sleep(args.sleep)
            continue
        if not bibcode:
            print(f"{prefix}  -> not found in ADS")
            n_not_found += 1
            time.sleep(args.sleep)
            continue
        if ARXIV_BIBCODE_RE.search(bibcode):
            print(f"{prefix}  -> {bibcode} (still preprint, skip)")
            n_still_preprint += 1
            time.sleep(args.sleep)
            continue
        # If the local entry already cites this exact bibcode, the published
        # version is already what we have — re-exporting would just churn
        # formatting. Skip.
        local_bc = local_bibcode(fields)
        if local_bc and local_bc.lower() == bibcode.lower():
            print(f"{prefix}  -> {bibcode} (already canonical, skip)")
            n_already_canonical += 1
            time.sleep(args.sleep)
            continue
        try:
            bibtex = ads_bibtex(session, bibcode)
        except requests.HTTPError as e:
            print(f"{prefix}  -> {bibcode}  but export failed: {e}")
            n_failed += 1
            time.sleep(args.sleep)
            continue
        if not bibtex or "@" not in bibtex:
            print(f"{prefix}  -> {bibcode}  but export returned no bibtex")
            n_failed += 1
            time.sleep(args.sleep)
            continue
        new_block = patch_citekey(bibtex.strip(), citekey)
        replacements.append((start, end, new_block))
        print(f"{prefix}  -> {bibcode} (replace)")
        n_replaced += 1
        time.sleep(args.sleep)

    print()
    print(f"Summary: replaced={n_replaced}  still-preprint={n_still_preprint}  "
          f"already-canonical={n_already_canonical}  "
          f"not-found={n_not_found}  failed={n_failed}")

    if not args.write:
        if replacements:
            print("\n(dry run — pass --write to apply)")
        return 0

    if not replacements:
        print("Nothing to write.")
        return 0

    # Apply replacements back-to-front so earlier offsets stay valid.
    new_text = text
    for start, end, new_block in sorted(replacements, key=lambda x: x[0], reverse=True):
        new_text = new_text[:start] + new_block + new_text[end:]

    backup = args.bib + ".bak"
    shutil.copy2(args.bib, backup)
    with open(args.bib, "w", encoding="utf-8") as f:
        f.write(new_text)
    print(f"\nWrote {args.bib} ({n_replaced} replacements); backup at {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
