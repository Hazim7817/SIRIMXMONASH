"""Load the FDA DILIrank dataset into reference_outcome.

DILIrank classifies drugs by their potential to cause drug-induced liver
injury (DILI) in humans, based on FDA drug labels and the literature:

    Most-DILI-concern   -> positive (toxic)
    Less-DILI-concern   -> positive by default (see --less-concern-as)
    No-DILI-concern     -> negative
    Ambiguous           -> ambiguous (not scored)

Download the spreadsheet from the FDA Liver Toxicity Knowledge Base page
(search "DILIrank dataset fda.gov"), then run, from the ooc_db folder:

    python -m loaders.load_dilirank --file DILIrank.xlsx

Re-running it updates existing DILIrank rows instead of duplicating them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import re
import sys
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from loaders import db

SOURCE = "DILIrank"
CITATION = (
    "Chen M, Suzuki A, Thakkar S, Yu K, Hu C, Tong W. DILIrank: the largest "
    "reference drug list ranked by the risk for developing drug-induced liver "
    "injury in humans. Drug Discov Today. 2016;21(4):648-653."
)
CATEGORIES = ("most", "less", "no", "ambiguous")


@dataclass
class Entry:
    name: str
    category: str              # one of CATEGORIES
    raw_category: str
    ltkbid: str | None = None
    severity: str | None = None
    label_section: str | None = None


def _key(text: object) -> str:
    """Lower-case letters and digits only, for forgiving header matching."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _category(raw: str) -> str | None:
    k = _key(raw)
    if not k or k == "nan":
        return None
    if "ambiguous" in k:
        return "ambiguous"
    for cat in ("most", "less", "no"):
        # vMost-DILI-Concern, Most DILI concern, vNo-DILI-Concern ...
        if re.fullmatch(rf"v?{cat}diliconcern", k):
            return cat
    raise ValueError(f"Unrecognised DILI concern category: {raw!r}")


def _read_table(path: pathlib.Path) -> pd.DataFrame:
    if path.suffix.lower() in (".csv", ".txt", ".tsv"):
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        return pd.read_csv(path, header=None, dtype=str, sep=sep, keep_default_na=False)
    return pd.read_excel(path, header=None, dtype=str, keep_default_na=False)


def parse(path: pathlib.Path) -> list[Entry]:
    """Read DILIrank entries, finding the header row and columns by name."""
    raw = _read_table(path)

    header_row = None
    for i in range(min(len(raw), 20)):
        if "compoundname" in {_key(v) for v in raw.iloc[i]}:
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"{path}: no 'Compound Name' header found in the first 20 rows")

    headers = [_key(v) for v in raw.iloc[header_row]]

    def column(*candidates: str, contains: str | None = None, required: bool = True) -> int | None:
        for i, h in enumerate(headers):
            if h in candidates or (contains and contains in h):
                return i
        if required:
            raise ValueError(f"{path}: missing column {candidates or contains}; found {headers}")
        return None

    name_col = column("compoundname")
    concern_col = column(contains="diliconcern")
    id_col = column("ltkbid", required=False)
    severity_col = column(contains="severity", required=False)
    label_col = column(contains="labelsection", required=False)

    def cell(row, col):
        if col is None:
            return None
        value = str(row.iloc[col]).strip()
        return value or None

    entries = []
    for _, row in raw.iloc[header_row + 1:].iterrows():
        name = cell(row, name_col)
        raw_cat = cell(row, concern_col)
        if not name or not raw_cat:
            continue  # blank lines, footnotes
        category = _category(raw_cat)
        if category is None:
            continue
        entries.append(Entry(
            name=" ".join(name.split()),
            category=category,
            raw_category=raw_cat,
            ltkbid=cell(row, id_col),
            severity=cell(row, severity_col),
            label_section=cell(row, label_col),
        ))
    if not entries:
        raise ValueError(f"{path}: no DILIrank entries found")
    return entries


def merge_duplicates(entries: list[Entry]) -> tuple[list[Entry], list[str]]:
    """One entry per drug name. Conflicting duplicates become ambiguous."""
    by_name: dict[str, list[Entry]] = {}
    for e in entries:
        by_name.setdefault(e.name.lower(), []).append(e)

    merged, warnings = [], []
    for group in by_name.values():
        first = group[0]
        cats = {e.category for e in group}
        if len(cats) > 1:
            warnings.append(
                f"{first.name}: listed {len(group)} times with different categories "
                f"({', '.join(sorted(e.raw_category for e in group))}); loaded as ambiguous"
            )
            first = Entry(first.name, "ambiguous",
                          " / ".join(e.raw_category for e in group), first.ltkbid,
                          first.severity, first.label_section)
        merged.append(first)
    return merged, warnings


def verdict_for(category: str, less_concern_as: str) -> str:
    return {
        "most": "positive",
        "less": less_concern_as,
        "no": "negative",
        "ambiguous": "ambiguous",
    }[category]


def load(conn, entries: list[Entry], *, less_concern_as: str = "positive",
         only_existing: bool = False, retrieved_on: dt.date | None = None) -> Counter:
    retrieved_on = retrieved_on or dt.date.today()
    stats: Counter = Counter()
    for e in entries:
        if only_existing:
            drug_id = db.find_drug(conn, e.name)
            if drug_id is None:
                stats["skipped (drug not in database)"] += 1
                continue
        else:
            drug_id, created = db.get_or_create_drug(conn, e.name)
            stats["drugs created"] += created

        details = [e.raw_category]
        if e.severity:
            details.append(f"severity class {e.severity}")
        if e.label_section:
            details.append(f"label section: {e.label_section}")

        db.upsert_reference(conn, {
            "drug_id": drug_id,
            "endpoint": "toxicity",
            "organ": "liver",
            "species": "human",
            "evidence_type": "FDA drug labeling and literature",
            "source": SOURCE,
            "finding": "; ".join(details),
            "verdict": verdict_for(e.category, less_concern_as),
            "citation": CITATION,
            "use_for_scoring": True,
            "method": "database",
            "source_record_id": e.ltkbid,
            "retrieved_on": retrieved_on,
        })
        stats[f"loaded: {e.category} concern"] += 1
    return stats


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", required=True, type=pathlib.Path, help="DILIrank .xlsx or .csv")
    p.add_argument("--dsn", help=f"database URL (default: ${db.DSN_ENV})")
    p.add_argument("--less-concern-as", choices=("positive", "ambiguous", "negative"), default="positive",
                   help="verdict for Less-DILI-concern drugs (default: positive)")
    p.add_argument("--only-existing", action="store_true",
                   help="only load drugs already in the drug table; do not add new drugs")
    p.add_argument("--dry-run", action="store_true", help="parse and report, write nothing")
    args = p.parse_args(argv)

    entries, warnings = merge_duplicates(parse(args.file))
    for w in warnings:
        print("warning:", w, file=sys.stderr)
    counts = Counter(e.category for e in entries)
    print(f"Read {len(entries)} drugs from {args.file}: "
          + ", ".join(f"{counts[c]} {c}" for c in CATEGORIES))

    if args.dry_run:
        return
    with db.connect(args.dsn) as conn:
        stats = load(conn, entries, less_concern_as=args.less_concern_as,
                     only_existing=args.only_existing)
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
