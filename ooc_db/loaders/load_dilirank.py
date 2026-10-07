"""Load the FDA DILIrank dataset into reference_outcome.

DILIrank classifies drugs by their potential to cause drug-induced liver
injury (DILI) in humans, based on FDA drug labels and the literature:

    Most-DILI-concern   -> positive (toxic)
    Less-DILI-concern   -> positive by default (see --less-concern-as)
    No-DILI-concern     -> negative
    Ambiguous           -> ambiguous (not scored)

Run from the ooc_db folder, either downloading the current DILIrank 2.0
file from the FDA or using a copy you downloaded yourself:

    python -m loaders.load_dilirank --download
    python -m loaders.load_dilirank --file DILIrank.xlsx

The DILIrank 2.0 workbook has two sheets, 'version 2' (current, 1,336
drugs) and 'version 1' (the original 1,036); 'version 2' is used unless you
pass --sheet.

Each load replaces the previous DILIrank list: rows are updated rather than
duplicated, and DILIrank rows for drugs not in the loaded file (e.g. after
switching between --sheet "version 1" and "version 2") are removed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import pathlib
import re
import sys
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from loaders import db

SOURCE = "DILIrank"
DOWNLOAD_URL = "https://www.fda.gov/media/113052/download"
PREFERRED_SHEET = "version 2"
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
    comment: str | None = None   # DILIrank 2.0: Unchanged / New / Revised


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


def _is_text(path: pathlib.Path) -> bool:
    return path.suffix.lower() in (".csv", ".txt", ".tsv")


def resolve_sheet(path: pathlib.Path, sheet: str | None = None) -> str | None:
    """The worksheet to read: `sheet` if given, else 'version 2' if present, else the first."""
    if _is_text(path):
        return None
    with pd.ExcelFile(path) as book:
        if sheet is None:
            return PREFERRED_SHEET if PREFERRED_SHEET in book.sheet_names else book.sheet_names[0]
        if sheet not in book.sheet_names:
            raise ValueError(f"{path}: no sheet {sheet!r}; sheets are {book.sheet_names}")
        return sheet


def _read_text_table(path: pathlib.Path) -> pd.DataFrame:
    """CSV/TXT/TSV as saved by Excel: UTF-8 or Windows encoding, comma,
    semicolon or tab separated, title rows narrower than the table."""
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = data.decode("latin-1")
    # The delimiter is whichever splits some early line (the header) into the
    # most columns; title rows above the header may contain none at all.
    sample = text.splitlines()[:30]
    preferred = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    delimiter = max((preferred, ",", ";", "\t"),
                    key=lambda d: max((len(next(csv.reader([line], delimiter=d), []))
                                       for line in sample), default=0))
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    width = max((len(r) for r in rows), default=0)
    return pd.DataFrame([r + [""] * (width - len(r)) for r in rows], dtype=str)


def _read_table(path: pathlib.Path, sheet: str | None) -> pd.DataFrame:
    if _is_text(path):
        return _read_text_table(path)
    return pd.read_excel(path, sheet_name=resolve_sheet(path, sheet), header=None, dtype=str,
                         keep_default_na=False)


def parse(path: pathlib.Path, sheet: str | None = None) -> list[Entry]:
    """Read DILIrank entries, finding the header row and columns by name.

    Handles the DILIrank 1.0 and 2.0 spellings ('Compound Name' or
    'CompoundName', 'vDILIConcern' or 'vDILI-Concern', with or without the
    'v' prefix on categories, any capitalisation).
    """
    raw = _read_table(path, sheet)

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
    comment_col = column("comment", required=False)

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
            comment=cell(row, comment_col),
        ))
    if not entries:
        raise ValueError(f"{path}: no DILIrank entries found")
    return entries


def merge_duplicates(entries: list[Entry], less_concern_as: str = "positive"
                     ) -> tuple[list[Entry], list[str]]:
    """One entry per drug name.

    Duplicates with the same verdict keep the most serious category;
    duplicates with different verdicts become ambiguous.
    """
    by_name: dict[str, list[Entry]] = {}
    for e in entries:
        by_name.setdefault(e.name.lower(), []).append(e)

    merged, warnings = [], []
    for group in by_name.values():
        verdicts = {verdict_for(e.category, less_concern_as) for e in group}
        if len(verdicts) > 1:
            first = group[0]
            warnings.append(
                f"{first.name}: listed {len(group)} times with different categories "
                f"({', '.join(sorted(e.raw_category for e in group))}); loaded as ambiguous"
            )
            merged.append(Entry(first.name, "ambiguous",
                                " / ".join(e.raw_category for e in group), first.ltkbid,
                                first.severity, first.label_section, first.comment))
        else:
            merged.append(min(group, key=lambda e: CATEGORIES.index(e.category)))
    return merged, warnings


def verdict_for(category: str, less_concern_as: str) -> str:
    return {
        "most": "positive",
        "less": less_concern_as,
        "no": "negative",
        "ambiguous": "ambiguous",
    }[category]


def _finding(e: Entry) -> str:
    details = [e.raw_category]
    if e.severity:
        details.append(f"severity class {e.severity}")
    if e.label_section:
        details.append(f"label section: {e.label_section}")
    if e.comment:
        details.append(f"DILIrank 2.0: {e.comment}")
    return "; ".join(details)


def load(conn, entries: list[Entry], *, less_concern_as: str = "positive",
         only_existing: bool = False, retrieved_on: dt.date | None = None) -> Counter:
    """Write one DILIrank reference row per drug and remove DILIrank rows for
    drugs not in `entries`, so the database matches the file just loaded."""
    retrieved_on = retrieved_on or dt.date.today()
    stats: Counter = Counter()
    loaded: dict[int, tuple[Entry, str]] = {}   # drug_id -> (entry, verdict)

    for e in entries:
        if only_existing:
            drug_id = db.find_drug(conn, e.name)
            if drug_id is None:
                stats["skipped (drug not in database)"] += 1
                continue
        else:
            drug_id, created = db.get_or_create_drug(conn, e.name)
            stats["drugs created"] += created

        verdict = verdict_for(e.category, less_concern_as)
        finding = _finding(e)
        if drug_id in loaded:
            # Two DILIrank names for one drug (e.g. via drug_alias).
            other, other_verdict = loaded[drug_id]
            if other_verdict == verdict:
                stats["merged: same drug under two names"] += 1
                continue
            print(f"warning: DILIrank lists '{other.name}' ({other.raw_category}) and "
                  f"'{e.name}' ({e.raw_category}), which are the same drug in the database; "
                  "loaded as ambiguous", file=sys.stderr)
            verdict = "ambiguous"
            finding = f"{other.raw_category} / {e.raw_category}; conflicting DILIrank entries " \
                      f"'{other.name}' and '{e.name}'"
            stats["merged: conflicting names for one drug"] += 1
        loaded[drug_id] = (e, verdict)

        db.upsert_reference(conn, {
            "drug_id": drug_id,
            "endpoint": "toxicity",
            "organ": "liver",
            "species": "human",
            "evidence_type": "FDA drug labeling and literature",
            "source": SOURCE,
            "finding": finding,
            "verdict": verdict,
            "citation": CITATION,
            "use_for_scoring": True,
            "method": "database",
            "source_record_id": e.ltkbid,
            "retrieved_on": retrieved_on,
        })
        stats[f"loaded: {e.category} concern"] += 1

    removed = conn.execute(
        """DELETE FROM reference_outcome
           WHERE source = %s AND method = 'database' AND endpoint = 'toxicity'
             AND organ = 'liver' AND species = 'human' AND NOT (drug_id = ANY(%s))""",
        (SOURCE, list(loaded)),
    ).rowcount
    if removed:
        stats["removed: earlier DILIrank rows for drugs not in this file"] = removed

    hand_typed = conn.execute(
        "SELECT count(*) FROM reference_outcome WHERE source = %s AND method = 'curated'",
        (SOURCE,),
    ).fetchone()[0]
    if hand_typed:
        print(f"warning: {hand_typed} hand-entered reference rows have source 'DILIrank' "
              "(method 'curated'). They are kept alongside the imported rows and may "
              "duplicate or contradict them; change their method to 'database' or delete them.",
              file=sys.stderr)
    return stats


def download(dest: pathlib.Path = pathlib.Path("DILIrank_2.0.xlsx")) -> pathlib.Path:
    import requests

    advice = ("Download it in a browser from the FDA page 'Drug-Induced Liver Injury Rank "
              "(DILIrank 2.0) Dataset' and use --file instead.")
    try:
        r = requests.get(DOWNLOAD_URL, params={"attachment": ""}, timeout=120,
                         headers={"User-Agent": "ooc-db-loader"})
        r.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"Could not download DILIrank from {DOWNLOAD_URL}: {e}\n{advice}")
    if not r.content.startswith(b"PK"):  # .xlsx files are zip archives
        sys.exit(f"The FDA returned something other than a spreadsheet from {DOWNLOAD_URL}. {advice}")
    dest.write_bytes(r.content)
    print(f"Downloaded {len(r.content):,} bytes to {dest}")
    return dest


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=pathlib.Path, help="DILIrank .xlsx or .csv")
    src.add_argument("--download", action="store_true",
                     help=f"download DILIrank 2.0 from the FDA ({DOWNLOAD_URL}) and keep a copy")
    p.add_argument("--sheet", help=f"worksheet to read (default: '{PREFERRED_SHEET}' if present)")
    p.add_argument("--dsn", help=f"database URL (default: ${db.DSN_ENV})")
    p.add_argument("--less-concern-as", choices=("positive", "ambiguous", "negative"), default="positive",
                   help="verdict for Less-DILI-concern drugs (default: positive)")
    p.add_argument("--only-existing", action="store_true",
                   help="only load drugs already in the drug table; do not add new drugs")
    p.add_argument("--dry-run", action="store_true", help="parse and report, write nothing")
    args = p.parse_args(argv)

    path = download() if args.download else args.file
    sheet = resolve_sheet(path, args.sheet)
    entries, warnings = merge_duplicates(parse(path, sheet), args.less_concern_as)
    for w in warnings:
        print("warning:", w, file=sys.stderr)
    counts = Counter(e.category for e in entries)
    where = f"{path}" + (f", sheet '{sheet}'" if sheet else "")
    print(f"Read {len(entries)} drugs from {where}: "
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
