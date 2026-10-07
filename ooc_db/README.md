# Organ-on-chip validation database (PostgreSQL)

Stores organ-on-chip (OOC) experiment results alongside known clinical and
animal outcomes for the same drugs, and scores how often the chip agrees.

## Setup

1. Install PostgreSQL (pgAdmin is included in the installer).
2. In pgAdmin, create a database, e.g. `ooc`.
3. Open the Query Tool on that database, then open and run `schema.sql`.
4. Optional: run `example_data.sql` to load three demo drugs. The
   measurements are invented and the drug facts must be verified. Re-run
   `schema.sql` to wipe them.

Command-line equivalent:

```
psql -U postgres -d ooc -f schema.sql
psql -U postgres -d ooc -f example_data.sql
```

On Windows, `psql` is in `C:\Program Files\PostgreSQL\<version>\bin\`
unless you added that folder to your PATH.

All tables live in the `ooc` schema. Write `ooc.drug`, or run
`SET search_path TO ooc;` once per session.

### Upgrading

`schema.sql` deletes everything in the `ooc` schema. If your database was
built with an earlier version and holds data you want to keep, the loader
scripts will say so. Back up the data, rebuild, and restore it:

```
pg_dump -U postgres -d ooc --data-only --schema=ooc --exclude-table=ooc.faers_signal -f ooc_backup.sql
psql -U postgres -d ooc -f schema.sql
psql -U postgres -d ooc -f ooc_backup.sql
```

Then re-run the FAERS script, since its statistics are not kept. If the
restore stops on a rule the new schema adds (for example an organ written
`Liver` instead of `liver`), fix that row in `ooc_backup.sql` and run the
last two commands again.

## Tables

| Table | One row per | Filled from |
| --- | --- | --- |
| `drug` | compound, with PubChem CID and human Cmax | DrugBank, literature |
| `drug_alias` | other name for a drug (e.g. paracetamol), used to match external sources | you |
| `chip_model` | organ chip design | your lab |
| `readout` | measured quantity and its unit (ALT, albumin, ...) | your lab |
| `experiment` | chip run: one chip, one treatment, in a batch | your lab |
| `measurement` | readout value at a timepoint | your lab |
| `reference_outcome` | known finding from one source, for one species | DILIrank, LiverTox, ToxRefDB, FAERS, papers |
| `chip_call` | the chip's overall verdict for a drug | you, from the measurements |
| `faers_signal` | FAERS disproportionality statistics for a drug and event | `loaders/faers_signals.py` |

Notes:

- **Controls** are experiments with `is_control = true` and no drug.
  Treated chips are compared with the average of the control chips that
  share their `batch_label` (each control chip counts once, however many
  replicates it has). Give controls with a different vehicle their own
  batch label.
- **Verdicts** are `positive` (the effect happened: toxic, or effective),
  `negative`, or `ambiguous`. `endpoint` says whether you are asking about
  `toxicity` or `efficacy`.
- **Organ and species** are written in lower case (`liver`, `human`,
  `rat`); the database rejects `Liver` so that spellings cannot split the
  results.
- **Names** are unique regardless of capitals (`Aspirin` = `ASPIRIN`). A
  drug name cannot also be another drug's alias.
- **New organs or assays** need new `chip_model` and `readout` rows, not
  new columns.

## Views (ready-made queries)

| View | Shows |
| --- | --- |
| `v_measurement_vs_control` | each treated value, its batch control mean, fold change, and dose as a multiple of Cmax |
| `v_reference_consensus` | one known answer per drug and species. `ambiguous` sources are ignored when another gives a clear answer; if clear answers disagree it says `conflicting`. |
| `v_concordance` | chip verdict next to the known verdict: true/false positive/negative, `not scored` (reference ambiguous or conflicting, or chip ambiguous) or `no reference` |
| `v_performance` | sensitivity, specificity and accuracy per chip model, endpoint and species, plus how many drugs were not scored |
| `v_faers_vs_dilirank` | FAERS verdict next to the DILIrank verdict, per drug |

```sql
SELECT * FROM ooc.v_performance;
```

When reading the results, treat **human** outcomes as the ground truth.
A chip that disagrees with the animal data can still be right. In the demo
data, troglitazone shows up as a "false positive" against rat but a true
positive against human, the pattern seen when animal studies missed its
liver toxicity. Score against human outcomes, and use the animal comparison
to ask whether the chip predicts humans better than animals do.

## Loading reference data automatically

The `loaders/` folder has Python scripts that fill `reference_outcome` from
public sources. Install Python 3.10+, open a terminal in the `ooc_db`
folder, and install the packages:

```
python -m pip install -r requirements.txt
```

(On Windows, use `py` if `python` is not found.) Then tell the scripts
where your database is.

PowerShell (Windows):

```
$env:OOC_DATABASE_URL = 'postgresql://postgres:YOUR_PASSWORD@localhost:5432/ooc'
```

macOS / Linux:

```
export OOC_DATABASE_URL='postgresql://postgres:YOUR_PASSWORD@localhost:5432/ooc'
```

If the password contains `@ : / # ? %` or spaces, use this form instead,
with the password in single quotes (inside them, write `\'` for a quote
and `\\` for a backslash):
`host=localhost port=5432 dbname=ooc user=postgres password='YOUR PASSWORD'`.
Wrap the whole value in double quotes when you set it in PowerShell. You
can also pass either form with `--dsn` on each command.

### DILIrank: curated human liver-injury classification

DILIrank (FDA Liver Toxicity Knowledge Base) ranks drugs by their risk of
liver injury in humans, from FDA drug labels and the literature.

```
python -m loaders.load_dilirank --download --dry-run   # fetch from fda.gov and check it parses
python -m loaders.load_dilirank --download
```

Or download it in a browser from the FDA page "Drug-Induced Liver Injury
Rank (DILIrank 2.0) Dataset" and pass `--file "path\to\file.xlsx"`. The
2.0 workbook has a `version 2` sheet (1,336 drugs, used by default) and a
`version 1` sheet (the original 1,036; `--sheet "version 1"`). Older
DILIrank 1.0 files and Excel CSV/TXT exports (including "Unicode Text"
and Mac formats) also work.

Every DILIrank drug is added to `drug` (use `--only-existing` to load only
drugs you already have). Most-DILI-concern drugs become `positive`,
No-DILI-concern `negative` and Ambiguous `ambiguous`. Less-DILI-concern is
`positive` by default (DILI-positive = Most + Less); benchmark studies
often leave Less-DILI-concern out instead, which `--less-concern-as
ambiguous` does.

Each load replaces the previous DILIrank list. Rows are updated, not
duplicated, and rows the loader wrote earlier for drugs missing from the
file you load (for example after switching from `version 2` to
`version 1`) are removed. If several DILIrank names turn out to be the
same drug in your database, through `drug_alias`, they are merged: the
most serious category is kept if their verdicts agree, and the drug is
marked ambiguous if they do not. Rows you typed by hand with source
`DILIrank` are never deleted; the loader warns if they duplicate an
imported row or no longer match a DILIrank name.

### FAERS: signals from adverse event reports

```
python -m loaders.faers_signals --drug acetaminophen --drug troglitazone
python -m loaders.faers_signals --source DILIrank      # every DILIrank drug
```

Each drug needs 2 requests to openFDA, and openFDA allows 1,000 a day
without an API key. For more than about 500 drugs a day, get a free key
from open.fda.gov and add `--api-key YOUR_KEY`, or set `OPENFDA_API_KEY`
the same way as `OOC_DATABASE_URL` above.

For each drug the script asks openFDA how many reports mention the drug, a
set of liver-injury terms, both, and neither, and computes the PRR and ROR.
With the default rule (`--criterion ror`), a drug has a **signal** when the
lower 95% confidence bound of the ROR is above 1 and at least 3 reports
mention both. `--criterion evans` uses PRR >= 2, chi-squared >= 4 and at
least 3 reports instead.

**Verdicts.** A signal is recorded as `positive`. No signal is recorded as
`negative` only if the data could have shown one: at least 5 liver-injury
reports were expected for the drug at the background rate
(`--min-expected`), and no more than that were seen. Otherwise the
verdict is `ambiguous`, because a drug with few reports, or with a
non-significant excess, is not evidence of safety.

**How drugs are found.** The drug's name, its `drug_alias` names, and
their base names without salt words ("Abacavir sulfate" is also searched as
"abacavir") are searched in three fields: openFDA's harmonised generic
name, the product name as reported, and the reported active ingredient.
No single field is complete. In particular, openFDA has no harmonised name
for many withdrawn drugs, which are common in DILIrank. Add brand names
and other spellings to `drug_alias` if a drug is missed.

Names are matched as words inside longer names, so "estradiol" also counts
reports of ethinyl estradiol contraceptives, and "acetaminophen" counts
combination products. The search used is saved in
`faers_signal.drug_query`. For drugs whose name is part of another drug's
name, re-run them with `--exact`, which matches whole harmonised names only
("ESTRADIOL" but not "ETHINYL ESTRADIOL"). Add salt forms ("Estradiol
valerate") as aliases, and note that `--exact` cannot find drugs openFDA
has no harmonised name for. Salt words are never stripped down to a bare
element, so "Lithium citrate" is searched under that name only; add
"Lithium carbonate" as an alias to include it.

**Which events count.** `loaders/event_terms/dili_narrow.txt` lists the
MedDRA preferred terms for liver injury itself (e.g. Drug-induced liver
injury, Hepatotoxicity, Hepatic failure, Hepatitis toxic). Lab results
(ALT increased), viral hepatitis and chronic liver disease are left out.
`dili_extended.txt` adds debated terms (jaundice, cholestasis, hepatic
encephalopathy, liver transplant, ...). Re-run with
`--terms loaders/event_terms/dili_extended.txt` to see whether your
conclusions depend on the term choice. The extended results are stored
separately, as source `FAERS (extended terms)`. Term files with other
names are stored in `faers_signal` only. The script warns about any term
that matches no reports.

Results go to `faers_signal`, with the cut-off and openFDA data release
used. For the two supplied term files a summary row also goes to
`reference_outcome`, with `use_for_scoring = false`, because reporting
signals are weaker evidence than DILIrank. If a drug later returns no
reports, its statistics are removed and its summary row becomes
`ambiguous`, keeping your `use_for_scoring` choice.

`v_faers_vs_dilirank` compares signal / no signal with DILIrank. A drug
without a signal whose expected count was below `--min-expected` is listed
as `too few reports` rather than counted, since FAERS could not have
shown a signal for it. Check how far they agree:

```sql
SELECT event_definition, agreement, count(*)
FROM ooc.v_faers_vs_dilirank GROUP BY 1, 2 ORDER BY 1, 2;
```

Expect partial agreement. The closest published benchmark is Courtois et
al., *Front Pharmacol* 2018;9:1010, which used the French
pharmacovigilance database, not FAERS. It reported that a
disproportionality method flagged about 75% of Most-DILI-concern drugs and
correctly cleared about 79% of No-DILI-concern drugs. It left
Less-DILI-concern drugs out, so compare like with like:

```sql
SELECT event_definition, dilirank_class,
       round(100.0 * avg(faers_signal::int), 1) AS pct_with_signal, count(*) AS drugs
FROM ooc.v_faers_vs_dilirank
WHERE dilirank_class IN ('most', 'no') AND agreement <> 'too few reports'
GROUP BY 1, 2 ORDER BY 1, 2;
```

`pct_with_signal` for `most` corresponds to the 75% above, and 100 minus
it for `no` to the 79%.

Treat FAERS signals as supporting evidence only:

- Reports are voluntary and unverified, and publicity inflates them.
- A report counts for every drug it lists. The API cannot restrict
  counts to the drug the reporter suspected, so a drug often taken
  alongside liver-toxic drugs, or in combination products, picks up their
  reports.
- The patient's illness can cause the event (e.g. TB drugs and hepatitis).
- Reports are not deduplicated.

Counts are cached in `.faers_cache.json` in the folder you run from, so an
interrupted run resumes where it stopped. The cache is discarded
automatically when openFDA publishes new data, and a release change during
a run stops it, so every count in one run comes from the same data.
Deleting the cache is always safe; it only means counts are fetched again.

### Tests

```
python -m pytest tests                  # no database needed
```

To also run the database tests, point `OOC_TEST_DSN` at a **scratch**
database. The tests drop and recreate the `ooc` schema in it, so never
use your real database.

```
$env:OOC_TEST_DSN = 'postgresql://postgres:pw@localhost/scratch'; python -m pytest tests; Remove-Item Env:OOC_TEST_DSN
OOC_TEST_DSN='postgresql://postgres:pw@localhost/scratch' python -m pytest tests     # macOS / Linux
```

## Typical workflow

1. Add drugs, with PubChem CID and human Cmax.
2. Enter reference outcomes for each drug: human, and animal if available.
3. Run chips. Enter each experiment and its measurements, including controls.
4. Check `v_measurement_vs_control` and decide each drug's `chip_call`.
   Use the same written rule for every drug (the `basis` column).
5. Read `v_performance`. A credible validation needs roughly 20 to 30 drugs
   with a mix of known positives and negatives. Ewart et al. 2022
   (*Communications Medicine*) used 27 for a liver chip.
