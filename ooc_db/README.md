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
psql -d ooc -f schema.sql
psql -d ooc -f example_data.sql
```

All tables live in the `ooc` schema. Write `ooc.drug`, or run
`SET search_path TO ooc;` once per session.

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
  Treated chips are compared with controls that share their `batch_label`.
- **Verdicts** are `positive` (the effect happened: toxic, or effective),
  `negative`, or `ambiguous`. `endpoint` says whether you are asking about
  `toxicity` or `efficacy`.
- **New organs or assays** need new `chip_model` and `readout` rows, not
  new columns.

## Views (ready-made queries)

| View | Shows |
| --- | --- |
| `v_measurement_vs_control` | each treated value, its batch control mean, fold change, and dose as a multiple of Cmax |
| `v_reference_consensus` | one known answer per drug and species. If sources disagree it says `conflicting`, and that drug is excluded from scoring until you resolve it. |
| `v_concordance` | chip verdict next to the known verdict, labelled true/false positive/negative |
| `v_performance` | sensitivity, specificity and accuracy per chip model, endpoint and species |
| `v_faers_vs_dilirank` | FAERS signal next to the DILIrank verdict, per drug |

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
public sources. Install Python 3.10+, then from the `ooc_db` folder:

```
pip install -r requirements.txt
export OOC_DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/ooc
```

(On Windows PowerShell: `$env:OOC_DATABASE_URL = "postgresql://..."`.)

### DILIrank: curated human liver-injury classification

DILIrank (FDA Liver Toxicity Knowledge Base) ranks drugs by their risk of
liver injury in humans, from FDA drug labels and the literature.

```
python -m loaders.load_dilirank --download --dry-run   # fetch from fda.gov and check it parses
python -m loaders.load_dilirank --download
```

Or download it in a browser from the FDA page "Drug-Induced Liver Injury
Rank (DILIrank 2.0) Dataset" and pass `--file "path/to/file.xlsx"`. The
2.0 workbook has a `version 2` sheet (1,336 drugs, used by default) and a
`version 1` sheet (the original 1,036; `--sheet "version 1"`). Older
DILIrank 1.0 files and CSV exports also work.

Every DILIrank drug is added to `drug` (use `--only-existing` to load only
drugs you already have). Most-DILI-concern drugs become `positive`,
No-DILI-concern `negative` and Ambiguous `ambiguous`. Less-DILI-concern is
`positive` by default (DILI-positive = Most + Less); benchmark studies
often leave Less-DILI-concern out instead, which `--less-concern-as
ambiguous` does.

### FAERS: signals from adverse event reports

```
export OPENFDA_API_KEY=...          # free from open.fda.gov; needed for >500 drugs a day
python -m loaders.faers_signals --drug acetaminophen --drug troglitazone
python -m loaders.faers_signals --source DILIrank      # every DILIrank drug
```

For each drug the script asks openFDA how many reports mention the drug, a
set of liver-injury terms, both, and neither, and computes the PRR and ROR.
With the default rule (`--criterion ror`), a drug has a signal when the
lower 95% confidence bound of the ROR is above 1 and at least 3 reports
mention both. `--criterion evans` uses PRR >= 2, chi-squared >= 4 and at
least 3 reports instead.

**How drugs are found.** The drug's name, its `drug_alias` names, and
their base names without salt words ("Abacavir sulfate" is also searched as
"abacavir") are searched in three fields: openFDA's harmonised generic
name, the product name as reported, and the reported active ingredient.
No single field is complete. In particular, openFDA has no harmonised name
for many withdrawn drugs, which are common in DILIrank. Add brand names
and other spellings to `drug_alias` if a drug is missed.

**Which events count.** `loaders/event_terms/dili_narrow.txt` lists the
MedDRA preferred terms for liver injury itself (e.g. Drug-induced liver
injury, Hepatotoxicity, Hepatic failure, Hepatitis toxic). Lab results
(ALT increased), viral hepatitis and chronic liver disease are left out.
`dili_extended.txt` adds debated terms (jaundice, cholestasis, hepatic
encephalopathy, liver transplant, ...). Re-run with
`--terms loaders/event_terms/dili_extended.txt` to see whether your
conclusions depend on the term choice. The extended results are stored
separately (source `FAERS (extended terms)`). The script warns about any
term that matches no reports.

Results go to `faers_signal`. A summary row also goes to `reference_outcome`
with `use_for_scoring = false`, because reporting signals are weaker
evidence than DILIrank. Check how far they agree:

```sql
SELECT event_definition, agreement, count(*)
FROM ooc.v_faers_vs_dilirank GROUP BY 1, 2 ORDER BY 1, 2;
```

Expect partial agreement. In the closest published benchmark (Courtois et
al., *Front Pharmacol* 2018;9:1010, French pharmacovigilance database,
not FAERS), a disproportionality method found 75% of Most-DILI-concern
drugs and correctly cleared 79% of No-DILI-concern drugs.

Treat FAERS signals as supporting evidence only:

- Reports are voluntary and unverified, and publicity inflates them.
- A report counts for every drug it lists. The API cannot restrict
  counts to the drug the reporter suspected, so a drug often taken
  alongside liver-toxic drugs, or in combination products, picks up their
  reports.
- The patient's illness can cause the event (e.g. TB drugs and hepatitis).
- Reports are not deduplicated.

Counts are cached in `.faers_cache.json`, so an interrupted run resumes
where it stopped. The cache is discarded automatically when openFDA
publishes new data, and a release change during a run stops it, so every
count in one run comes from the same data.

### Tests

```
pytest tests                                         # no database needed
OOC_TEST_DSN=postgresql://postgres:pw@localhost/scratch pytest tests
```

The second form also runs the database tests. It drops and recreates the
`ooc` schema, so point it at a scratch database, never your real one.

## Typical workflow

1. Add drugs, with PubChem CID and human Cmax.
2. Enter reference outcomes for each drug: human, and animal if available.
3. Run chips. Enter each experiment and its measurements, including controls.
4. Check `v_measurement_vs_control` and decide each drug's `chip_call`.
   Use the same written rule for every drug (the `basis` column).
5. Read `v_performance`. A credible validation needs roughly 20 to 30 drugs
   with a mix of known positives and negatives. Ewart et al. 2022
   (*Communications Medicine*) used 27 for a liver chip.
