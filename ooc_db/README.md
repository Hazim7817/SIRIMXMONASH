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
| `chip_model` | organ chip design | your lab |
| `readout` | measured quantity and its unit (ALT, albumin, ...) | your lab |
| `experiment` | chip run: one chip, one treatment, in a batch | your lab |
| `measurement` | readout value at a timepoint | your lab |
| `reference_outcome` | known finding from one source, for one species | DILIrank, LiverTox, ToxRefDB, FAERS, papers |
| `chip_call` | the chip's overall verdict for a drug | you, from the measurements |

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

```sql
SELECT * FROM ooc.v_performance;
```

When reading the results, treat **human** outcomes as the ground truth.
A chip that disagrees with the animal data can still be right. In the demo
data, troglitazone shows up as a "false positive" against rat but a true
positive against human, the pattern seen when animal studies missed its
liver toxicity. Score against human outcomes, and use the animal comparison
to ask whether the chip predicts humans better than animals do.

## Typical workflow

1. Add drugs, with PubChem CID and human Cmax.
2. Enter reference outcomes for each drug: human, and animal if available.
3. Run chips. Enter each experiment and its measurements, including controls.
4. Check `v_measurement_vs_control` and decide each drug's `chip_call`.
   Use the same written rule for every drug (the `basis` column).
5. Read `v_performance`. A credible validation needs roughly 20 to 30 drugs
   with a mix of known positives and negatives. Ewart et al. 2022
   (*Communications Medicine*) used 27 for a liver chip.
