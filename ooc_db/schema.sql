-- Organ-on-chip (OOC) validation database
--
-- Purpose: store OOC experiment results next to known clinical and animal
-- outcomes for the same drugs, then measure how often the chip agrees.
--
-- Run this once on an empty database (pgAdmin: Query Tool -> open file -> Run).
-- Running it again drops and recreates everything, so do not re-run it on a
-- database that holds real data.

DROP SCHEMA IF EXISTS ooc CASCADE;
CREATE SCHEMA ooc;
-- The loaders check this version; change it with any incompatible change.
COMMENT ON SCHEMA ooc IS 'ooc_db schema version 3';
SET search_path TO ooc;


-- ---------------------------------------------------------------------------
-- 1. Drugs: one row per compound. Everything else links back here.
-- ---------------------------------------------------------------------------
CREATE TABLE drug (
    drug_id        serial PRIMARY KEY,
    -- Single spaces only: no leading/trailing or doubled spaces, tabs, line
    -- breaks or non-breaking spaces, which would make 'Aspirin ' a
    -- different drug from 'Aspirin'.
    name           text NOT NULL CHECK (name = btrim(name) AND name !~ '[\t\r\n\u00a0]|  '
                                        AND name <> ''),
    pubchem_cid    integer UNIQUE,          -- use IDs, not names, to match external databases
    inchikey       text UNIQUE,
    human_cmax_um  numeric CHECK (human_cmax_um > 0),  -- peak blood concentration in patients, µM
    cmax_source    text,
    notes          text
);
-- 'Acetaminophen' and 'ACETAMINOPHEN' are the same drug.
CREATE UNIQUE INDEX drug_name_ci ON drug (lower(name));

-- Other names for the same drug, so external sources that spell it
-- differently (e.g. 'paracetamol', a salt form) still match.
CREATE TABLE drug_alias (
    alias_id  serial PRIMARY KEY,
    drug_id   integer NOT NULL REFERENCES drug ON DELETE CASCADE,
    alias     text NOT NULL CHECK (alias = btrim(alias) AND alias !~ '[\t\r\n\u00a0]|  '
                                   AND alias <> ''),
    source    text                          -- where the alias came from
);
CREATE UNIQUE INDEX drug_alias_ci ON drug_alias (lower(alias));

-- A name must not be one drug's name and another drug's alias, or lookups
-- by name would be ambiguous.
CREATE FUNCTION check_name_alias_clash() RETURNS trigger LANGUAGE plpgsql
SET search_path = ooc, pg_catalog AS $$
BEGIN
    IF TG_TABLE_NAME = 'drug_alias' THEN
        IF EXISTS (SELECT 1 FROM ooc.drug WHERE lower(name) = lower(NEW.alias) AND drug_id <> NEW.drug_id) THEN
            RAISE EXCEPTION 'alias "%" is already the name of another drug; merge the two drugs instead', NEW.alias;
        END IF;
    ELSIF EXISTS (SELECT 1 FROM ooc.drug_alias WHERE lower(alias) = lower(NEW.name) AND drug_id <> NEW.drug_id) THEN
        RAISE EXCEPTION 'name "%" is already an alias of another drug', NEW.name;
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER drug_alias_no_clash BEFORE INSERT OR UPDATE ON drug_alias
    FOR EACH ROW EXECUTE FUNCTION check_name_alias_clash();
CREATE TRIGGER drug_name_no_clash BEFORE INSERT OR UPDATE OF name ON drug
    FOR EACH ROW EXECUTE FUNCTION check_name_alias_clash();


-- ---------------------------------------------------------------------------
-- 2. Chip models: which organ chip, from which platform, with which cells.
-- ---------------------------------------------------------------------------
CREATE TABLE chip_model (
    chip_model_id  serial PRIMARY KEY,
    name           text NOT NULL,            -- e.g. 'Liver chip v1 (primary hepatocytes)'
    -- lower case ('liver', 'kidney', ...) so it matches reference_outcome.organ
    organ          text NOT NULL CHECK (organ = lower(btrim(organ)) AND organ <> ''),
    platform       text,                     -- vendor / in-house design
    cell_source    text,                     -- cell types, donor, passage
    notes          text
);
CREATE UNIQUE INDEX chip_model_name_ci ON chip_model (lower(name));


-- ---------------------------------------------------------------------------
-- 3. Readouts: the list of things you measure, so names and units stay
--    consistent (no 'ALT' vs 'alt' vs 'ALT release').
-- ---------------------------------------------------------------------------
CREATE TABLE readout (
    readout_id       serial PRIMARY KEY,
    name             text NOT NULL,          -- e.g. 'ALT', 'Albumin', 'Viability'
    unit             text NOT NULL,
    increase_means   text NOT NULL CHECK (increase_means IN ('harm', 'benefit', 'neutral')),
    description      text
);
CREATE UNIQUE INDEX readout_name_ci ON readout (lower(name));


-- ---------------------------------------------------------------------------
-- 4. Experiments: one row per chip run (one chip, one treatment).
--    Untreated/vehicle chips are experiments too, with is_control = true and
--    no drug. Controls and treated chips run together share a batch_label,
--    which is how results are normalised to their own controls. All control
--    chips in a batch are averaged together, so give controls with a
--    different vehicle their own batch label.
-- ---------------------------------------------------------------------------
CREATE TABLE experiment (
    experiment_id     serial PRIMARY KEY,
    code              text NOT NULL UNIQUE,  -- your lab ID, e.g. 'EXP-001'
    chip_model_id     integer NOT NULL REFERENCES chip_model,
    batch_label       text NOT NULL,
    is_control        boolean NOT NULL DEFAULT false,
    drug_id           integer REFERENCES drug,
    concentration_um  numeric CHECK (concentration_um >= 0),
    vehicle           text,                  -- e.g. '0.1% DMSO'
    run_date          date,
    operator          text,
    notes             text,
    -- Controls have no drug; treated chips need a drug and a concentration.
    CONSTRAINT control_or_treated CHECK (
        (is_control AND drug_id IS NULL)
        OR (NOT is_control AND drug_id IS NOT NULL
            AND concentration_um IS NOT NULL AND concentration_um > 0)
    )
);


-- ---------------------------------------------------------------------------
-- 5. Measurements: one row per readout value. Long format ("readout name +
--    value") means new organs or assays only add rows, never new columns.
-- ---------------------------------------------------------------------------
CREATE TABLE measurement (
    measurement_id   serial PRIMARY KEY,
    experiment_id    integer NOT NULL REFERENCES experiment ON DELETE CASCADE,
    readout_id       integer NOT NULL REFERENCES readout,
    timepoint_h      numeric NOT NULL CHECK (timepoint_h >= 0),
    replicate        integer NOT NULL DEFAULT 1,
    value            numeric NOT NULL,
    UNIQUE (experiment_id, readout_id, timepoint_h, replicate)
);


-- ---------------------------------------------------------------------------
-- 6. Reference outcomes: what is already known about each drug from
--    clinical or animal studies. One row per finding per source.
--    verdict: 'positive' = the effect occurred (drug was toxic, or drug was
--    effective), 'negative' = it did not, 'ambiguous' = unclear.
-- ---------------------------------------------------------------------------
CREATE TABLE reference_outcome (
    reference_id      serial PRIMARY KEY,
    drug_id           integer NOT NULL REFERENCES drug,
    endpoint          text NOT NULL CHECK (endpoint IN ('toxicity', 'efficacy')),
    -- lower case, so 'Liver' and 'liver' are not treated as different organs
    organ             text NOT NULL CHECK (organ = lower(btrim(organ)) AND organ <> ''),
    species           text NOT NULL CHECK (species = lower(btrim(species)) AND species <> ''),
                                             -- 'human', 'rat', 'dog', ...
    evidence_type     text,                  -- 'clinical trial', 'post-market', 'GLP study', ...
    source            text NOT NULL,         -- 'DILIrank', 'LiverTox', 'ToxRefDB', 'literature', ...
    finding           text,
    verdict           text NOT NULL CHECK (verdict IN ('positive', 'negative', 'ambiguous')),
    dose_description  text,
    citation          text,
    use_for_scoring   boolean NOT NULL DEFAULT true,
    -- How the row was obtained: 'curated' (a person read the source),
    -- 'database' (imported from a curated database such as DILIrank),
    -- 'statistical_signal' (computed, e.g. FAERS disproportionality) or
    -- 'text_mined' (extracted from documents by software, then reviewed).
    -- Rows you copy by hand from DILIrank should use 'database', so the
    -- DILIrank loader updates them instead of adding a second row.
    method            text NOT NULL DEFAULT 'curated'
                      CHECK (method IN ('curated', 'database', 'statistical_signal', 'text_mined')),
    source_record_id  text,                  -- ID in the source, e.g. DILIrank LTKBID
    retrieved_on      date
);
-- Imported rows are keyed so re-running a loader updates them instead of
-- adding duplicates. Hand-curated rows have no such limit.
CREATE UNIQUE INDEX reference_outcome_imported
    ON reference_outcome (drug_id, endpoint, organ, species, source)
    WHERE method IN ('database', 'statistical_signal');


-- ---------------------------------------------------------------------------
-- 7. Chip calls: the chip's overall answer for a drug ("toxic or not?"),
--    decided by you from the measurements using a rule you write down in
--    `basis`. Kept separate from raw data so the rule can change without
--    touching measurements.
-- ---------------------------------------------------------------------------
CREATE TABLE chip_call (
    chip_call_id        serial PRIMARY KEY,
    drug_id             integer NOT NULL REFERENCES drug,
    chip_model_id       integer NOT NULL REFERENCES chip_model,
    endpoint            text NOT NULL CHECK (endpoint IN ('toxicity', 'efficacy')),
    verdict             text NOT NULL CHECK (verdict IN ('positive', 'negative', 'ambiguous')),
    max_tested_conc_um  numeric,
    basis               text,                -- e.g. 'ALT >= 2x control at <= 25x Cmax'
    decided_on          date DEFAULT current_date,
    UNIQUE (drug_id, chip_model_id, endpoint)
);


-- ---------------------------------------------------------------------------
-- 8. FAERS signals: disproportionality statistics from the FDA Adverse Event
--    Reporting System, written by loaders/faers_signals.py. Each signal is
--    also summarised as a reference_outcome row (method 'statistical_signal',
--    not used for scoring unless you switch it on).
--    2x2 table: a = reports with the drug and the event, b = drug without
--    event, c = event without drug, d = neither.
-- ---------------------------------------------------------------------------
CREATE TABLE faers_signal (
    faers_signal_id   serial PRIMARY KEY,
    drug_id           integer NOT NULL REFERENCES drug ON DELETE CASCADE,
    event_definition  text NOT NULL,         -- term file name without .txt, e.g. 'dili_narrow'
    event_terms       text[] NOT NULL,       -- MedDRA preferred terms counted as the event
    drug_query        text NOT NULL,         -- the openFDA search used for the drug
    a                 bigint NOT NULL CHECK (a >= 0),
    b                 bigint NOT NULL CHECK (b >= 0),
    c                 bigint NOT NULL CHECK (c >= 0),
    d                 bigint NOT NULL CHECK (d >= 0),
    expected_a        numeric,               -- event reports expected at the background rate
    prr               numeric,
    prr_lower95       numeric,
    prr_upper95       numeric,
    ror               numeric,
    ror_lower95       numeric,
    ror_upper95       numeric,
    chi2_yates        numeric,
    is_signal         boolean NOT NULL,
    criteria          text NOT NULL,         -- rule used for is_signal
    -- positive = signal; negative = reported no more often than expected,
    -- with at least min_expected reports expected; ambiguous = neither
    verdict           text NOT NULL CHECK (verdict IN ('positive', 'negative', 'ambiguous')),
    min_expected      numeric NOT NULL,      -- cut-off used for the verdict
    faers_updated     text,                  -- openFDA data release (meta.last_updated)
    queried_on        date NOT NULL DEFAULT current_date,
    UNIQUE (drug_id, event_definition)
);


CREATE INDEX ON experiment (drug_id);
CREATE INDEX ON experiment (batch_label);
CREATE INDEX ON measurement (experiment_id);
CREATE INDEX ON reference_outcome (drug_id);


-- ===========================================================================
-- Views: ready-made queries. Use them like tables: SELECT * FROM ooc.<view>;
-- ===========================================================================

-- Every treated measurement next to the mean of its batch's controls (same
-- chip model, readout and timepoint), plus the dose as a multiple of human
-- Cmax. Replicates are averaged per control chip first, so every control
-- chip counts equally; n_control is the number of control chips.
CREATE VIEW v_measurement_vs_control AS
WITH control_chip AS (
    SELECT e.experiment_id, e.batch_label, e.chip_model_id, m.readout_id, m.timepoint_h,
           avg(m.value) AS chip_mean
    FROM measurement m
    JOIN experiment e USING (experiment_id)
    WHERE e.is_control
    GROUP BY 1, 2, 3, 4, 5
), control_mean AS (
    SELECT batch_label, chip_model_id, readout_id, timepoint_h,
           avg(chip_mean) AS control_mean, count(*) AS n_control
    FROM control_chip
    GROUP BY 1, 2, 3, 4
)
SELECT e.code AS experiment, cm.name AS chip_model, cm.organ, e.batch_label,
       d.name AS drug, e.concentration_um,
       round(e.concentration_um / d.human_cmax_um, 2) AS x_cmax,
       r.name AS readout, r.unit, r.increase_means, m.timepoint_h, m.replicate,
       m.value, round(c.control_mean, 4) AS control_mean, c.n_control,
       round(m.value / nullif(c.control_mean, 0), 3) AS fold_change
FROM measurement m
JOIN experiment e USING (experiment_id)
JOIN chip_model cm USING (chip_model_id)
JOIN drug d USING (drug_id)
JOIN readout r USING (readout_id)
LEFT JOIN control_mean c
       ON c.batch_label = e.batch_label AND c.chip_model_id = e.chip_model_id
      AND c.readout_id = m.readout_id AND c.timepoint_h = m.timepoint_h
WHERE NOT e.is_control;


-- One reference answer per drug / endpoint / organ / species, from the rows
-- with use_for_scoring = true. 'ambiguous' rows are ignored when another
-- source gives a clear answer. If the clear answers disagree, the consensus
-- is 'conflicting' and the drug is left out of scoring until you resolve it
-- (fix a row or set use_for_scoring = false).
CREATE VIEW v_reference_consensus AS
SELECT drug_id, endpoint, organ, species,
       CASE
           WHEN count(*) FILTER (WHERE verdict <> 'ambiguous') = 0 THEN 'ambiguous'
           WHEN count(DISTINCT verdict) FILTER (WHERE verdict <> 'ambiguous') = 1
               THEN min(verdict) FILTER (WHERE verdict <> 'ambiguous')
           ELSE 'conflicting'
       END AS verdict,
       CASE WHEN count(*) FILTER (WHERE verdict <> 'ambiguous') > 0
            THEN string_agg(DISTINCT source, ', ') FILTER (WHERE verdict <> 'ambiguous')
            ELSE string_agg(DISTINCT source, ', ')
       END AS sources,
       CASE WHEN count(*) FILTER (WHERE verdict <> 'ambiguous') > 0
            THEN count(*) FILTER (WHERE verdict <> 'ambiguous')
            ELSE count(*)
       END AS n_findings
FROM reference_outcome
WHERE use_for_scoring
GROUP BY drug_id, endpoint, organ, species;


-- The chip's answer next to the known answer, drug by drug, for every
-- species that has scored references for that organ (and always for
-- human). A drug missing a reference for a species appears with outcome
-- 'no reference' on that species' row.
CREATE VIEW v_concordance AS
WITH scored_species AS (
    SELECT DISTINCT endpoint, organ, species FROM reference_outcome WHERE use_for_scoring
    UNION
    SELECT e.endpoint, cm.organ, 'human'
    FROM (VALUES ('toxicity'), ('efficacy')) AS e(endpoint) CROSS JOIN chip_model cm
)
SELECT d.name AS drug, cm.name AS chip_model, cc.endpoint, sp.species,
       cc.verdict AS chip_says, rc.verdict AS reference_says, rc.sources,
       CASE
           WHEN rc.verdict IS NULL THEN 'no reference'
           WHEN rc.verdict IN ('conflicting', 'ambiguous') OR cc.verdict = 'ambiguous' THEN 'not scored'
           WHEN cc.verdict = 'positive' AND rc.verdict = 'positive' THEN 'true positive'
           WHEN cc.verdict = 'negative' AND rc.verdict = 'negative' THEN 'true negative'
           WHEN cc.verdict = 'positive' AND rc.verdict = 'negative' THEN 'false positive'
           WHEN cc.verdict = 'negative' AND rc.verdict = 'positive' THEN 'false negative'
       END AS outcome
FROM chip_call cc
JOIN drug d USING (drug_id)
JOIN chip_model cm USING (chip_model_id)
JOIN scored_species sp ON sp.endpoint = cc.endpoint AND sp.organ = cm.organ
LEFT JOIN v_reference_consensus rc
     ON rc.drug_id = cc.drug_id AND rc.endpoint = cc.endpoint AND rc.organ = cm.organ
    AND rc.species = sp.species;


-- The headline numbers: how well does each chip model predict each species?
-- Drugs counted as 'not scored' or 'no reference' are left out of the
-- percentages; check those columns to see how many.
CREATE VIEW v_performance AS
SELECT chip_model, endpoint, species,
       count(*) FILTER (WHERE outcome = 'true positive')  AS tp,
       count(*) FILTER (WHERE outcome = 'false negative') AS fn,
       count(*) FILTER (WHERE outcome = 'true negative')  AS tn,
       count(*) FILTER (WHERE outcome = 'false positive') AS fp,
       count(*) FILTER (WHERE outcome = 'not scored')     AS not_scored,
       count(*) FILTER (WHERE outcome = 'no reference')   AS no_reference,
       round(100.0 * count(*) FILTER (WHERE outcome = 'true positive')
             / nullif(count(*) FILTER (WHERE outcome IN ('true positive', 'false negative')), 0), 1)
           AS sensitivity_pct,
       round(100.0 * count(*) FILTER (WHERE outcome = 'true negative')
             / nullif(count(*) FILTER (WHERE outcome IN ('true negative', 'false positive')), 0), 1)
           AS specificity_pct,
       round(100.0 * count(*) FILTER (WHERE outcome IN ('true positive', 'true negative'))
             / nullif(count(*) FILTER (WHERE outcome IN ('true positive', 'true negative',
                                                         'false positive', 'false negative')), 0), 1)
           AS accuracy_pct
FROM v_concordance
GROUP BY chip_model, endpoint, species;


-- How well do FAERS signals agree with DILIrank? A sanity check on the
-- statistical signals before you rely on them for drugs DILIrank lacks.
-- Compares signal / no signal with DILIrank's verdict, as published
-- benchmarks do. Drugs without a signal that had fewer than min_expected
-- event reports expected are 'too few reports' (FAERS could not have shown
-- a signal); Ambiguous DILIrank drugs are 'not compared'. dilirank_class
-- lets you compare Most-DILI-concern vs No-DILI-concern only. Imported
-- DILIrank rows are used, or a hand-typed one if there is no imported row.
CREATE VIEW v_faers_vs_dilirank AS
WITH dilirank AS (
    SELECT DISTINCT ON (drug_id) drug_id, verdict, finding,
           CASE regexp_replace(lower(split_part(finding, ';', 1)), '[^a-z/]', '', 'g')
               WHEN 'vmostdiliconcern' THEN 'most'
               WHEN 'mostdiliconcern' THEN 'most'
               WHEN 'vlessdiliconcern' THEN 'less'
               WHEN 'lessdiliconcern' THEN 'less'
               WHEN 'vnodiliconcern' THEN 'no'
               WHEN 'nodiliconcern' THEN 'no'
               WHEN 'ambiguousdiliconcern' THEN 'ambiguous'
               WHEN 'vambiguousdiliconcern' THEN 'ambiguous'
               ELSE 'other'                  -- e.g. conflicting duplicates
           END AS dilirank_class
    FROM reference_outcome
    WHERE source = 'DILIrank' AND endpoint = 'toxicity' AND organ = 'liver' AND species = 'human'
    ORDER BY drug_id, (method = 'database') DESC, reference_id
)
SELECT d.name AS drug, fs.event_definition, fs.a AS n_reports_with_event,
       round(fs.expected_a, 1) AS expected_reports,
       round(fs.ror, 2) AS ror, round(fs.ror_lower95, 2) AS ror_lower95,
       round(fs.prr, 2) AS prr, fs.is_signal AS faers_signal, fs.verdict AS faers_verdict,
       dr.verdict AS dilirank_verdict, dr.dilirank_class, dr.finding AS dilirank_category,
       CASE
           WHEN dr.verdict IS NULL OR dr.verdict = 'ambiguous' THEN 'not compared'
           WHEN NOT fs.is_signal AND fs.expected_a < fs.min_expected THEN 'too few reports'
           WHEN fs.is_signal = (dr.verdict = 'positive') THEN 'agree'
           ELSE 'disagree'
       END AS agreement,
       fs.drug_query
FROM faers_signal fs
JOIN drug d USING (drug_id)
LEFT JOIN dilirank dr USING (drug_id);
