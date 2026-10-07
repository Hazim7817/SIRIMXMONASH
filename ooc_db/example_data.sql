-- DEMO DATA ONLY. Run after schema.sql to see how the tables fit together.
--
-- The chip measurements below are invented. The drug facts are illustrative
-- and must be checked against the cited sources before any real use.
-- To start clean afterwards, re-run schema.sql.

SET search_path TO ooc;

INSERT INTO drug (name, pubchem_cid, human_cmax_um, cmax_source, notes) VALUES
    ('Acetaminophen', 1983, 139,  'Illustrative - verify', 'Liver injury in overdose'),
    ('Troglitazone',  5591, 6.4,  'Illustrative - verify', 'Withdrawn 2000 for liver failure'),
    ('Buspirone',     2477, 0.01, 'Illustrative - verify', 'Example of a drug without liver-injury concern');

INSERT INTO chip_model (name, organ, platform, cell_source) VALUES
    ('Demo liver chip', 'liver', 'Example platform', 'Primary human hepatocytes + endothelial cells');

INSERT INTO readout (name, unit, increase_means, description) VALUES
    ('ALT',       'U/L',    'harm',    'Enzyme released by damaged hepatocytes'),
    ('Albumin',   'ug/mL',  'benefit', 'Hepatocyte function; drops when cells are injured'),
    ('Viability', 'percent','benefit', 'Live cells, e.g. from nuclei counts');

INSERT INTO experiment (code, chip_model_id, batch_label, is_control, drug_id, concentration_um, vehicle, run_date) VALUES
    ('EXP-001', 1, 'B1', true,  NULL, NULL, '0.1% DMSO', '2026-09-01'),
    ('EXP-002', 1, 'B1', false, (SELECT drug_id FROM drug WHERE name = 'Acetaminophen'), 3475, '0.1% DMSO', '2026-09-01'),
    ('EXP-003', 1, 'B1', false, (SELECT drug_id FROM drug WHERE name = 'Troglitazone'),  160,  '0.1% DMSO', '2026-09-01'),
    ('EXP-004', 1, 'B1', false, (SELECT drug_id FROM drug WHERE name = 'Buspirone'),     0.25, '0.1% DMSO', '2026-09-01');

-- (experiment, readout, timepoint, replicate, value)
INSERT INTO measurement (experiment_id, readout_id, timepoint_h, replicate, value)
SELECT e.experiment_id, r.readout_id, v.t, v.rep, v.val
FROM (VALUES
    ('EXP-001', 'ALT', 72, 1, 20),   ('EXP-001', 'ALT', 72, 2, 22),
    ('EXP-001', 'Albumin', 72, 1, 30), ('EXP-001', 'Albumin', 72, 2, 32),
    ('EXP-001', 'Viability', 72, 1, 95),
    ('EXP-002', 'ALT', 72, 1, 84),   ('EXP-002', 'Albumin', 72, 1, 12), ('EXP-002', 'Viability', 72, 1, 45),
    ('EXP-003', 'ALT', 72, 1, 61),   ('EXP-003', 'Albumin', 72, 1, 17), ('EXP-003', 'Viability', 72, 1, 58),
    ('EXP-004', 'ALT', 72, 1, 23),   ('EXP-004', 'Albumin', 72, 1, 30), ('EXP-004', 'Viability', 72, 1, 93)
) AS v(code, readout, t, rep, val)
JOIN experiment e ON e.code = v.code
JOIN readout r ON r.name = v.readout;

-- The DILIrank rows use method 'database', as if imported, so running the
-- DILIrank loader later updates them instead of adding a second DILIrank row.
INSERT INTO reference_outcome (drug_id, endpoint, organ, species, evidence_type, source, finding, verdict, citation, method) VALUES
    ((SELECT drug_id FROM drug WHERE name = 'Acetaminophen'), 'toxicity', 'liver', 'human', 'post-market', 'LiverTox', 'Hepatotoxicity in overdose', 'positive', 'Verify in LiverTox', 'curated'),
    ((SELECT drug_id FROM drug WHERE name = 'Acetaminophen'), 'toxicity', 'liver', 'rat',   'GLP study',   'literature', 'Hepatic necrosis at high dose', 'positive', 'Verify', 'curated'),
    ((SELECT drug_id FROM drug WHERE name = 'Troglitazone'),  'toxicity', 'liver', 'human', 'post-market', 'DILIrank', 'vMost-DILI-concern; withdrawn', 'positive', 'Verify in DILIrank', 'database'),
    ((SELECT drug_id FROM drug WHERE name = 'Troglitazone'),  'toxicity', 'liver', 'rat',   'GLP study',   'literature', 'No clear liver injury in preclinical studies', 'negative', 'Verify', 'curated'),
    ((SELECT drug_id FROM drug WHERE name = 'Buspirone'),     'toxicity', 'liver', 'human', 'post-market', 'DILIrank', 'vNo-DILI-concern', 'negative', 'Verify in DILIrank', 'database');

INSERT INTO chip_call (drug_id, chip_model_id, endpoint, verdict, max_tested_conc_um, basis) VALUES
    ((SELECT drug_id FROM drug WHERE name = 'Acetaminophen'), 1, 'toxicity', 'positive', 3475, 'ALT >= 2x control at <= 25x Cmax'),
    ((SELECT drug_id FROM drug WHERE name = 'Troglitazone'),  1, 'toxicity', 'positive', 160,  'ALT >= 2x control at <= 25x Cmax'),
    ((SELECT drug_id FROM drug WHERE name = 'Buspirone'),     1, 'toxicity', 'negative', 0.25, 'ALT >= 2x control at <= 25x Cmax');
