# Analysis pipeline test

The test pipeline uses JSON configuration files rather than hard-coded paths
and settings in the test runner.

Run the complete preprocessing, vowel extraction, and time-window sampling
pipeline with:

    python analysis/scripts/04_run_test_pipeline.py

The default test-pipeline configuration is:

    analysis/config/test_pipeline.json

That file points to the stage-specific test configurations:

- `analysis/config/preprocessing_test.json`
- `analysis/config/extraction_test.json`
- `analysis/config/sampling_test.json`

The test runner:

1. preprocesses the configured raw fixture;
2. extracts the diachronic vowel-token CSV;
3. applies the configured sampling settings;
4. checks that the sampled output contains no word forms above the configured
   maximum vowel count.

All generated files are placed under the configured test output directory.
The default runner deletes that directory at the beginning of each run, so each
test starts clean.

The test configuration uses `min_window_wordforms: 0` because the fixture is
much smaller than the full corpus. This is a testing convenience and does not
change the production sampling configuration.

## Configuration structure

Quickly changeable settings are kept in JSON:

- `preprocessing_default.json`: production preprocessing paths and chunking.
- `extraction_default.json`: production extraction paths, filters, historical
  phase boundaries, and Idu extraction settings.
- `sampling_default.json`: production temporal sampling settings.
- `test_pipeline.json`: which test fixture and stage-specific test configs to
  run.
- `preprocessing_test.json`, `extraction_test.json`, and
  `sampling_test.json`: test-specific overrides.

The Python scripts contain the processing logic. They should not normally need
to be edited just to change an input/output path, threshold, historical date,
filter, or sampling parameter.
