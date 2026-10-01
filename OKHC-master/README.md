# Open Korean Historical Corpus

This repository contains preprocessing and analysis code for the Open Korean Historical Corpus.

## Pipeline

The project has three processing stages before the phonological analysis:

```
raw OKHC JSONL
     |
     v
01_preprocess_corpus.py
     |
     v
processed/classified OKHC JSONL
     |
     v
02_extract_diachronic_vowels.py
     |
     v
hangul_vowel_tokens_diachronic.csv
     |
     v
03_prepare_time_window_samples.py
     |
     v
historical time-window samples
     |
     v
TP / D2L analysis
```

### Configuration

Quickly changeable pipeline settings are kept in JSON configuration files rather
than in the processing scripts:

```
analysis/config/
├── preprocessing_default.json
├── preprocessing_test.json
├── extraction_default.json
├── extraction_test.json
├── sampling_default.json
└── testing/
    ├── preprocessing_test.json
    ├── extraction_test.json
    ├── sampling_test.json
    └── test_pipeline.json
```

The production configurations control normal input locations, chunking,
extraction filters, historical event boundaries, Idu extraction options, and
sampling design. Sampling runs also have a named output directory, so separate
experimental configurations can preserve their results independently. The test
configurations point the same processing logic at the small test fixture.

For the production pipeline, run each stage with its corresponding
configuration:

```bash
python analysis/scripts/01_preprocess_corpus.py --config analysis/config/preprocessing_default.json
python analysis/scripts/02_extract_diachronic_vowels.py --config analysis/config/extraction_default.json
python analysis/scripts/03_prepare_time_window_samples.py --config analysis/config/sampling_default.json
```

The Python files contain the processing logic; the JSON files are the normal
place to change experimental or run-specific settings. Core linguistic
definitions such as the vowel inventory and RTR classification remain in
Python because they define how the extractor works rather than merely
configuring a run.

The extractor records two harmony classifications. The **core** inventory is
the classical seven-vowel Middle Korean system (light/+RTR `ㆍ ㅏ ㅗ`,
dark/-RTR `ㅡ ㅓ ㅜ`, neutral `ㅣ`). The **expanded** inventory is an
explicit analytical extension in which additional complex/derived vowel
symbols inherit the harmony value of their historical nucleus. A vowel outside
the selected inventory is `OTHER`; it is never silently treated as neutral.
A sequence containing `OTHER` is therefore labeled
`unclassifiable_due_to_other` rather than being called harmonic or
disharmonic.

### Test pipeline

The complete pipeline can be tested with the small fixture using:

```bash
python analysis/scripts/04_run_test_pipeline.py
```

The test runner reads `analysis/config/testing/test_pipeline.json`, which selects the
preprocessing, extraction, and sampling test configurations. Generated test
output stays under `analysis/tests/output/`; the sampling results are placed
in the named `sampling` run directory beneath it.

### 1. Corpus preprocessing

`01_preprocess_corpus.py` is the entry point for the first stage. It reads the raw OKHC JSONL files in chunks, then calls the existing preprocessing and classification modules:

```
preprocessing/text_preprocessing.py
preprocessing/classification_logic.py
```

Those modules perform text normalization, language/script metadata, and corpus-specific classification. The runner handles file discovery, chunking, and output. It is therefore an important part of the pipeline, not just a convenience script.

The Idu classifier in `idu/idu_classifier.py` is a separate corpus component. It is not called by the diachronic vowel extractor because the extractor needs Idu dictionary-to-Hangul correspondences, rather than only document-level Idu/Chinese classification.

### 2. Diachronic vowel extraction

`02_extract_diachronic_vowels.py` reads the processed JSONL files and creates the standardized token-level CSV used by the historical analysis.

It does four main things:

1. extracts Hangul tokens and their vowel sequences;
2. records historical/event metadata;
3. optionally identifies Idu dictionary matches and their Hangul reading candidates;
4. writes one analysis-ready row per usable token/reading.

When `exclude_idu_derived_wordforms` is enabled, dictionary-derived Hangul
reading candidates are omitted from the analytical CSV. Independently written
Hangul tokens are not removed merely because an Idu match occurs elsewhere in
the same document: the dictionary correspondence does not by itself establish
that an ordinary Hangul token is the realization of that Idu match. This
distinction prevents the extractor from making an unsupported occurrence-level
pronunciation inference.

It deliberately does **not** assign analytical time periods. The raw `year` remains the temporal information used by the sampling stage.

### 3. Time-window sampling

`03_prepare_time_window_samples.py` is the entry point for the sampling stage. The script itself is intentionally small. Its supporting code lives in:

```
analysis/scripts/sampling_logic/
├── config.py       # read and validate settings
├── windows.py      # construct historical windows
├── corpus.py       # prepare and count corpus word forms
├── sampling.py     # decide which word forms enter the sample
└── output.py       # write results and run metadata
```

The main script therefore reads almost like the research procedure:

```
load settings
     |
prepare corpus word forms
     |
construct time windows
     |
select frequency sample
     |
save results
```

The sampler is controlled by a JSON file rather than settings buried in the Python code.

Default configuration:

```
analysis/config/sampling_default.json
```

Each sampling configuration has two run-specific settings:

```json
"runs_dir": "analysis/data/runs",
"run_name": "25yr_step25"
```

The sampler combines these into the actual output directory:

```
analysis/data/runs/25yr_step25/
```

This means changing the sampling design does not require reusing the same output
directory. To create another analysis run, copy the configuration, change the
settings you want to test, and give it a different run name, such as
`50yr_step25`. The resulting outputs are written to
`analysis/data/runs/50yr_step25/`.

The run directory contains both the analysis outputs and the generated
`sampling_config.json`, which records the exact settings used for that run.


The two most important settings are:

```json
"window_width": 25,
"window_step": 25
```

They are independent:

| Width | Step | Design |
|---:|---:|---|
| 25 | 25 | non-overlapping 25-year windows |
| 50 | 50 | non-overlapping 50-year windows |
| 100 | 100 | non-overlapping 100-year windows |
| 100 | 25 | 100-year windows beginning every 25 years |
| 50 | 25 | 50-year windows beginning every 25 years |

For another sampling design, copy `analysis/config/sampling_default.json`, change the settings you want to test, and give the copy a different `run_name`. For example, a 100-year window with a 25-year step could use:

```json
"runs_dir": "analysis/data/runs",
"run_name": "100yr_step25",
"window_width": 100,
"window_step": 25
```

Then run it with the copied configuration file.


### Sampling modes

The default mode is:

```json
"sampling_mode": "cap-preserve-windows"
```

This keeps eligible windows and selects up to:

```json
"target_wordforms_per_window": 1000
```

of the highest-frequency word forms in each window. Sparse windows can therefore contain fewer than 1,000 forms rather than automatically disappearing.

The alternative is:

```json
"sampling_mode": "strict-balanced"
```

This excludes windows below `min_window_wordforms` and gives the remaining windows the same sample size, based on the smallest eligible window. `max_wordforms_per_window`, when set, is a hard upper bound in either sampling mode.

### Output

The sampler writes these main files to the run-specific output directory:

```
sampled_time_window_wordforms.csv
time_window_diagnostics.csv
excluded_time_windows.csv
sampled_time_window_original_token_rows.csv
sampling_config.json
```

`sampled_time_window_wordforms.csv` is the main input for later phonological analysis. `sampling_config.json` records the settings actually used for the run.

When `window_step < window_width`, neighboring windows overlap in their corpus observations. They should therefore be treated as overlapping temporal views rather than statistically independent samples.

### D2L boundary

The sampling pipeline does not construct UR/SR pairs and does not implement D2L. The intended D2L implementation is Caleb Belth's implementation in `algophon`. Once the historical UR/SR representation is defined, D2L should be run separately for each `time_window_id` using the selected sample.

Generated analysis outputs are not part of the source pipeline and should not be committed as project code or test data unless they are deliberately being used as fixtures.
