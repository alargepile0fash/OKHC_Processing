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
3. adds Idu dictionary-derived Hangul readings when enabled;
4. writes one analysis-ready row per usable token/reading.

It deliberately does **not** assign analytical time periods. The raw `year` remains the temporal information used by the sampling stage.

### 3. Time-window sampling

`03_prepare_time_window_samples.py` is the entry point for the sampling stage. The script itself is intentionally small. Its supporting code lives in:

```
analysis/sampling/
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

For example, the repository includes:

```
analysis/config/sampling_100yr_step25.json
```

Run it with:

```bash
python analysis/processing_scripts/03_prepare_time_window_samples.py --config analysis/config/sampling_100yr_step25.json
```

You can also override an individual setting for a quick test:

```bash
python analysis/processing_scripts/03_prepare_time_window_samples.py --config analysis/config/sampling_default.json --window-width 100 --window-step 25
```

The JSON file is the normal place to make experimental changes.

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

This excludes windows below `min_window_wordforms` and gives the remaining windows the same sample size, based on the smallest eligible window (optionally capped by `max_wordforms_per_window`).

### Output

The sampler writes these main files to the configured output directory:

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
