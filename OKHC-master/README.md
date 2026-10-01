# Open Korean Historical Corpus

This repository contains preprocessing and analysis code for the Open Korean Historical Corpus.

## Diachronic vowel-harmony analysis

The historical vowel-analysis pipeline is divided into three stages:

```text
processed OKHC corpus
        |
        v
extract_diachronic_hangul_vowels_with_idu_integrated.py
        |
        v
hangul_vowel_tokens_diachronic.csv
        |
        v
01_prepare_timeperiod_samples.py
        |
        v
sampled time-window word forms
        |
        v
TP / D2L analysis
```

The extractor prepares the token-level data. The sampler controls the historical sampling design. The later phonological analysis should not change that sample.

### Time-window configuration

The sampler is controlled by a JSON file rather than by settings buried in the Python script.

Default configuration:

```text
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

```text
analysis/config/sampling_100yr_step25.json
```

To use it:

```bash
python analysis/processing_scripts/01_prepare_timeperiod_samples.py \
    --config analysis/config/sampling_100yr_step25.json
```

You can also override an individual setting from the command line. For example:

```bash
python analysis/processing_scripts/01_prepare_timeperiod_samples.py \
    --config analysis/config/sampling_default.json \
    --window-width 100 \
    --window-step 25
```

The JSON file is therefore the normal place to make experimental changes. The command line is useful for quick tests.

### What the sampler does

`01_prepare_timeperiod_samples.py` has a deliberately small set of responsibilities:

1. **`load_config()` / `parse_args()`** — read the JSON settings and optional command-line overrides.
2. **`choose_anchor_year()` / `windows_for_year()`** — define the historical windows.
3. **`extract_vowels()` / `prepare_chunk()`** — clean the token data and assign tokens to windows.
4. **`count_wordforms()`** — count word forms within each window.
5. **`make_window_diagnostics()`** — determine which windows are eligible and record sample-size information.
6. **`select_wordforms()`** — select the highest-frequency word forms.
7. **`write_outputs()` / `write_selected_token_rows()`** — save the analysis sample and optional audit rows.
8. **`write_run_config()`** — save the exact settings used for the run beside the output.

This means that if you want to change the historical design, you normally only need to edit the JSON file. If you want to understand how windows are constructed, `windows_for_year()` is the relevant function. If you want to understand how the frequency sample is chosen, `select_wordforms()` is the relevant function.

### Sampling modes

The default mode is:

```json
"sampling_mode": "cap-preserve-windows"
```

This keeps every eligible time window and selects up to:

```json
"target_wordforms_per_window": 1000
```

The highest-frequency word forms in that window. A sparse window can therefore contain fewer than 1,000 forms rather than disappearing from the historical sequence.

The alternative is:

```json
"sampling_mode": "strict-balanced"
```

This excludes windows below `min_window_wordforms` and gives every remaining window the same sample size, based on the smallest eligible window (optionally capped by `max_wordforms_per_window`).

### Output

The sampler writes to the directory specified by `output_dir` in the configuration file. The main files are:

```text
sampled_time_window_wordforms.csv
all_time_window_diagnostics_before_sampling.csv
excluded_time_windows_due_to_low_sample.csv
time_window_summary.csv
sampled_time_window_original_token_rows.csv
sampling_config.json
```

`sampled_time_window_wordforms.csv` is the main input for later phonological analysis. Each row belongs to a specific `time_window_id`.

`sampling_config.json` records the settings actually used, so an output can always be traced back to its sampling design.

When `window_step < window_width`, neighboring windows overlap in their corpus observations. Those windows are therefore useful as overlapping temporal views, but should not be treated as statistically independent samples.

### D2L boundary

The sampler does not construct UR/SR pairs and does not run D2L. The intended D2L implementation is Caleb Belth's pre-tested implementation in `algophon`. Once the historical UR/SR representation is defined, D2L should be run separately for each `time_window_id` using the already-selected sample.
