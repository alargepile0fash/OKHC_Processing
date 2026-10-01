# Open Korean Historical Corpus

This repository provides the code and resources for the **Open Korean Historical Corpus**, a large-scale, openly licensed dataset of historical Korean texts.

## About the Corpus

This dataset is a large-scale collection of texts spanning 1,300 years of Korean history.

**Key Features:**

- **Size:** 17.7 million documents and 5.1 billion tokens.
- **Temporal Span:** 7th century to 2025.
- **Sources:** Compiled from 19 distinct institutional archives and public domain collections.
- **Languages:** Covers 6 languages, including:
  - Classical Chinese
  - Middle Korean
  - Early Modern Korean
  - Modern Korean
  - North Korean
  - Japanese
- **Writing Systems:** Includes under-represented scripts like Korean-style Sinitic (Idu) and Hanja-Hangul mixed script.

## Getting the Dataset

The full corpus is available for download on the Hugging Face Hub.

- **Sample (1.3 MB)**: [`./sample.jsonl`](analysis/processing_test/sample.jsonl)
- **Dataset (28.6 GB)**: https://huggingface.co/datasets/seyoungsong/Open-Korean-Historical-Corpus

## Repository Contents

This repository contains the code used for preprocessing, language and script identification, and classification (including the Idu classifier) described in our paper. Please note that the initial web crawling scripts are not included here to avoid placing an undue burden on the source institutions' servers. Researchers interested in the crawling code may request it from the corresponding authors, subject to an agreement on responsible use.

## License

- **Code:** The code in this repository is released under the **MIT License**.
- **Data:** The Open Korean Historical Corpus is released under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License**.

## Citation

```
@article{song2025open,
  author  = {Seyoung Song and Nawon Kim and Songeun Chae and Kiwoong Park and Jiho Jin and Haneul Yoo and Kyunghyun Cho and Alice Oh},
  journal = {ArXiv preprint},
  title   = {Open Korean Historical Corpus: A Millennia-Scale Diachronic Collection of Public Domain Texts},
  url     = {https://arxiv.org/abs/2510.24541},
  volume  = {abs/2510.24541},
  year    = {2025}
}
```


## Diachronic vowel-harmony analysis

The historical vowel-analysis pipeline is deliberately separated into preprocessing, flexible time-window sampling, and phonological analysis.

### Current analysis flow

    processed OKHC corpus
            |
            v
    extract_diachronic_hangul_vowels_with_idu_integrated.py
            |
            |  raw year + raw normalized vowel identities + provenance
            v
    hangul_vowel_tokens_diachronic.csv
            |
            v
    01_prepare_timeperiod_samples.py
            |
            |  configurable time-window width + step
            |  frequency-based word-form sampling
            v
    analysis/data/timeperiod_samples/
            |
            v
    later phonological analysis (TP/D2L)

The extractor **does not assign analysis periods or time windows**. Earlier versions emitted a legacy 50-year `period_50yr` column; that column has been removed so there is one authoritative temporal-sampling step.

### Flexible time-window sampling

The sampler treats **window width** and **window step** as independent parameters:

- `--window-width 25 --window-step 25` → non-overlapping 25-year periods
- `--window-width 50 --window-step 50` → non-overlapping 50-year periods
- `--window-width 100 --window-step 100` → non-overlapping 100-year periods
- `--window-width 100 --window-step 25` → 100-year rolling windows whose starts are 25 years apart
- `--window-width 50 --window-step 25` → 50-year overlapping windows sampled every 25 years

This makes temporal sensitivity analyses a configuration change rather than a rewrite of the sampling logic.

Each output row carries a stable `time_window_id`, `time_window_start`, `time_window_end`, and `time_window_label`. The exact temporal and sampling configuration is also written to `sampling_config.json`. This gives later D2L code a stable interface:

```python
for window_id, sample in samples.groupby("time_window_id"):
    run_d2l(sample)
```

The D2L implementation therefore does not need to know whether a run used 25-year periods, 50-year periods, 100-year periods, or overlapping windows.

**Important:** when the step is smaller than the width, adjacent windows share corpus observations. Those D2L results should therefore be treated as overlapping temporal windows, not independent samples.

### Sampling unit and frequency control

The sampling unit is a **time-window-specific word form**: orthographic token plus its normalized vowel sequence. Each time window gets its own frequency-ranked sample of available corpus word forms.

The default `cap-preserve-windows` mode keeps every non-excluded time window after the optional start-year cutoff and caps the number of selected word forms at 1,000 per window. Sparse windows are retained and flagged rather than silently removed.

The alternative `strict-balanced` mode excludes windows below the configured minimums and forces all included windows to the same selected sample size. This remains available when an exactly balanced design is needed.

The extractor retains the actual vowel symbols. The `+RTR`, `-RTR`, and neutral classifications are analytical labels, not replacements for the underlying vowel identities.

Idu-derived observations are explicitly labeled with `token_source` and the time-window sampling output aggregates that provenance into `token_sources_present` and `contains_idu_derived_observation`. This makes it possible to run source-sensitivity analyses without making provenance part of the sampling unit.

### D2L input boundary

`algophon` is the intended implementation for D2L. The time-window sampling output is **not itself a D2L training file**. Belth's D2L implementation learns from UR/SR pairs of equal length, so a historically defensible mapping from observed historical forms to UR/SR representations must be established before a D2L runner is added. The pipeline therefore does not invent URs from surface vowel sequences merely to satisfy the library API.

Once that representation is defined, D2L should be run independently for each `time_window_id` using the already-selected sample, rather than changing the sample while testing different phonological conditions.

### Recommended sensitivity runs

The same sampler can be rerun with different temporal designs while keeping the rest of the sampling procedure fixed. For example:

```bash
# 25-year non-overlapping baseline
python analysis/processing_scripts/01_prepare_timeperiod_samples.py \
    --window-width 25 --window-step 25

# 50-year non-overlapping windows
python analysis/processing_scripts/01_prepare_timeperiod_samples.py \
    --window-width 50 --window-step 50

# 100-year windows, sampled every 25 years
python analysis/processing_scripts/01_prepare_timeperiod_samples.py \
    --window-width 100 --window-step 25
```

For reproducibility, keep the resulting `sampling_config.json` with each output directory. If multiple temporal designs are being compared, give each run its own `--output-dir` rather than overwriting a previous sample.
