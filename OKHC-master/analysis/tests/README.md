# Analysis pipeline test

sample.jsonl is a small raw OKHC fixture. Run the complete preprocessing,
vowel extraction, and time-window sampling pipeline with:

    python analysis/scripts/04_run_test_pipeline.py

The test runner:

1. preprocesses analysis/tests/sample.jsonl;
2. extracts the diachronic vowel-token CSV;
3. applies the test sampling configuration, including max_word_vowels: 5;
4. checks that the sampled output contains no word forms with more than five vowels.

All generated files are placed under analysis/tests/output/. The runner
deletes that directory at the beginning of each run, so each test starts
clean.

The test configuration uses min_window_wordforms: 0 because the fixture is
much smaller than the full corpus. This is a testing convenience and does not
change the production sampling configuration.
