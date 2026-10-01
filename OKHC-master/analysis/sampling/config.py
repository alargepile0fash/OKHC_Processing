"""Load settings for the time-window sampler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "analysis" / "config" / "sampling_default.json"


def load_config(path: Path) -> dict:
    """Read the JSON settings file."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    """Load JSON settings, then allow command-line options to override them."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    config_args, _ = config_parser.parse_known_args()

    config_path = config_args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    settings = load_config(config_path)

    parser = argparse.ArgumentParser(
        description="Prepare the historical Korean word-form sample for phonological analysis."
    )
    parser.add_argument("--config", type=Path, default=config_path)
    parser.add_argument("--input", default=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default=argparse.SUPPRESS)
    parser.add_argument("--year-col", default=argparse.SUPPRESS)
    parser.add_argument("--token-col", default=argparse.SUPPRESS)
    parser.add_argument("--vowels-col", default=argparse.SUPPRESS)
    parser.add_argument("--window-width", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--window-step", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--start-year", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--min-word-vowels", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--chunksize", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--min-window-wordforms", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--min-window-tokens", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max-wordforms-per-window", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--sampling-mode", choices=["cap-preserve-windows", "strict-balanced"], default=argparse.SUPPRESS)
    parser.add_argument("--target-wordforms-per-window", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--exclude-windows", nargs="*", default=argparse.SUPPRESS)
    parser.add_argument("--skip-token-row-output", action="store_true", default=argparse.SUPPRESS)

    parser.set_defaults(**settings)
    args = parser.parse_args()
    args.config = config_path
    return args


def validate_settings(args: argparse.Namespace) -> None:
    """Check settings that would otherwise produce an invalid sample."""
    if args.window_width <= 0 or args.window_step <= 0:
        raise ValueError("window_width and window_step must be positive.")
    if args.min_word_vowels < 1:
        raise ValueError("min_word_vowels must be at least 1.")
    if args.target_wordforms_per_window <= 0:
        raise ValueError("target_wordforms_per_window must be positive.")
    if args.min_window_wordforms < 0 or args.min_window_tokens < 0:
        raise ValueError("minimum sample thresholds cannot be negative.")
    if args.max_wordforms_per_window is not None and args.max_wordforms_per_window <= 0:
        raise ValueError("max_wordforms_per_window must be positive when provided.")

