"""
Sentiment Analysis Tool — CLI entry point.

Usage
-----
    python main.py --text "I loved every minute of it!"
    python main.py --file data/sample_reviews.csv --output results.csv
    python main.py --text "Meh." --methods wordnet senticnet --verbose
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sentiment",
        description="Compare WordNet, SenticNet, and RoBERTa sentiment methods.",
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", "-t", type=str, help="Single text string to analyse.")
    group.add_argument("--file", "-f", type=Path, help="CSV file with a 'text' column.")

    p.add_argument(
        "--methods",
        "-m",
        nargs="+",
        choices=["wordnet", "senticnet", "roberta"],
        default=["wordnet", "senticnet", "roberta"],
        help="Which methods to run. Defaults to all three.",
    )
    p.add_argument("--output", "-o", type=Path, help="Save CSV results to this path.")
    p.add_argument("--verbose", "-v", action="store_true", help="Show scores alongside labels.")
    return p


def _load_analyzers(methods: list[str]) -> dict:
    analyzers = {}
    if "wordnet" in methods:
        from wordnet.analyzer import WordNetAnalyzer
        analyzers["wordnet"] = WordNetAnalyzer()
    if "senticnet" in methods:
        from senticnett.analyzer import SenticNetAnalyzer
        analyzers["senticnet"] = SenticNetAnalyzer()
    if "roberta" in methods:
        from rober.analyzer import RoBERTaAnalyzer
        analyzers["roberta"] = RoBERTaAnalyzer()
    return analyzers


def _analyse_single(text: str, analyzers: dict, verbose: bool) -> None:
    print(f"\nInput : {text!r}\n")
    col_w = max(len(k) for k in analyzers) + 2

    for name, analyzer in analyzers.items():
        t0 = time.perf_counter()
        result = analyzer.predict(text)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if verbose:
            detail = str(result)
        else:
            detail = result.label

        print(f"  {name:<{col_w}} {detail}   ({elapsed_ms:.1f}ms)")
    print()


def _analyse_file(
    path: Path,
    analyzers: dict,
    output: Path | None,
    verbose: bool,
) -> None:
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_csv(path)
    if "text" not in df.columns:
        sys.exit(f"ERROR: CSV must have a 'text' column. Found: {list(df.columns)}")

    texts = df["text"].astype(str).tolist()
    print(f"Loaded {len(texts)} rows from {path}")

    for name, analyzer in analyzers.items():
        print(f"Running {name}…")
        t0 = time.perf_counter()
        results = analyzer.predict_batch(texts)
        elapsed = time.perf_counter() - t0

        df[f"{name}_label"] = [r.label for r in results]
        if verbose:
            df[f"{name}_score"] = [
                getattr(r, "score", getattr(r, "polarity", getattr(r, "confidence", None)))
                for r in results
            ]
        print(f"  → done in {elapsed:.2f}s ({elapsed / len(texts) * 1000:.1f}ms/sample)")

    if output:
        df.to_csv(output, index=False)
        print(f"\nResults saved to {output}")
    else:
        print("\nFirst 10 rows:")
        print(df.head(10).to_string(index=False))


def cli() -> None:
    args = _build_parser().parse_args()
    analyzers = _load_analyzers(args.methods)

    if args.text:
        _analyse_single(args.text, analyzers, args.verbose)
    else:
        _analyse_file(args.file, analyzers, args.output, args.verbose)


if __name__ == "__main__":
    cli()
