#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from difflib import SequenceMatcher
from typing import Iterable, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DATASETS_SERVER = "https://datasets-server.huggingface.co"
DEFAULT_DATASET = "HuggingFaceFW/fineweb"
DEFAULT_FINEWEB_CONFIG = "sample-10BT"
DEFAULT_FINEWEB_SPLIT = "train"


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def call_json(path: str, params: dict, timeout: int = 30) -> dict:
    token = os.environ.get("HF_TOKEN")
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    query = urlencode(params)
    url = f"{DATASETS_SERVER}{path}?{query}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def resolve_config_split(dataset: str, config: Optional[str], split: Optional[str]) -> Tuple[str, str]:
    if config and split:
        return config, split
    try:
        payload = call_json("/splits", {"dataset": dataset})
    except (HTTPError, URLError):
        # FineWeb fallback when API split discovery is temporarily unavailable.
        return config or DEFAULT_FINEWEB_CONFIG, split or DEFAULT_FINEWEB_SPLIT
    splits = payload.get("splits", [])
    if not splits:
        raise RuntimeError(f"No splits returned for dataset {dataset}")
    if config and not split:
        for row in splits:
            if row.get("config") == config:
                return config, row["split"]
        raise RuntimeError(f"Config {config} not found for {dataset}")
    if split and not config:
        for row in splits:
            if row.get("split") == split:
                return row["config"], split
        raise RuntimeError(f"Split {split} not found for {dataset}")
    # Favor "train" split when available.
    for row in splits:
        if row.get("split") == "train":
            return row["config"], row["split"]
    return splits[0]["config"], splits[0]["split"]


def _extract_text_candidates(value: object) -> Iterable[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            yield stripped
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _extract_text_candidates(item)
        return
    if isinstance(value, list):
        for item in value:
            yield from _extract_text_candidates(item)


def best_sentence_match(source_sentence: str, candidate_texts: Iterable[str]) -> Tuple[str, float]:
    source = source_sentence.strip()
    if not source:
        return "", 0.0
    best = source
    best_score = -1.0
    for text in candidate_texts:
        for sent in split_sentences(text):
            score = SequenceMatcher(None, source.lower(), sent.lower()).ratio()
            if score > best_score:
                best = sent
                best_score = score
    if best_score < 0:
        return source, 0.0
    return best, best_score


def load_local_corpus_sentences(path: str) -> list[str]:
    sentences: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            # Parse markdown-table rows from dataset previews and use first column.
            if stripped.startswith("|") and stripped.count("|") >= 2:
                cols = [part.strip() for part in stripped.split("|")]
                if len(cols) >= 3:
                    first_col = cols[1]
                    if first_col and not set(first_col) <= {"-"}:
                        sentences.extend(split_sentences(first_col))
                continue
            sentences.extend(split_sentences(stripped))
    return sentences


def replace_with_fineweb_matches(
    text: str,
    dataset: str,
    config: str,
    split: str,
    search_rows: int,
    min_score: float,
    fallback_rows_offset: int,
    fallback_rows_length: int,
    local_corpus_sentences: Optional[list[str]] = None,
    verbose: bool = False,
) -> str:
    replaced = []
    fallback_pool: Optional[list[str]] = None
    for sentence in split_sentences(text):
        row_texts: list[str] = []
        try:
            payload = call_json(
                "/search",
                {
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                    "query": sentence,
                    "offset": 0,
                    "length": search_rows,
                },
            )
            rows = payload.get("rows", [])
            for row in rows:
                row_data = row.get("row", {})
                row_texts.extend(_extract_text_candidates(row_data))
        except (HTTPError, URLError) as exc:
            # Some datasets (including FineWeb configs) do not support /search.
            if isinstance(exc, HTTPError) and exc.code != 501:
                raise
            if fallback_pool is None:
                try:
                    payload = call_json(
                        "/rows",
                        {
                            "dataset": dataset,
                            "config": config,
                            "split": split,
                            "offset": fallback_rows_offset,
                            "length": fallback_rows_length,
                        },
                    )
                    rows = payload.get("rows", [])
                    fallback_pool = []
                    for row in rows:
                        row_data = row.get("row", {})
                        fallback_pool.extend(_extract_text_candidates(row_data))
                except (HTTPError, URLError):
                    fallback_pool = local_corpus_sentences or []
            row_texts = fallback_pool
            if verbose:
                print(
                    "[info] /search unavailable; using /rows or local corpus fallback pool.",
                    file=sys.stderr,
                )

        best_text, score = best_sentence_match(sentence, row_texts)
        if score < min_score:
            best_text = sentence
        replaced.append(best_text)
        if verbose:
            print(f"[score={score:.3f}] {sentence} -> {best_text}", file=sys.stderr)
    return " ".join(replaced)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split input into sentences, find the closest sentence-level match in "
            "HuggingFaceFW/fineweb using Dataset Viewer search, and replace each sentence."
        )
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--input-text", help="Direct input text to process.")
    src_group.add_argument("--input-file", help="Path to a UTF-8 text file to process.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help=f"Dataset id (default: {DEFAULT_DATASET}).")
    parser.add_argument("--config", default=None, help="Dataset config/subset. Auto-detected if omitted.")
    parser.add_argument("--split", default=None, help="Dataset split. Auto-detected if omitted.")
    parser.add_argument(
        "--search-rows",
        type=int,
        default=20,
        help="How many rows to retrieve per sentence search (default: 20, max typically 100).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.35,
        help="Minimum similarity score to accept replacement; otherwise keeps original sentence.",
    )
    parser.add_argument(
        "--fallback-rows-offset",
        type=int,
        default=0,
        help="When /search is unavailable, starting row offset for /rows fallback.",
    )
    parser.add_argument(
        "--fallback-rows-length",
        type=int,
        default=100,
        help="When /search is unavailable, row count for /rows fallback (1-100).",
    )
    parser.add_argument(
        "--local-corpus-file",
        default=None,
        help="Optional local text/markdown file used if FineWeb API endpoints are unavailable.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-sentence match details to stderr.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.search_rows < 1 or args.search_rows > 100:
        raise SystemExit("--search-rows must be between 1 and 100")
    if args.fallback_rows_length < 1 or args.fallback_rows_length > 100:
        raise SystemExit("--fallback-rows-length must be between 1 and 100")
    if args.fallback_rows_offset < 0:
        raise SystemExit("--fallback-rows-offset must be >= 0")
    text = args.input_text
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as handle:
            text = handle.read()
    if not text or not text.strip():
        raise SystemExit("Input text is empty.")
    local_corpus_sentences = None
    if args.local_corpus_file:
        local_corpus_sentences = load_local_corpus_sentences(args.local_corpus_file)

    config, split = resolve_config_split(args.dataset, args.config, args.split)
    output = replace_with_fineweb_matches(
        text=text,
        dataset=args.dataset,
        config=config,
        split=split,
        search_rows=args.search_rows,
        min_score=args.min_score,
        fallback_rows_offset=args.fallback_rows_offset,
        fallback_rows_length=args.fallback_rows_length,
        local_corpus_sentences=local_corpus_sentences,
        verbose=args.verbose,
    )
    print(output)


if __name__ == "__main__":
    main()
