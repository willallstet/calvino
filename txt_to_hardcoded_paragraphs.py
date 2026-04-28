#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a .txt file into HARDCODED_INPUT_PARAGRAPHS Python literal."
    )
    parser.add_argument("input_file", help="Path to input .txt file")
    parser.add_argument(
        "-o",
        "--output-file",
        help="Optional path to write output. Defaults to stdout.",
    )
    parser.add_argument(
        "--single-group",
        action="store_true",
        help="Emit one outer list item containing all paragraphs: [[p1, p2, ...]].",
    )
    parser.add_argument(
        "--with-assignment",
        action="store_true",
        help="Prefix output with 'HARDCODED_INPUT_PARAGRAPHS = '.",
    )
    return parser.parse_args()


def split_paragraphs(text: str) -> List[str]:
    # Normalize newlines and trim leading/trailing whitespace.
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    # Paragraphs are separated by one or more blank lines.
    raw_paragraphs = re.split(r"\n\s*\n+", normalized)
    cleaned: List[str] = []
    for p in raw_paragraphs:
        lines = [line.strip() for line in p.split("\n") if line.strip()]
        paragraph = " ".join(lines).strip()
        if paragraph:
            cleaned.append(paragraph)
    return cleaned


def to_2d_list(paragraphs: List[str], single_group: bool) -> List[List[str]]:
    if single_group:
        return [paragraphs]
    return [[p] for p in paragraphs]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    text = input_path.read_text(encoding="utf-8")
    paragraphs = split_paragraphs(text)
    payload = to_2d_list(paragraphs, single_group=args.single_group)

    literal = repr(payload)
    if args.with_assignment:
        literal = f"HARDCODED_INPUT_PARAGRAPHS = {literal}"

    if args.output_file:
        Path(args.output_file).write_text(literal + "\n", encoding="utf-8")
    else:
        print(literal)


if __name__ == "__main__":
    main()
