#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import List


def split_paragraphs(text: str) -> List[str]:
    chunks = re.split(r"\n\s*\n+", text.strip())
    return [c.strip() for c in chunks if c.strip()]


def split_sections(text: str) -> List[str]:
    # Two or more blank lines separate sections.
    chunks = re.split(r"\n(?:\s*\n){2,}", text.strip())
    return [c.strip() for c in chunks if c.strip()]


def build_2d_list(text: str, by_sections: bool) -> List[List[str]]:
    if by_sections:
        sections = split_sections(text)
        return [split_paragraphs(section) for section in sections]
    return [[p] for p in split_paragraphs(text)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a text file into a 2D list of paragraphs."
    )
    parser.add_argument("input_file", help="Path to input .txt file")
    parser.add_argument(
        "-o",
        "--output-file",
        help="Optional path to write the generated 2D list text",
    )
    parser.add_argument(
        "--by-sections",
        action="store_true",
        help=(
            "Group paragraphs into sections first (sections split by 2+ blank lines). "
            "Without this flag, output is [[paragraph1], [paragraph2], ...]."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON array instead of Python list literal",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    text = input_path.read_text(encoding="utf-8")
    paragraph_2d = build_2d_list(text, by_sections=args.by_sections)

    if args.json:
        rendered = json.dumps(paragraph_2d, ensure_ascii=False, indent=2)
    else:
        rendered = repr(paragraph_2d)

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(str(output_path))
    else:
        print(rendered)


if __name__ == "__main__":
    main()
