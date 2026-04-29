#!/usr/bin/env python3
import argparse
import ast
import html
import re
from pathlib import Path
from typing import Dict, List, Set


SENTENCE_HEADER_RE = re.compile(r"^\[Sentence\s+\d+\]")
INLINE_EM_RE = re.compile(r"\*(.+?)\*")
PREFACE_NOTE_RE = re.compile(r"(\S+)\s*\[([^\[\]]+)\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Consistency as minimal centered HTML with per-line source fonts."
    )
    parser.add_argument("input_file", help="Path to calvino_fineweb_output_*.txt")
    parser.add_argument(
        "-o",
        "--output-file",
        help="Output HTML file path (default: input path with .html extension).",
    )
    parser.add_argument(
        "--preface-file",
        default="preface.txt",
        help=(
            "Optional preface .txt rendered full-width above both columns "
            "(default: preface.txt)."
        ),
    )
    return parser.parse_args()


def parse_font_payload(payload: str) -> Dict[str, str]:
    fields = ["family", "size", "weight", "style", "node"]
    markers = [f"{k}=" for k in fields]
    out: Dict[str, str] = {}

    for i, field in enumerate(fields):
        marker = markers[i]
        start = payload.find(marker)
        if start == -1:
            continue
        start += len(marker)
        end = len(payload)
        for next_marker in markers[i + 1 :]:
            j = payload.find(next_marker, start)
            if j != -1:
                end = j
                break
        out[field] = payload[start:end].strip()
    return out


def parse_blocks(raw_text: str) -> List[Dict[str, str]]:
    lines = raw_text.splitlines()
    blocks: List[Dict[str, str]] = []
    i = 0
    while i < len(lines):
        if not SENTENCE_HEADER_RE.match(lines[i].strip()):
            i += 1
            continue

        i += 1
        match_text = ""
        original_text = ""
        calvino_text = ""
        source_url = ""
        font: Dict[str, str] = {}
        current_field = ""
        while i < len(lines):
            s = lines[i].rstrip()
            if not s:
                i += 1
                break
            if SENTENCE_HEADER_RE.match(s) or s == "Consistency":
                break
            if s.startswith("ORIGINAL:"):
                original_text = s.split("ORIGINAL:", 1)[1].strip()
                current_field = "original"
            elif s.startswith("CALVINO:"):
                calvino_text = s.split("CALVINO:", 1)[1].strip()
                current_field = "calvino"
            elif s.startswith("MATCH:"):
                match_text = s.split("MATCH:", 1)[1].strip()
                current_field = "match"
            elif s.startswith("SOURCE_URL:"):
                source_url = s.split("SOURCE_URL:", 1)[1].strip()
                current_field = ""
            elif s.startswith("FONT:"):
                payload = s.split("FONT:", 1)[1].strip()
                font = parse_font_payload(payload)
                current_field = ""
            elif current_field == "original":
                original_text = f"{original_text} {s.strip()}".strip()
            elif current_field == "calvino":
                calvino_text = f"{calvino_text} {s.strip()}".strip()
            elif current_field == "match":
                match_text = f"{match_text} {s.strip()}".strip()
            i += 1

        blocks.append(
            {
                "original": original_text,
                "calvino": calvino_text,
                "match": match_text,
                "source_url": source_url,
                "family": font.get("family", ""),
                "size": font.get("size", ""),
                "weight": font.get("weight", ""),
                "style": font.get("style", ""),
            }
        )
    return blocks


def parse_final_section(raw_text: str) -> str:
    lines = raw_text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == "Consistency":
            start = idx + 1
            break
    if start is None:
        return ""

    collected: List[str] = []
    for line in lines[start:]:
        s = line.strip()
        if not s:
            continue
        if set(s) <= {"-"}:
            continue
        collected.append(s)
    return " ".join(collected).strip()


def style_for_block(block: Dict[str, str]) -> str:
    parts: List[str] = []
    if block["family"]:
        parts.append(f"font-family: {block['family']};")
    if block["size"]:
        parts.append(f"font-size: {block['size']};")
    if block["weight"]:
        parts.append(f"font-weight: {block['weight']};")
    if block["style"]:
        parts.append(f"font-style: {block['style']};")
    return " ".join(parts)


def escape_with_inline_emphasis(text: str) -> str:
    parts: List[str] = []
    last = 0
    for match in INLINE_EM_RE.finditer(text):
        start, end = match.span()
        inner = match.group(1)
        if not inner.strip():
            continue
        parts.append(html.escape(text[last:start]))
        parts.append(f"<em>{html.escape(inner)}</em>")
        last = end
    parts.append(html.escape(text[last:]))
    return "".join(parts)


def split_sentences_like_source(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def render_preface_section(preface_text: str) -> str:
    cleaned = preface_text.strip()
    if not cleaned:
        return ""
    paragraphs = re.split(r"\n\s*\n", cleaned)
    rendered = []
    for paragraph in paragraphs:
        paragraph_text = " ".join(paragraph.splitlines()).strip()
        if not paragraph_text:
            continue
        rendered.append(f"<p>{render_preface_paragraph(paragraph_text)}</p>")
    if not rendered:
        return ""
    return (
        '<section class="preface" aria-label="Preface">'
        + "".join(rendered)
        + "</section>"
    )


def render_preface_paragraph(paragraph_text: str) -> str:
    def replace_note(match: re.Match[str]) -> str:
        word = match.group(1)
        note = match.group(2).strip()
        if not note:
            return escape_with_inline_emphasis(word)
        return (
            '<span class="preface-note-anchor">'
            f'{escape_with_inline_emphasis(word)}'
            f'<span class="preface-note-popup">{escape_with_inline_emphasis(note)}</span>'
            "</span>"
        )

    rendered = PREFACE_NOTE_RE.sub(replace_note, paragraph_text)
    return escape_with_inline_emphasis(rendered) if rendered == paragraph_text else rendered


def paragraph_start_indices(block_count: int, source_file: Path) -> Set[int]:
    starts: Set[int] = {0}
    if not source_file.exists():
        return starts
    try:
        module = ast.parse(source_file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return starts

    payload = None
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "HARDCODED_INPUT_PARAGRAPHS":
                try:
                    payload = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    payload = None
                break
        if payload is not None:
            break
    if not isinstance(payload, list):
        return starts

    sentence_index = 0
    for paragraph_group in payload:
        if not isinstance(paragraph_group, list):
            continue
        for paragraph in paragraph_group:
            if not isinstance(paragraph, str):
                continue
            if sentence_index < block_count:
                starts.add(sentence_index)
            sentence_count = len(split_sentences_like_source(paragraph))
            if sentence_count <= 0:
                sentence_count = 1
            sentence_index += sentence_count

    return {idx for idx in starts if 0 <= idx < block_count}


def render_segment(
    raw_text: str,
    href: str = "",
    style: str = "",
    segment_index: int | None = None,
    interactive: bool = False,
    paragraph_start: bool = False,
) -> str:
    text = raw_text.strip()
    if not text:
        return ""

    is_quote = text.startswith("> ")
    if is_quote:
        text = text[2:].lstrip()

    escaped_text = escape_with_inline_emphasis(text)
    escaped_href = html.escape(href.strip())
    escaped_style = html.escape(style.strip())

    index_attr = f' data-segment-index="{segment_index}"' if segment_index is not None else ""
    hover_class = " hover-target" if interactive else ""
    paragraph_class = " paragraph-start" if paragraph_start else ""

    if escaped_href and escaped_style:
        inner = (
            f'<a class="segment" href="{escaped_href}" target="_blank" '
            f'rel="noopener noreferrer" style="{escaped_style}">{escaped_text}</a>'
        )
    elif escaped_href:
        inner = f'<a class="segment" href="{escaped_href}" target="_blank" rel="noopener noreferrer">{escaped_text}</a>'
    elif escaped_style:
        inner = f'<span class="segment" style="{escaped_style}">{escaped_text}</span>'
    else:
        inner = f'<span class="segment">{escaped_text}</span>'

    if is_quote:
        return f'<blockquote class="segment-block segment-quote{hover_class}{paragraph_class}"{index_attr}>{inner}</blockquote>'
    return f'<span class="segment-block segment-inline{hover_class}{paragraph_class}"{index_attr}>{inner}</span>'


def render_html(blocks: List[Dict[str, str]], final_text: str, preface_text: str = "") -> str:
    original_segments: List[str] = []
    final_segments: List[str] = []
    starts = paragraph_start_indices(
        block_count=len(blocks),
        source_file=Path(__file__).with_name("calvino_fineweb_vector_replace.py"),
    )
    for idx, block in enumerate(blocks):
        paragraph_start = idx in starts
        original_segment = render_segment(
            block["original"],
            segment_index=idx,
            paragraph_start=paragraph_start,
        )
        final_segment = render_segment(
            block["match"],
            href=block.get("source_url", ""),
            style=style_for_block(block),
            segment_index=idx,
            interactive=True,
            paragraph_start=paragraph_start,
        )
        if original_segment:
            original_segments.append(original_segment)
        if final_segment:
            final_segments.append(final_segment)

    joined_original = " ".join(original_segments)
    # Preserve sentence-level links and font metadata in the final output.
    if final_segments:
        joined_final = " ".join(final_segments)
    else:
        joined_final = f'<span class="segment">{html.escape(final_text.strip())}</span>'
    preface_section = render_preface_section(preface_text)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Consistency</title>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      height: 100%;
      color: #ffffff;
    }}
    body {{
      min-height: 100vh;
      font-family: serif;
      background-image: url("IMG_1604.jpeg");
      background-repeat: no-repeat;
      background-position: center center;
      background-size: 100% 100%;
      background-attachment: fixed;
    }}
    .wrap {{
      box-sizing: border-box;
      width: min(1600px, 96vw);
      margin: 0 auto;
      padding: 40px 48px 0;
      position: relative;
    }}
    .preface {{
      width: 100%;
      margin: 0 0 24px;
      line-height: 1.5;
      color: #ffffff;
      font-family: Arial, Helvetica, sans-serif;
    }}
    .preface p {{
      margin: 0 0 1em;
    }}
    .preface p:last-child {{
      margin-bottom: 0;
    }}
    .preface-note-anchor {{
      position: relative;
      display: inline-block;
      cursor: help;
      border-bottom: 1px dotted rgba(255, 255, 255, 0.45);
    }}
    .preface-note-popup {{
      position: absolute;
      left: 50%;
      top: calc(100% + 8px);
      transform: translateX(-50%);
      z-index: 10;
      display: none;
      min-width: 320px;
      max-width: min(760px, 86vw);
      padding: 8px 10px;
      border: 1px solid #000000;
      border-radius: 0;
      background: #ffffff;
      color: #000000;
      line-height: 1.4;
      font-size: 0.95em;
      box-shadow: none;
      white-space: normal;
    }}
    .preface-note-anchor:hover .preface-note-popup {{
      display: block;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 20px;
      border-top: 1px dashed #ffffff;
      padding-top: 20px;
      position: sticky;
      top: 0;
      height: 100vh;
      align-items: stretch;
    }}
    .panel {{
      padding: 0 12px 0;
      min-height: 0;
      box-sizing: border-box;
      overflow: hidden;
    }}
    .panel.original {{
      font-family: "Times New Roman", Times, serif;
    }}
    .chunk {{
      color: inherit;
      height: 100%;
      overflow: hidden;
    }}
    .segment {{
      line-height: 1.4;
      color: inherit;
      text-decoration: none;
    }}
    .segment-inline {{
      display: inline;
    }}
    .segment-inline.paragraph-start {{
      display: inline;
    }}
    .segment-inline.paragraph-start::before {{
      content: "\\A\\A";
      white-space: pre;
    }}
    .segment-inline.paragraph-start:first-child::before {{
      content: "";
    }}
    .segment-quote {{
      margin: 0.6em 0;
      padding-left: 0.9em;
      border-left: 2px solid #666666;
    }}
    .panel.original .segment-block {{
      transition: background-color 120ms ease-in-out, color 120ms ease-in-out;
    }}
    .panel.original .segment-block.is-hovered {{
      background: #ffffff;
      color: #000000;
    }}
    .panel.original .segment-block.is-hovered .segment {{
      color: #000000 !important;
    }}
    a.segment:hover {{
      text-decoration: underline;
    }}
    @media (max-width: 1024px) {{
      .grid {{
        grid-template-columns: 1fr;
        position: static;
        height: auto;
      }}
      .panel {{
        min-height: auto;
        overflow: visible;
      }}
      .chunk {{
        height: auto;
        overflow: visible;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    {preface_section}
    <div class="grid">
      <section class="panel original">
        <div class="chunk sync-scroll">{joined_original}</div>
      </section>
      <section class="panel final-panel">
        <div class="chunk sync-scroll">{joined_final}</div>
      </section>
    </div>
    <div id="scroll-track" aria-hidden="true"></div>
  </div>
  <script>
    (function () {{
      const columns = Array.from(document.querySelectorAll(".chunk.sync-scroll"));
      const track = document.getElementById("scroll-track");
      const preface = document.querySelector(".preface");
      const mobileQuery = window.matchMedia("(max-width: 1024px)");
      let syncStartOffset = 0;
      let syncScrollRange = 0;

      function syncColumns() {{
        if (mobileQuery.matches) return;
        const activeScroll = Math.max(0, window.scrollY - syncStartOffset);
        const progress = syncScrollRange > 0
          ? Math.min(1, activeScroll / syncScrollRange)
          : 0;
        for (const col of columns) {{
          const colRange = Math.max(0, col.scrollHeight - col.clientHeight);
          col.scrollTop = colRange * progress;
        }}
      }}

      function updateTrack() {{
        if (mobileQuery.matches) {{
          track.style.height = "0px";
          for (const col of columns) col.scrollTop = 0;
          return;
        }}
        const maxRange = Math.max(
          0,
          ...columns.map((col) => Math.max(0, col.scrollHeight - col.clientHeight))
        );
        syncStartOffset = preface ? preface.offsetHeight : 0;
        syncScrollRange = maxRange;
        track.style.height = `${{Math.max(1, syncStartOffset + syncScrollRange + window.innerHeight)}}px`;
        syncColumns();
      }}

      window.addEventListener("scroll", syncColumns, {{ passive: true }});
      window.addEventListener("resize", updateTrack);
      if (mobileQuery.addEventListener) {{
        mobileQuery.addEventListener("change", updateTrack);
      }}
      window.addEventListener("load", updateTrack);
      requestAnimationFrame(updateTrack);

      const hoverTargets = Array.from(document.querySelectorAll(".hover-target[data-segment-index]"));
      for (const target of hoverTargets) {{
        const idx = target.getAttribute("data-segment-index");
        const source = document.querySelector(`.panel.original .segment-block[data-segment-index="${{idx}}"]`);
        if (!source) continue;
        target.addEventListener("mouseenter", () => source.classList.add("is-hovered"));
        target.addEventListener("mouseleave", () => source.classList.remove("is-hovered"));
      }}

    }})();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    raw_text = input_path.read_text(encoding="utf-8")
    blocks = parse_blocks(raw_text)
    final_text = parse_final_section(raw_text)
    preface_text = ""
    if args.preface_file:
        preface_path = Path(args.preface_file)
        if not preface_path.is_absolute():
            preface_path = input_path.parent / preface_path
        if preface_path.exists():
            preface_text = preface_path.read_text(encoding="utf-8")
    output_html = render_html(blocks, final_text, preface_text=preface_text)

    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = input_path.with_suffix(".html")
    output_path.write_text(output_html, encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
