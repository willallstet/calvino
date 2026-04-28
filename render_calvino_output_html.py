#!/usr/bin/env python3
import argparse
import html
import re
from pathlib import Path
from typing import Dict, List


SENTENCE_HEADER_RE = re.compile(r"^\[Sentence\s+\d+\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Final Replaced Text as minimal centered HTML with per-line source fonts."
    )
    parser.add_argument("input_file", help="Path to calvino_fineweb_output_*.txt")
    parser.add_argument(
        "-o",
        "--output-file",
        help="Output HTML file path (default: input path with .html extension).",
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
            if SENTENCE_HEADER_RE.match(s) or s == "Final Replaced Text":
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
        if line.strip() == "Final Replaced Text":
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


def render_segment(
    raw_text: str,
    href: str = "",
    style: str = "",
    segment_index: int | None = None,
    interactive: bool = False,
) -> str:
    text = raw_text.strip()
    if not text:
        return ""

    is_quote = text.startswith("> ")
    if is_quote:
        text = text[2:].lstrip()

    escaped_text = html.escape(text)
    escaped_href = html.escape(href.strip())
    escaped_style = html.escape(style.strip())

    index_attr = f' data-segment-index="{segment_index}"' if segment_index is not None else ""
    hover_class = " hover-target" if interactive else ""

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
        return f'<blockquote class="segment-block segment-quote{hover_class}"{index_attr}>{inner}</blockquote>'
    return f'<p class="segment-block segment-paragraph{hover_class}"{index_attr}>{inner}</p>'


def render_html(blocks: List[Dict[str, str]], final_text: str) -> str:
    original_segments: List[str] = []
    final_segments: List[str] = []
    for idx, block in enumerate(blocks):
        original_segment = render_segment(block["original"], segment_index=idx)
        final_segment = render_segment(
            block["match"],
            href=block.get("source_url", ""),
            style=style_for_block(block),
            segment_index=idx,
            interactive=True,
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
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Final Replaced Text</title>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      background: #ffffff;
      color: #000000;
    }}
    body {{
      font-family: serif;
    }}
    .wrap {{
      box-sizing: border-box;
      width: min(1600px, 96vw);
      margin: 0 auto;
      padding: 24px;
      position: relative;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 20px;
      position: sticky;
      top: 24px;
      height: calc(100vh - 48px);
      align-items: stretch;
    }}
    .panel {{
      padding: 8px 12px;
      min-height: 0;
      box-sizing: border-box;
      overflow: hidden;
    }}
    .panel.original {{
      font-family: Arial, Helvetica, sans-serif;
    }}
    .chunk {{
      color: #000000;
      height: 100%;
      overflow: hidden;
    }}
    .segment {{
      line-height: 1.4;
      color: #000000;
      text-decoration: none;
    }}
    .segment-paragraph {{
      margin: 0 0 0.9em 0;
    }}
    .segment-paragraph:last-child {{
      margin-bottom: 0;
    }}
    .segment-quote {{
      margin: 0.6em 0;
      padding-left: 0.9em;
      border-left: 2px solid #c7c7c7;
    }}
    .panel.original .segment-block {{
      transition: background-color 120ms ease-in-out, color 120ms ease-in-out;
    }}
    .panel.original .segment-block.is-hovered {{
      background: #000000;
      color: #ffffff;
    }}
    .panel.original .segment-block.is-hovered .segment {{
      color: #ffffff !important;
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
    <div class="grid">
      <section class="panel original">
        <div class="chunk sync-scroll">{joined_original}</div>
      </section>
      <section class="panel">
        <div class="chunk sync-scroll">{joined_final}</div>
      </section>
    </div>
    <div id="scroll-track" aria-hidden="true"></div>
  </div>
  <script>
    (function () {{
      const columns = Array.from(document.querySelectorAll(".chunk.sync-scroll"));
      const track = document.getElementById("scroll-track");
      const mobileQuery = window.matchMedia("(max-width: 1024px)");

      function syncColumns() {{
        if (mobileQuery.matches) return;
        const docRange = document.documentElement.scrollHeight - window.innerHeight;
        const progress = docRange > 0 ? window.scrollY / docRange : 0;
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
        track.style.height = `${{Math.max(1, maxRange + window.innerHeight)}}px`;
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
    output_html = render_html(blocks, final_text)

    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = input_path.with_suffix(".html")
    output_path.write_text(output_html, encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
