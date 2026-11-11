"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📚 scripts/build_knowledge_base.py: course notes → PDFs             │
│                                                                      │
│  Turns the Markdown course notes in scripts/knowledge_base_source/   │
│  into the styled PDFs in data/knowledge-base/ that GradeWise         │
│  ingests in week 2.                                                  │
│                                                                      │
│  Run:  uv run scripts/build_knowledge_base.py                        │
│  💡 Learners never need to run this: the PDFs are committed. It's    │
│     here so you can see exactly where the data came from, and        │
│     write your own course material for your own GradeWise variant.   │
│                                                                      │
│  Why PDFs and not Markdown? Real course material arrives as PDFs:    │
│  page numbers, headers and footers, tables. Week 2 teaches you to    │
│  handle exactly that.                                                │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf
from markdown_it import MarkdownIt

from gradewise.console import banner, step, success

ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path(__file__).resolve().parent / "knowledge_base_source"
OUTPUT = ROOT / "data" / "knowledge-base"

PAGE = pymupdf.paper_rect("a4")
MARGIN = (56, 64, -56, -64)  # left, top, right, bottom (points)

# 💡 PyMuPDF's Story engine understands a useful subset of HTML + CSS.
CSS = """
* { font-family: sans-serif; }
body { font-size: 10.5pt; line-height: 1.45; color: #1f2433; }
h1 { font-size: 22pt; color: #3b2a9e; margin-bottom: 2pt; }
h2 { font-size: 14pt; color: #3b2a9e; margin-top: 14pt; }
h3 { font-size: 11.5pt; color: #1f6f8b; margin-top: 10pt; }
p { margin-top: 3pt; margin-bottom: 5pt; text-align: justify; }
li { margin-bottom: 2pt; }
code { font-family: monospace; font-size: 9.5pt; color: #8a1c4a; }
pre { font-family: monospace; font-size: 9pt; background-color: #f3f1fb; padding: 6pt; }
table { border-collapse: collapse; margin-top: 4pt; margin-bottom: 8pt; width: 100%; }
th { color: #3b2a9e; font-weight: bold; font-size: 9.5pt; padding: 3pt; text-align: left; border: 1px solid #c9c3ef; }
td { border: 1px solid #c9c3ef; font-size: 9.5pt; padding: 3pt; }
"""

md = MarkdownIt("commonmark", {"breaks": True}).enable("table")


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Read the small `--- key: value ---` header at the top of each source file."""
    meta: dict[str, str] = {}
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if match:
        for line in match.group(1).splitlines():
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip('"')
        text = text[match.end() :]
    return meta, text


def render_pdf(source: Path) -> Path:
    meta, body = split_front_matter(source.read_text(encoding="utf-8"))
    html = f"<body>{md.render(body)}</body>"
    target = OUTPUT / f"{source.stem}.pdf"

    # 1) Lay the HTML out across as many A4 pages as it needs
    story = pymupdf.Story(html=html, user_css=CSS)
    writer = pymupdf.DocumentWriter(str(target))
    more = True
    while more:
        device = writer.begin_page(PAGE)
        more, _ = story.place(PAGE + MARGIN)
        story.draw(device)
        writer.end_page()
    writer.close()

    # 2) Re-open to add a running header, page numbers and PDF metadata.
    #    ⚠️ These headers/footers repeat on every page: in week 2 you'll see why
    #    they need cleaning out before chunking!
    doc = pymupdf.open(target)
    total = doc.page_count
    for number, page in enumerate(doc, start=1):
        page.insert_text(
            (56, 36), f"CS201 Systems Fundamentals · {meta.get('title', '')}", fontsize=8, color=(0.45, 0.45, 0.55)
        )
        page.insert_text(
            (56, PAGE.height - 30), "Northbridge Institute · School of Computing", fontsize=8, color=(0.45, 0.45, 0.55)
        )
        page.insert_text(
            (PAGE.width - 110, PAGE.height - 30), f"Page {number} of {total}", fontsize=8, color=(0.45, 0.45, 0.55)
        )
    doc.set_metadata(
        {
            "title": meta.get("title", source.stem),
            "author": "Dr. Asha Raman (fictional), for the GradeWise course",
            "subject": meta.get("subject", ""),
            "keywords": f"CS201; unit {meta.get('unit', '')}; Northbridge Institute",
            "creator": "scripts/build_knowledge_base.py",
        }
    )
    doc.saveIncr()
    doc.close()
    return target


if __name__ == "__main__":
    banner("Building the CS201 knowledge base", "Markdown course notes → styled PDFs")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    built = {}
    for source in sorted(SOURCE.glob("*.md")):
        step(f"Rendering {source.name}")
        pdf = render_pdf(source)
        with pymupdf.open(pdf) as doc:
            built[pdf.name] = f"{doc.page_count} pages"
    success("Knowledge base built", built, next_step="week 2 ingests these PDFs into the vector store")
