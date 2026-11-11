"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🎨 scripts/make_assets.py: generates the repo's images              │
│                                                                      │
│  Creates, in the course palette (same as the slides):                │
│    assets/callouts/*.png       notebook callout icons                │
│    assets/hero.png             README hero image                     │
│    assets/roadmap/*.png        6-week roadmap (overview + per week)  │
│    assets/banners/wNdM.png     one banner per day notebook           │
│                                                                      │
│  Run:  uv run scripts/make_assets.py                                 │
│  💡 Learners never need to run this: the images are committed.       │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import platform
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

# ── 🎨 Palette (COURSE_PLAN §8) ──────────────────────────────────────────
BG = "#0B1020"
SURFACE = "#141B34"
PRIMARY = "#7C5CFF"
ACCENT = "#22D3EE"
TEXT = "#E6E9F5"
MUTED = "#9AA3C7"
OK = "#34D399"

WEEKS = [
    (1, "Foundations", "Exam → JSON", "#7C5CFF"),
    (2, "RAG", "Knowledge base", "#22D3EE"),
    (3, "Agents & LangGraph", "Self-correcting exam graph", "#34D399"),
    (4, "Agentic RAG & Vision", "Read handwriting", "#FBBF24"),
    (5, "Grading & Evals", "Report cards", "#F472B6"),
    (6, "Ship It", "App + Docker", "#F87171"),
]

DAY_TITLES = {
    (1, 1): "Welcome, Setup & Your First Call",
    (1, 2): "LLMs & the LangChain 1.x Ecosystem",
    (1, 3): "Prompts & Structured Output",
    (1, 4): "Multimodal Inputs & Tools",
    (1, 5): "Project Day: Exam Paper → JSON",
    (2, 1): "Why RAG? Embeddings Explained",
    (2, 2): "Loading & Chunking PDFs",
    (2, 3): "Vector Stores & Idempotent Ingestion",
    (2, 4): "Hybrid Search, Reranking & Citations",
    (2, 5): "Project Day: Measuring Retrieval",
    (3, 1): "What Is an Agent?",
    (3, 2): "create_agent & Middleware",
    (3, 3): "LangGraph: State, Nodes & Edges",
    (3, 4): "Send, Checkpoints & Human-in-the-Loop",
    (3, 5): "Project Day: The Exam Graph",
    (4, 1): "Agentic RAG",
    (4, 2): "Grounded Answers & the Evidence Retriever",
    (4, 3): "Vision Models & Page Images",
    (4, 4): "Faithful Transcription & Mapping",
    (4, 5): "Project Day: The Transcription Subgraph",
    (5, 1): "LLM-as-Grader",
    (5, 2): "Tools, Parallel Grading & Guardrails",
    (5, 3): "Report Cards & Chat With Results",
    (5, 4): "Project Day: GradeWise End to End",
    (5, 5): "Evaluating the Grader",
    (6, 1): "Production Architecture & FastAPI",
    (6, 2): "The Streamlit Dashboard",
    (6, 3): "Streaming Progress & the Report Viewer",
    (6, 4): "Docker & Deployment",
    (6, 5): "Responsible AI & the Capstone Showcase",
}

CALLOUTS = {
    "important": ("#FF7800", "!"),
    "exercise": ("#FF9F1C", "✎"),
    "business": ("#22C55E", "$"),
    "stop": ("#EF4444", "✋"),
    "resources": ("#3B82F6", "≡"),
    "gradewise": ("#7C5CFF", "G"),
    "thankyou": ("#EC4899", "♥"),
}


# ── fonts ────────────────────────────────────────────────────────────────
def font(size: int, bold: bool = False, symbol: bool = False) -> ImageFont.FreeTypeFont:
    """Find a nice system font on Windows / Mac / Linux, falling back to Pillow's default."""
    candidates: list[str] = []
    if platform.system() == "Windows":
        win = Path("C:/Windows/Fonts")
        if symbol:
            candidates += [str(win / "seguisym.ttf")]
        candidates += [str(win / ("segoeuib.ttf" if bold else "segoeui.ttf")), str(win / "arial.ttf")]
    elif platform.system() == "Darwin":
        candidates += ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/System/Library/Fonts/Helvetica.ttc"]
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def centered(draw: ImageDraw.ImageDraw, box, text, fnt, fill) -> None:
    x0, y0, x1, y1 = box
    left, top, right, bottom = draw.textbbox((0, 0), text, font=fnt)
    draw.text(((x0 + x1 - (right - left)) / 2 - left, (y0 + y1 - (bottom - top)) / 2 - top), text, font=fnt, fill=fill)


def gradient(w: int, h: int, top: str = BG, bottom: str = SURFACE) -> Image.Image:
    img = Image.new("RGB", (w, h), top)
    t, b = Image.new("RGB", (1, 1), top).getpixel((0, 0)), Image.new("RGB", (1, 1), bottom).getpixel((0, 0))
    draw = ImageDraw.Draw(img)
    for y in range(h):
        mix = y / max(h - 1, 1)
        draw.line([(0, y), (w, y)], fill=tuple(int(t[i] + (b[i] - t[i]) * mix) for i in range(3)))
    return img


# ── generators ───────────────────────────────────────────────────────────
def make_callouts() -> None:
    out = ASSETS / "callouts"
    out.mkdir(parents=True, exist_ok=True)
    for name, (colour, glyph) in CALLOUTS.items():
        size = 300
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([10, 10, size - 10, size - 10], radius=70, fill=colour)
        d.rounded_rectangle([10, 10, size - 10, size - 10], radius=70, outline="white", width=8)
        centered(d, (10, 10, size - 10, size - 20), glyph, font(170, bold=True, symbol=True), "white")
        img.save(out / f"{name}.png")


def make_hero() -> None:
    w, h = 1600, 640
    img = gradient(w, h)
    d = ImageDraw.Draw(img)
    d.text((80, 70), "GradeWise", font=font(96, bold=True), fill=TEXT)
    d.text((84, 190), "Agentic RAG with LangChain & LangGraph · a 6-week build", font=font(36), fill=MUTED)
    stages = [
        ("Course PDFs", "RAG knowledge base"),
        ("Exam paper", "→ JSON rubric"),
        ("Handwritten", "→ faithful text"),
        ("Grading agent", "rubric + evidence"),
        ("Report cards", "stored & queryable"),
    ]
    bw, gap, y = 252, 44, 330
    x = 80
    for i, (title, sub) in enumerate(stages):
        colour = WEEKS[min(i + 1, 5)][3] if i else WEEKS[1][3]
        d.rounded_rectangle([x, y, x + bw, y + 190], radius=24, fill=SURFACE, outline=colour, width=4)
        d.text((x + 24, y + 30), f"{i + 1}", font=font(44, bold=True), fill=colour)
        d.text((x + 24, y + 92), title, font=font(30, bold=True), fill=TEXT)
        d.text((x + 24, y + 136), sub, font=font(22), fill=MUTED)
        if i < len(stages) - 1:
            ax = x + bw + 6
            d.line([(ax, y + 95), (ax + gap - 12, y + 95)], fill=ACCENT, width=5)
            d.polygon([(ax + gap - 12, y + 85), (ax + gap - 2, y + 95), (ax + gap - 12, y + 105)], fill=ACCENT)
        x += bw + gap
    d.text(
        (80, 570),
        "Asha uploads the exam  ·  GradeWise reads, grades and explains  ·  a human stays in control",
        font=font(26),
        fill=ACCENT,
    )
    img.save(ASSETS / "hero.png")


def make_roadmap(highlight: int | None) -> Image.Image:
    w, h = 1600, 420
    img = gradient(w, h)
    d = ImageDraw.Draw(img)
    title = "Your 6-week roadmap" if highlight is None else f"Week {highlight} of 6: you are here"
    d.text((70, 40), title, font=font(44, bold=True), fill=TEXT)
    xs = [140 + i * 264 for i in range(6)]
    y = 220
    d.line([(xs[0], y), (xs[-1], y)], fill="#2A335C", width=10)
    if highlight:
        d.line([(xs[0], y), (xs[highlight - 1], y)], fill=PRIMARY, width=10)
    for (num, theme, milestone, colour), x in zip(WEEKS, xs, strict=True):
        done = highlight is not None and num < highlight
        current = highlight == num
        r = 58 if current else 44
        fill = colour if (current or highlight is None) else (OK if done else SURFACE)
        d.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline="white" if current else "#2A335C", width=5)
        centered(
            d,
            (x - r, y - r, x + r, y + r),
            "✓" if done else str(num),
            font(44 if current else 36, True, True),
            BG if (current or highlight is None or done) else MUTED,
        )
        dim = highlight is not None and not current
        centered(d, (x - 130, y + 70, x + 130, y + 110), theme, font(22, bold=True), MUTED if dim else TEXT)
        centered(d, (x - 130, y + 108, x + 130, y + 145), milestone, font(19), MUTED if dim else colour)
    return img


def make_roadmaps() -> None:
    out = ASSETS / "roadmap"
    out.mkdir(parents=True, exist_ok=True)
    make_roadmap(None).save(out / "overview.png")
    for week, *_ in WEEKS:
        make_roadmap(week).save(out / f"week{week}.png")


def make_banner(week: int, day: int, title: str) -> None:
    out = ASSETS / "banners"
    out.mkdir(parents=True, exist_ok=True)
    colour = WEEKS[week - 1][3]
    w, h = 1600, 300
    img = gradient(w, h)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 18, h], fill=colour)
    d.text((70, 52), f"WEEK {week}  ·  DAY {day}", font=font(32, bold=True), fill=colour)
    d.text((70, 110), title, font=font(64, bold=True), fill=TEXT)
    d.text((70, 214), "Agentic RAG with LangChain & LangGraph  ·  GradeWise", font=font(26), fill=MUTED)
    # progress dots: 30 days in the course
    idx = (week - 1) * 5 + day
    for i in range(30):
        x = 1050 + (i % 15) * 34
        y = 222 + (i // 15) * 30
        d.ellipse([x, y, x + 18, y + 18], fill=colour if i < idx else "#2A335C")
    img.save(out / f"w{week}d{day}.png")


if __name__ == "__main__":
    make_callouts()
    make_hero()
    make_roadmaps()
    for (week, day), title in DAY_TITLES.items():
        make_banner(week, day, title)
    from gradewise.console import success

    success("Assets generated", {"Folder": ASSETS})
