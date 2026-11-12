"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 4 · Day 3: PDF pages to images                              │
│                                                                      │
│  What: renders each page of a scanned answer sheet to a Pillow       │
│        image, cleans it up (contrast, grayscale, deskew, size cap)   │
│        and packs it as a data URL for a multimodal message.          │
│  Run:  from gradewise.transcription import render_pages, preprocess  │
│  ⏮ Built on: week2 (PyMuPDF for text PDFs)                          │
│  ⏭ Used by: vision.py, graph.py                                     │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path

import pymupdf
from PIL import Image, ImageOps

from gradewise.errors import DataNotFoundError

# 💡 DPI vs TOKEN COST: the trade-off you tune all week.
#    An A4 page is 8.27 × 11.69 inches, so the pixel size is simply inches × DPI:
#        100 DPI →  827 × 1169 px   cheap, but thin pen strokes start to break up
#        200 DPI → 1654 × 2339 px   the sweet spot for handwriting
#        300 DPI → 2481 × 3508 px   great for tiny writing, but ~2.25× the pixels of 200
#    Vision models bill by pixels, not by pages. OpenAI cuts a "high detail" image into
#    512 px tiles (≈170 tokens each + 85 base) after shrinking it to fit 2048 px and then
#    768 px on the short side; Anthropic charges roughly (width × height) / 750 tokens and
#    downsizes anything over ~1568 px on the long edge. So pixels beyond ~1600-2000 on the
#    long side mostly cost upload time and money without the model "seeing" more.
#    That is why we render at 200 DPI (crisp strokes) and THEN cap the size (MAX_SIDE).
# ⚠️ Pricing rules change: check your provider's vision docs before trusting these numbers.
DEFAULT_DPI = 200
MAX_SIDE = 1600


def _open_pdf(pdf_path: str | Path) -> pymupdf.Document:
    path = Path(pdf_path)
    if not path.exists():
        raise DataNotFoundError(
            f"Answer sheet not found: {path}",
            fix="Check the path. The synthetic sheets live in data/answer-sheets/synthetic/.",
        )
    return pymupdf.open(path)


def _pixmap_to_image(pix: pymupdf.Pixmap) -> Image.Image:
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def page_count(pdf_path: str | Path) -> int:
    """How many pages the PDF has (cheap: nothing is rendered)."""
    with _open_pdf(pdf_path) as doc:
        return doc.page_count


def render_pages(pdf_path: str | Path, dpi: int = DEFAULT_DPI) -> list[Image.Image]:
    """Render every page of a PDF to an RGB Pillow image, in page order."""
    with _open_pdf(pdf_path) as doc:
        # ⚠️ alpha=False: transparent PDFs would otherwise render on black, not paper-white.
        return [_pixmap_to_image(page.get_pixmap(dpi=dpi, alpha=False)) for page in doc]


def render_page(pdf_path: str | Path, page_number: int, dpi: int = DEFAULT_DPI) -> Image.Image:
    """Render ONE page (1-based). Used by double-pass verification to re-read a page at a higher DPI."""
    with _open_pdf(pdf_path) as doc:
        if not 1 <= page_number <= doc.page_count:
            raise DataNotFoundError(f"Page {page_number} does not exist; the PDF has {doc.page_count} pages.")
        return _pixmap_to_image(doc[page_number - 1].get_pixmap(dpi=dpi, alpha=False))


def estimate_skew(img: Image.Image, max_angle: float = 5.0, step: float = 0.5) -> float:
    """Estimate how many degrees a scanned page is rotated, with a projection profile.

    💡 Lines of handwriting are horizontal bands of ink. When the page is straight, summing
       the dark pixels per row gives sharp peaks (lines) and valleys (gaps); when it is tilted,
       the peaks smear out. We try small angles and keep the one with the "spikiest" profile.
    🚀 PRO: this is deliberately small and dependency-free. OpenCV's `minAreaRect` or a Hough
       transform are faster and more robust for production scanners.
    """
    small = img.convert("L")
    small.thumbnail((600, 600))
    ink = small.point(lambda p: 255 if p < 128 else 0)  # white = ink, so rotation fills with 0 (paper)
    best_angle, best_score = 0.0, -1.0
    steps = int(max_angle / step)
    for i in range(-steps, steps + 1):
        angle = i * step
        rotated = ink.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
        width, height = rotated.size
        data = rotated.tobytes()
        rows = [sum(data[r * width : (r + 1) * width]) for r in range(height)]
        mean = sum(rows) / len(rows)
        score = sum((v - mean) ** 2 for v in rows)  # variance: high = crisp lines
        if score > best_score:
            best_angle, best_score = angle, score
    return best_angle


def preprocess(
    img: Image.Image,
    *,
    max_side: int = MAX_SIDE,
    grayscale: bool = True,
    autocontrast: bool = True,
    deskew: bool = False,
) -> Image.Image:
    """Clean a page image before it goes to the vision model.

    1. grayscale: colour rarely helps with pen on paper, and it makes JPEGs smaller
    2. auto-contrast: faint pencil and grey scanner backgrounds become dark ink on white
    3. deskew (optional): straighten a tilted scan; off by default as it costs a little CPU
    4. size cap: shrink so the long side is at most `max_side` px (see DPI vs token cost above)
    """
    out = img.convert("L") if grayscale else img.convert("RGB")
    if autocontrast:
        # 💡 cutoff=1 ignores the darkest/lightest 1% so one speck of dust can't skew the stretch.
        out = ImageOps.autocontrast(out, cutoff=1)
    if deskew:
        angle = estimate_skew(out)
        if angle:
            out = out.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white")
    if max(out.size) > max_side:
        out = out.copy()
        out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return out


def to_data_url(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    """Encode an image as a `data:image/...;base64,...` URL for a LangChain image content block.

    💡 JPEG at quality 85 is ~5-10× smaller than PNG for a scanned page and looks the same to
       the model. Use PNG if you see compression smudges around thin strokes.
    """
    fmt = fmt.upper()
    if fmt == "JPG":
        fmt = "JPEG"
    buffer = io.BytesIO()
    if fmt == "JPEG":
        # ⚠️ JPEG can't store transparency or palette images.
        img = img.convert("L") if img.mode in ("L", "LA") else img.convert("RGB")
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
    else:
        img.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def image_block(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> dict[str, str]:
    """A LangChain 1.x image content block that EVERY provider accepts:
    {"type": "image", "base64": "...", "mime_type": "image/jpeg"}.

    ⚠️ Prefer this to {"type": "image", "url": to_data_url(img)}: OpenAI accepts a data URL there, but
       langchain-google-genai treats any "url" as a web address to download, so Gemini (custom mode)
       fails on it. The base64 form works for OpenRouter, OpenAI, Gemini and Anthropic alike.
    """
    url = to_data_url(img, fmt=fmt, quality=quality)
    mime, encoded = url[len("data:") :].split(";base64,", 1)
    return {"type": "image", "base64": encoded, "mime_type": mime}


def estimate_image_tokens(img: Image.Image) -> int:
    """A rough token estimate for one image, using OpenAI's "high detail" tile rule.

    ⚠️ Only an estimate for cost logging. Other providers count differently.
    """
    width, height = img.size
    scale = min(1.0, 2048 / max(width, height))
    width, height = width * scale, height * scale
    scale = min(1.0, 768 / min(width, height))
    width, height = width * scale, height * scale
    tiles = math.ceil(width / 512) * math.ceil(height / 512)
    return 85 + 170 * tiles
