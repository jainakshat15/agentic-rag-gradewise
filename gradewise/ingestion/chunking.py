"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  📍 Week 2 · Day 2: Structure-aware chunking                         │
│                                                                      │
│  What this file does                                                 │
│    Splits each page on its Markdown headings first (## / ###), so a  │
│    chunk never mixes two topics, then cuts long sections with a      │
│    RecursiveCharacterTextSplitter. Every chunk keeps its section     │
│    path ("Unit 1: Operating Systems > 1.6 Deadlocks") in metadata.   │
│                                                                      │
│  Use it                                                              │
│    from gradewise.ingestion.chunking import chunk_documents          │
│    chunks = chunk_documents(pages)                                   │
│                                                                      │
│  ⏮ Built on: loaders.py   ⏭ Used by: pipeline.py                    │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from gradewise.ingestion.loaders import clean_heading

CHUNK_SIZE = 1000  # 💡 characters, roughly 250 tokens: about one idea per chunk
CHUNK_OVERLAP = 150  # 💡 overlap so a sentence cut at a boundary still appears whole somewhere
MIN_CHUNK_SIZE = 200  # 💡 smaller leftovers ("storage.") are merged into a neighbour in the same section

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def chunk_documents(
    pages: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    contextual_header: bool = True,
    min_chunk_size: int = MIN_CHUNK_SIZE,
) -> list[Document]:
    """Split page Documents into chunks that respect the document's headings.

    Each chunk gets: everything from the page's metadata, plus `section` (the nearest
    ## heading), `section_path` (h1 > h2 > h3) and `chunk_index` (position in its file).

    Chunks shorter than `min_chunk_size` characters are merged into a neighbour from the
    same section (pass 0 to switch this off). A short section with no neighbour stays as it is.
    """
    header_splitter = MarkdownHeaderTextSplitter(_HEADERS, strip_headers=False)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # 1️⃣ Split: (text, page metadata, section, section_path) for every raw piece, in reading order.
    raw: list[tuple[str, dict, str, str]] = []
    for page in pages:
        for piece in header_splitter.split_text(page.page_content):
            heads = {k: clean_heading(v) for k, v in piece.metadata.items()}
            # ⚠️ Text before the first heading on a page belongs to the section that began on
            #    an EARLIER page. The loader carried that section forward for us.
            section = heads.get("h2") or page.metadata.get("section", "") or heads.get("h1", "")
            # 💡 The document title is always the root, so every chunk's path starts the same way.
            path_parts = [page.metadata.get("title", ""), section, heads.get("h3", "")]
            section_path = " > ".join(p for p in path_parts if p)
            for text in text_splitter.split_text(piece.page_content):
                if text.strip():
                    raw.append((text, page.metadata, section, section_path))

    # 2️⃣ Merge tiny leftovers into their neighbour (same file, same section path).
    raw = _merge_small(raw, min_chunk_size)

    # 3️⃣ Number the chunks per file and add the contextual header.
    chunks: list[Document] = []
    index_in_file: dict[str, int] = {}
    for text, meta, section, section_path in raw:
        source = meta["source"]
        n = index_in_file.get(source, 0)
        index_in_file[source] = n + 1
        # 🚀 PRO: "contextual chunk headers". Prepending the title and section to the
        #    text means both the embedding AND the BM25 index know a chunk about
        #    "the four conditions" is about deadlocks, even if the word isn't in it.
        content = f"[{meta.get('title', '')} > {section}]\n{text}" if contextual_header else text
        chunks.append(
            Document(
                page_content=content,
                metadata={**meta, "section": section, "section_path": section_path, "chunk_index": n},
            )
        )
    return chunks


def _join(first: str, second: str) -> str:
    """Glue two neighbouring chunks, dropping the text they share because of the overlap."""
    # ⚠️ Ignore tiny "overlaps" (a shared letter or two is a coincidence, not the splitter's overlap).
    for size in range(min(len(first), len(second)), 9, -1):
        if first.endswith(second[:size]):
            return first + second[size:]
    return f"{first}\n{second}"


def _merge_small(raw: list[tuple[str, dict, str, str]], min_size: int) -> list[tuple[str, dict, str, str]]:
    """Merge every piece shorter than `min_size` into the previous piece of the same section,
    or into the next one when it is the section's first piece.

    ⚠️ A one-word chunk like "storage." is pure noise: it matches almost nothing, and when it
       does match it gives the LLM no context. Merging it keeps the words searchable.
    """
    if min_size <= 0:
        return raw

    def same_section(a: tuple, b: tuple) -> bool:
        return a[1]["source"] == b[1]["source"] and a[3] == b[3]

    merged: list[tuple[str, dict, str, str]] = []
    for item in raw:
        if merged and same_section(merged[-1], item) and (len(item[0]) < min_size or len(merged[-1][0]) < min_size):
            prev = merged[-1]
            merged[-1] = (_join(prev[0], item[0]), prev[1], prev[2], prev[3])
        else:
            merged.append(item)
    return merged
