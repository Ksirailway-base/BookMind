import re
import statistics
from pathlib import Path

import fitz           
from langchain_core.documents import Document

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 500

_EXERCISE_NUM_RE = re.compile(r"^\s*(\d{1,2}[\.\)]\s|[a-z][\.\)]\s|•\s|–\s|-\s)")

_CHAPTER_RE = re.compile(r"^(unit|chapter|module)\s+\d", re.IGNORECASE)
_SECTION_RE = re.compile(r"^\d+\.\d+\s", re.IGNORECASE)

_EXERCISE_KEYWORDS = re.compile(
    r"(complete the sentence|fill in|put in|choose the correct|"
    r"write the correct form|rewrite|correct the mistake|"
    r"make sentence|put the verb|use the word|"
    r"which is correct|match the|cross out|"
    r"write a sentence|complete using|put the word|"
    r"are these sentences right|correct or incorrect|"
    r"word bank|choose from these|box|options)",
    re.IGNORECASE,
)
_GRAMMAR_KEYWORDS = re.compile(
    r"(we use|we say|you can use|is used to|compare|"
    r"note that|the difference|we do not|instead of|"
    r"present simple|past simple|present perfect|past continuous|"
    r"future|conditional|passive|modal|gerund|infinitive|"
    r"preposition|article|pronoun|adjective|adverb|"
    r"clause|tense|singular|plural|countable|uncountable)",
    re.IGNORECASE,
)
_VOCAB_KEYWORDS = re.compile(
    r"(means|meaning|definition|synonym|opposite|"
    r"phrasal verb|idiom|expression|collocation|"
    r"word list|vocabulary|word bank)",
    re.IGNORECASE,
)
_EXAMPLE_KEYWORDS = re.compile(
    r"(for example|e\.g\.|example:|such as|here are some)",
    re.IGNORECASE,
)
_REFERENCE_KEYWORDS = re.compile(
    r"(appendix|index|contents|answer key|see unit|see page|"
    r"additional exercise|study guide)",
    re.IGNORECASE,
)

def _classify_text_level(
    text: str,
    font_size: float,
    is_bold: bool,
    median_size: float,
) -> str:
    if font_size >= median_size * 1.4:
        return "heading_1"
    if font_size >= median_size * 1.15:
        return "heading_2"
    if font_size >= median_size * 1.05 and is_bold:
        return "heading_3"
    if _EXERCISE_NUM_RE.match(text):
        return "list_item"
    return "paragraph"

def _extract_page_blocks(page: fitz.Page) -> list[dict]:
    page_dict = page.get_text("dict", sort=True)
    blocks_raw = page_dict.get("blocks", [])

    all_sizes = []
    for block in blocks_raw:
        if block.get("type") != 0:                     
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("text", "").strip():
                    all_sizes.append(span["size"])

    median_size = statistics.median(all_sizes) if all_sizes else 12.0

    extracted = []
    for block in blocks_raw:
        if block.get("type") != 0:
            continue

        block_text_parts = []
        block_sizes = []
        block_bold = False

        for line in block.get("lines", []):
            line_parts = []
            for span in line.get("spans", []):
                txt = span.get("text", "")
                if txt.strip():
                    line_parts.append(txt)
                    block_sizes.append(span["size"])
                                                      
                    if span.get("flags", 0) & 16:
                        block_bold = True
            if line_parts:
                block_text_parts.append("".join(line_parts))

        block_text = "\n".join(block_text_parts).strip()
        if not block_text:
            continue

        avg_size = statistics.mean(block_sizes) if block_sizes else median_size
        text_level = _classify_text_level(block_text, avg_size, block_bold, median_size)

        extracted.append({
            "text": block_text,
            "bbox": list(block["bbox"]),
            "font_size": round(avg_size, 1),
            "is_bold": block_bold,
            "text_level": text_level,
        })

    return extracted

def _normalize_heading(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def _update_hierarchy(block: dict, state: dict) -> None:
    text = block["text"]
    level = block["text_level"]

    if level == "heading_1" or _CHAPTER_RE.match(text):
        state["chapter"] = _normalize_heading(text)
        state["section"] = ""
    elif level == "heading_2" or _SECTION_RE.match(text):
        state["section"] = _normalize_heading(text)
    elif level == "heading_3":
        state["section"] = _normalize_heading(text)

def _is_exercise_start(text: str) -> bool:
    return bool(re.match(r"^\s*1[\.\)]\s", text))

_GRAMMAR_TERMS = [
    "present simple", "present continuous", "present perfect",
    "present perfect continuous",
    "past simple", "past continuous", "past perfect",
    "past perfect continuous",
    "future simple", "going to", "will",
    "first conditional", "second conditional", "third conditional",
    "zero conditional", "conditional",
    "passive", "active", "modal verb",
    "gerund", "infinitive", "participle",
    "relative clause", "reported speech", "indirect speech",
    "countable", "uncountable", "article",
    "preposition", "phrasal verb",
    "comparative", "superlative",
    "wish", "used to", "would rather",
]

_TASK_PATTERNS = {
    "fill_blank":  re.compile(r"(complete|fill in|put in|put the)", re.I),
    "choose":      re.compile(r"(choose the correct|which is correct|select|tick)", re.I),
    "rewrite":     re.compile(r"(rewrite|write .* again|transform)", re.I),
    "reorder":     re.compile(r"(put .* in .* order|rearrange|reorder)", re.I),
    "correct":     re.compile(r"(correct the|find the mistake|are .* right|right or wrong)", re.I),
    "match":       re.compile(r"(match|connect|link .* to)", re.I),
    "translate":   re.compile(r"(translate|say .* in)", re.I),
    "open_ended":  re.compile(r"(write about|describe|explain why|give your opinion)", re.I),
}

def _extract_grammar_terms(text: str) -> str:
    lower = text.lower()
    found = [t for t in _GRAMMAR_TERMS if t in lower]
    return ", ".join(found) if found else ""

def _detect_task_pattern(text: str, section: str) -> str:
    combined = f"{section} {text}"
    for pattern_name, regex in _TASK_PATTERNS.items():
        if regex.search(combined):
            return pattern_name
    return "other"

def _classify_content_type(text: str, section: str) -> str:
    combined = f"{section} {text}"

    if _EXERCISE_KEYWORDS.search(combined):
        return "exercise"
    if _REFERENCE_KEYWORDS.search(combined):
        return "reference"
    if _VOCAB_KEYWORDS.search(combined):
        return "vocabulary"
    if _EXAMPLE_KEYWORDS.search(combined):
        return "example"
    if _GRAMMAR_KEYWORDS.search(combined):
        return "rule"
    return "other"

def _flush_chunk(
    buffer: list[dict],
    book_name: str,
    page_num: int,
    hierarchy: dict,
) -> Document | None:
    if not buffer:
        return None

    combined_text = "\n".join(b["text"] for b in buffer)
    if not combined_text.strip():
        return None

    bbox = buffer[0]["bbox"]
    levels = [b["text_level"] for b in buffer]
    dominant_level = max(set(levels), key=levels.count)
    avg_font = round(
        statistics.mean(b["font_size"] for b in buffer), 1
    )

    section = hierarchy.get("section", "")
    content_type = _classify_content_type(combined_text, section)

    metadata = {
        "book": book_name,
        "page": page_num,
        "chapter": hierarchy.get("chapter", ""),
        "section": section,
        "text_level": dominant_level,
        "content_type": content_type,
        "bbox": bbox,
        "font_size_avg": avg_font,
        "task_pattern": "",
        "grammar_terms": "",
        "related_rule": "",
    }

    if content_type == "exercise":
        metadata["task_pattern"] = _detect_task_pattern(combined_text, section)

    if content_type == "rule":
        metadata["grammar_terms"] = _extract_grammar_terms(combined_text)
        hierarchy["last_rule"] = f"{hierarchy.get('chapter', '')} > {section}"

    if content_type == "example":
        metadata["related_rule"] = hierarchy.get("last_rule", "")

    return Document(page_content=combined_text, metadata=metadata)

def parse_pdf(pdf_path: str | Path) -> list[Document]:
    path = Path(pdf_path)
    book_name = path.name
    if book_name.lower().endswith(".pdf"):
        book_name = book_name[:-4]

    doc = fitz.open(str(path))
    all_chunks: list[Document] = []

    hierarchy = {"chapter": "", "section": ""}
    buffer: list[dict] = []
    buffer_len = 0
    current_page = 0

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1             
        blocks = _extract_page_blocks(page)

        for block in blocks:
            _update_hierarchy(block, hierarchy)

            block_len = len(block["text"])

            if buffer_len + block_len > CHUNK_SIZE and buffer:
                                                                       
                if not _is_exercise_start(block["text"]):
                    chunk = _flush_chunk(buffer, book_name, current_page, hierarchy)
                    if chunk:
                        all_chunks.append(chunk)
                    buffer = []
                    buffer_len = 0

            if block_len > CHUNK_SIZE and not buffer:
                text = block["text"]
                while len(text) > CHUNK_SIZE:
                                                                       
                    split_at = text.rfind("\n", 0, CHUNK_SIZE)
                    if split_at < CHUNK_SIZE // 2:
                        split_at = text.rfind(". ", 0, CHUNK_SIZE)
                    if split_at < CHUNK_SIZE // 4:
                        split_at = CHUNK_SIZE

                    sub_block = block.copy()
                    sub_block["text"] = text[:split_at].strip()
                    chunk = _flush_chunk([sub_block], book_name, page_num, hierarchy)
                    if chunk:
                        all_chunks.append(chunk)
                    text = text[split_at:].strip()

                if text:
                    sub_block = block.copy()
                    sub_block["text"] = text
                    buffer = [sub_block]
                    buffer_len = len(text)
                    current_page = page_num
                continue

            buffer.append(block)
            buffer_len += block_len
            current_page = page_num

    if buffer:
        chunk = _flush_chunk(buffer, book_name, current_page, hierarchy)
        if chunk:
            all_chunks.append(chunk)

    doc.close()
    return all_chunks
