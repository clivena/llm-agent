"""文档分块。

分块是 RAG 里最容易被忽视、又最先决定上限的一步：
切得太碎，召回片段缺少上下文，模型答不出；切得太整，一个块里混进多个主题，
向量被平均掉，召回分数普遍偏低且区分度差。

这里的策略是「章节优先 + 段落优先 + 句子边界回退 + 固定重叠」：
  1. 章节（markdown 标题）是最高优先级边界：一个块尽量只讲一个话题，
     否则向量把两三个主题平均掉，召回分数低且区分度差，答案也会串；
  2. 段落（空行分隔）是天然语义单元，其次优先在段落边界断开；
  3. 单个段落超过上限时，回退到句末标点（。！？；）断开，避免切在句子中间；
  4. 相邻块保留 overlap 字符重叠，防止答案恰好落在接缝处。

所有块都记录相对原文的字符区间 [start, end)，页面上的引用高亮与
Citation.verify() 的强校验都依赖它 —— 区间一旦对不上，引用就是假的。
"""

from __future__ import annotations

import hashlib
import re

from agentkit.models import Chunk, Document

# 句末标点：中英文都要覆盖，否则中文文档会退化成"整段一个块"
_SENTENCE_END = "。！？；!?;.\n"
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
# 断开后允许的最小块长度比例：太小会产生一堆碎片块
_MIN_FILL_RATIO = 0.55
# 章节正文短于这个长度视为"只有标题"，并进相邻章节
_MIN_SECTION_BODY = 24
_CJK_RANGES = ((0x3000, 0x303F), (0x4E00, 0x9FFF), (0xFF00, 0xFFEF), (0x2000, 0x206F))


def _is_cjk(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def normalize_text(text: str) -> str:
    """把硬折行还原成正常段落。**这一步不能省。**

    markdown、PDF 抽取、OCR 出来的文本普遍在句子中间折行，
    单个换行只是排版换行，不是段落边界。不还原的话，切块会把
    "……并记录降级原因，\\n而不是继续使用……" 当成两段，
    切出以半句开头的块 —— 于是答案里出现"而不是继续使用未重排的结果"
    这种没头没尾的开头，而且看起来很像模型在胡说，排查时却会去怀疑模型。

    段落之间保留空行；中文行之间不补空格，涉及西文时补一个空格。
    """
    paragraph_sep = re.compile(r"\n\s*\n")
    out: list[str] = []
    for para in paragraph_sep.split(text.strip()):
        lines = [line.strip() for line in para.splitlines() if line.strip()]
        if not lines:
            continue
        joined = lines[0]
        for line in lines[1:]:
            if _is_cjk(joined[-1]) or _is_cjk(line[0]):
                joined += line
            else:
                joined += " " + line
        out.append(joined)
    return "\n\n".join(out)


def doc_id_for(source: str) -> str:
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]


def chunk_id_for(doc_id: str, ordinal: int, text: str) -> str:
    digest = hashlib.sha1(f"{doc_id}:{ordinal}:{text}".encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}-{ordinal:03d}-{digest}"


def _sentence_breaks(text: str) -> list[int]:
    """返回"可以在这里断开"的位置集合（断点在标点之后）。"""
    return [i + 1 for i, ch in enumerate(text) if ch in _SENTENCE_END]


def _headings(text: str) -> list[tuple[int, str]]:
    """Markdown 标题及其起始偏移，用来给块标注所属章节。"""
    out: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        m = _HEADING.match(line.rstrip("\n"))
        if m:
            out.append((offset, m.group(2).strip()))
        offset += len(line)
    return out


def _section_at(headings: list[tuple[int, str]], pos: int) -> str:
    current = ""
    for start, title in headings:
        if start <= pos:
            current = title
        else:
            break
    return current


def _snap(end: int, start: int, size: int, breaks: list[int]) -> int:
    """把窗口右边界吸附到就近的句末/段末，避免切在句子中间。

    只在章节内部使用 —— 章节边界由调用方直接把 limit 卡死，
    这样块永远不可能横跨两个话题。
    """
    floor = start + int(size * _MIN_FILL_RATIO)
    best = None
    for b in breaks:
        if b < floor:
            continue
        if b > end:
            break
        best = b
    return best if best is not None else end


def _is_title_only(text: str, start: int, end: int) -> bool:
    """这一段是否只有标题、没有正文。"""
    body = "".join(
        line for line in text[start:end].splitlines(keepends=True)
        if not _HEADING.match(line.rstrip("\n"))
    )
    return len(body.strip()) < _MIN_SECTION_BODY


def _split_sections(text: str) -> list[tuple[int, int]]:
    """按 markdown 标题切出章节区间 [start, end)。

    章节是比段落更强的语义单元：一个块横跨两节，向量会把两个话题
    平均在一起，召回区分度下降，答案也会把不相干的两段缝在一起。
    所以这里先把文档切成章节，再在章节内部做窗口切分。

    只有标题、没有正文的段（比如文档开头的 H1 大标题）会被并进相邻章节，
    避免产生一个 7 个字的垃圾块。
    """
    bounds = [start for start, _ in _headings(text)]
    if not bounds:
        return [(0, len(text))] if text.strip() else []

    starts = ([0] if bounds[0] > 0 else []) + bounds
    raw: list[tuple[int, int]] = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        if e > s:
            raw.append((s, e))

    merged: list[tuple[int, int]] = []
    for s, e in raw:
        if merged and _is_title_only(text, s, e):
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    # 纯标题段出现在最前面时没有"上一段"可并，改并到下一段
    if len(merged) > 1 and _is_title_only(text, *merged[0]):
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)
    return merged


def chunk_text(text: str, size: int = 320, overlap: int = 64) -> list[tuple[int, int, str]]:
    """返回 [(start, end, section)]。start/end 已去除首尾空白，保证 text[start:end] 即块内容。"""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be in [0, size)")

    stripped_offsets = _trim_to_content(text)
    if not text.strip():
        return []

    headings = _headings(text)
    breaks = _sentence_breaks(text)
    spans: list[tuple[int, int, str]] = []

    content_start, content_end = stripped_offsets
    for raw_start, raw_end in _split_sections(text):
        # 章节边界是硬边界：窗口只在本节内滑，重叠也只在本节内重叠。
        # 这里先把区间两侧的空白收掉，否则 end 永远追不上带空白的 seg_end，
        # 循环会以每次 1~2 个字符的速度空转（曾经真的跑出过 178 个碎片块）。
        seg_start = _trim_left(text, max(raw_start, content_start), content_end)
        seg_end = _trim_right(text, seg_start, min(raw_end, content_end))
        if seg_end <= seg_start:
            continue
        start = seg_start
        while start < seg_end:
            end = min(start + size, seg_end)
            if end < seg_end:
                end = _snap(end, start, size, breaks)
            end = _trim_right(text, start, end)
            if end <= start:
                # 极端情况（整段都是空白）保护，避免死循环
                break
            spans.append((start, end, _section_at(headings, start)))

            if end >= seg_end:
                break
            nxt = _advance(text, start, end, overlap, breaks, seg_end)
            if nxt <= start:  # 保险：起点必须严格前进
                break
            start = nxt
    return spans


def _advance(text: str, start: int, end: int, overlap: int, breaks: list[int],
             limit: int) -> int:
    """计算下一个块的起点，保证新块从完整的句子开始。

    绝不能直接取 end - overlap：那样块会从半句开始，带来两个后果 ——
    向量里混进无意义的碎片，以及答案拼接时出现
    "……和真实" + "用量”三步放进…" 这种接缝断裂，看起来像模型在胡说。

    三级回退，越往后越宽松：
      ① 块尾两倍重叠长度内的句末断点 —— 既保住重叠，又不至于让新块几乎没推进；
      ② 块内任意句末断点 —— 长句段落里窗口内可能一个断点都没有；
      ③ 直接取 end - overlap —— 整块都没有句末标点（表格式文本）才会走到这里。
    """
    target = end - overlap
    floor = max(start + 1, end - 2 * overlap)
    window = [b for b in breaks if floor <= b < end]
    if window:
        chosen = min(window, key=lambda b: abs(b - target))
    else:
        whole = [b for b in breaks if start < b < end]
        chosen = min(whole, key=lambda b: abs(b - target)) if whole else target
    return _trim_left(text, max(chosen, start + 1), limit)


def _trim_to_content(text: str) -> tuple[int, int]:
    if not text.strip():
        return 0, 0
    start = len(text) - len(text.lstrip())
    end = len(text.rstrip())
    return start, end


def _trim_right(text: str, start: int, end: int) -> int:
    while end > start and text[end - 1].isspace():
        end -= 1
    return end


def _trim_left(text: str, start: int, limit: int) -> int:
    while start < limit and text[start].isspace():
        start += 1
    return start


def chunk_document(source: str, text: str, size: int = 320, overlap: int = 64,
                   title: str = "") -> Document:
    # 先还原硬折行，再切块：Document.text 保存归一化后的文本，
    # 这样块的字符区间、Citation.verify 的校验、页面上高亮的原文三者才是同一份。
    text = normalize_text(text)
    doc_id = doc_id_for(source)
    if not title:
        first_heading = _headings(text)
        if first_heading:
            title = first_heading[0][1]
    chunks: list[Chunk] = []
    for ordinal, (start, end, section) in enumerate(chunk_text(text, size, overlap)):
        body = text[start:end]
        chunks.append(Chunk(
            id=chunk_id_for(doc_id, ordinal, body),
            doc_id=doc_id,
            source=source,
            title=title,
            section=section,
            ordinal=ordinal,
            text=body,
            start=start,
            end=end,
        ))
    return Document(id=doc_id, source=source, title=title, text=text, chunks=chunks)
