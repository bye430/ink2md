"""Assemble detected layout blocks into Markdown output."""

import re
from pathlib import Path

from PIL import Image

from .layout_detector import LayoutBlock


class MarkdownAssembler:
    """Convert layout blocks with extracted content into Markdown."""

    def __init__(self, image_dir: str = "images", image_rel_dir: str = "images"):
        self.image_dir = Path(image_dir)
        self.image_rel_dir = image_rel_dir
        self._image_counter = 0

    def assemble(
        self,
        blocks: list[LayoutBlock],
        contents: dict[int, str],
        images: dict[int, Image.Image] | None = None,
    ) -> str:
        """Assemble blocks into a Markdown string."""
        images = images or {}

        merged = self._merge_title_blocks(blocks, contents)
        self._merge_adjacent_plain_text(merged, contents)

        parts: list[str] = []
        for i, block in enumerate(merged):
            content = contents.get(i, "").strip()
            if not content and i not in images:
                continue

            md = self._block_to_markdown(block, content, i, images)
            if md:
                parts.append(md)

        return "\n\n".join(parts) + "\n"

    def _merge_title_blocks(
        self,
        blocks: list[LayoutBlock],
        contents: dict[int, str],
    ) -> list[LayoutBlock]:
        """Merge a number-only title with the immediately following text block.

        DocLayout-YOLO sometimes splits "2 Background" into a title block
        containing just "2" and a plain_text block with "Background".  This
        method detects that pattern and folds the second block into the title.
        """
        skip: set[int] = set()
        for i, block in enumerate(blocks):
            if block.category != "title":
                continue
            text = contents.get(i, "").strip()
            if not re.fullmatch(r"[\d.]+", text):
                continue
            j = i + 1
            if j >= len(blocks):
                continue
            next_block = blocks[j]
            next_text = contents.get(j, "").strip()
            if not next_text:
                continue
            if next_block.category in ("plain_text", "title"):
                gap = next_block.bbox[1] - block.bbox[3]
                if gap < (block.bbox[3] - block.bbox[1]) * 1.5:
                    contents[i] = f"{text} {next_text}"
                    contents[j] = ""
                    skip.add(j)

        return blocks

    def _merge_adjacent_plain_text(
        self,
        blocks: list[LayoutBlock],
        contents: dict[int, str],
    ) -> None:
        """Join plain_text blocks that are one sentence split across columns.

        When the first fragment does not end a sentence and the next starts with
        a lowercase letter (or continues a hyphenated word), merge into the first.
        Chains multiple fragments (A+B+C) in one pass.
        """
        i = 0
        while i < len(blocks):
            if blocks[i].category != "plain_text":
                i += 1
                continue
            a = contents.get(i, "").strip()
            if not a:
                i += 1
                continue
            j = i + 1
            while j < len(blocks):
                if blocks[j].category != "plain_text":
                    break
                b = contents.get(j, "").strip()
                if not b:
                    j += 1
                    continue
                if not self._should_merge_plain_text_fragments(a, b):
                    break
                a = self._join_plain_text_fragments(a, b)
                contents[i] = a
                contents[j] = ""
                j += 1
            i += 1

    @staticmethod
    def _should_merge_plain_text_fragments(a: str, b: str) -> bool:
        a = a.rstrip()
        b = b.lstrip()
        if not a or not b:
            return False
        if a.endswith("-"):
            return True
        if a[-1] in ".!?":
            return False
        return b[0].islower()

    @staticmethod
    def _join_plain_text_fragments(a: str, b: str) -> str:
        a = a.rstrip()
        b = b.lstrip()
        if a.endswith("-"):
            return (a[:-1].rstrip() + b).strip()
        return (a + " " + b).strip()

    def _block_to_markdown(
        self,
        block: LayoutBlock,
        content: str,
        idx: int,
        images: dict[int, Image.Image],
    ) -> str:
        cat = block.category

        if cat == "title":
            content = " ".join(content.split())
            level = self._guess_heading_level(block, content)
            return f"{'#' * level} {content}"

        if cat == "plain_text":
            return content

        if cat == "isolate_formula":
            return f"$$\n{content}\n$$"

        if cat == "figure":
            return self._handle_figure(idx, images)

        if cat == "figure_caption":
            return f"*{content}*"

        if cat == "table":
            if content.startswith("|"):
                return content
            return f"```\n{content}\n```"

        if cat == "table_caption":
            return f"*{content}*"

        if cat == "table_footnote":
            return f"> {content}"

        if cat == "formula_caption":
            return f"*{content}*"

        return content

    @staticmethod
    def _guess_heading_level(block: LayoutBlock, content: str) -> int:
        _, y0, _, y1 = block.bbox
        height = y1 - y0
        if height > 40:
            return 1
        if height > 25:
            return 2
        return 3

    def _handle_figure(self, idx: int, images: dict[int, Image.Image]) -> str:
        if idx not in images:
            return "![figure]()"

        self._image_counter += 1
        filename = f"fig_{self._image_counter:03d}.png"

        img = images[idx]
        save_path = self.image_dir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(save_path))

        rel_path = f"{self.image_rel_dir}/{filename}"
        return f"![figure]({rel_path})"
