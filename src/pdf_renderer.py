"""PDF page rendering and text/image extraction using PyMuPDF."""

from dataclasses import dataclass
from pathlib import Path

import fitz
import numpy as np
from PIL import Image


@dataclass
class PDFPage:
    """Rendered PDF page with metadata."""
    image: Image.Image
    page_num: int
    width: float
    height: float
    dpi: int


class PDFRenderer:
    """Render PDF pages to images and extract embedded content."""

    def __init__(self, dpi: int = 216):
        self.dpi = dpi

    def open(self, pdf_path: str | Path) -> fitz.Document:
        return fitz.open(str(pdf_path))

    def render_page(self, page: fitz.Page, page_num: int = 0) -> PDFPage:
        """Render a single PDF page to an image."""
        zoom = self.dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return PDFPage(
            image=img,
            page_num=page_num,
            width=pix.width,
            height=pix.height,
            dpi=self.dpi,
        )

    def extract_text_in_bbox(
        self, page: fitz.Page, bbox: tuple[float, float, float, float]
    ) -> str:
        """Extract text from a specific region of a PDF page.

        bbox is in rendered image coordinates (at self.dpi).
        We convert back to PDF coordinates (72 dpi).
        """
        scale = 72.0 / self.dpi
        pdf_rect = fitz.Rect(
            bbox[0] * scale,
            bbox[1] * scale,
            bbox[2] * scale,
            bbox[3] * scale,
        )
        return page.get_text("text", clip=pdf_rect).strip()

    def extract_image_in_bbox(
        self, page_image: Image.Image, bbox: tuple[float, float, float, float]
    ) -> Image.Image:
        """Crop a region from the rendered page image."""
        x0, y0, x1, y1 = [int(v) for v in bbox]
        return page_image.crop((x0, y0, x1, y1))

    def extract_table_text(
        self, page: fitz.Page, bbox: tuple[float, float, float, float]
    ) -> str:
        """Extract table content as text from a region.

        Uses PyMuPDF's built-in table finder when available,
        falls back to raw text extraction.
        """
        scale = 72.0 / self.dpi
        pdf_rect = fitz.Rect(
            bbox[0] * scale,
            bbox[1] * scale,
            bbox[2] * scale,
            bbox[3] * scale,
        )

        try:
            tabs = page.find_tables(clip=pdf_rect)
            if tabs and tabs.tables:
                table = tabs.tables[0]
                return self._table_to_markdown(table)
        except Exception:
            pass

        return page.get_text("text", clip=pdf_rect).strip()

    @staticmethod
    def _table_to_markdown(table) -> str:
        """Convert a PyMuPDF table to Markdown format."""
        data = table.extract()
        if not data:
            return ""

        def clean_cell(cell):
            if cell is None:
                return ""
            return str(cell).replace("\n", " ").strip()

        lines = []
        header = [clean_cell(c) for c in data[0]]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        for row in data[1:]:
            cells = [clean_cell(c) for c in row]
            while len(cells) < len(header):
                cells.append("")
            lines.append("| " + " | ".join(cells[:len(header)]) + " |")

        return "\n".join(lines)
