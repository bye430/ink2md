"""End-to-end PDF to Markdown conversion pipeline."""

import gc
import re
from pathlib import Path

import fitz
import torch
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .layout_detector import LayoutDetector, LayoutBlock
from .inline_formula_detector import InlineFormulaDetector, FormulaRegion
from .pdf_renderer import PDFRenderer, normalize_flow_text
from .recognizer import FormulaRecognizer
from .table_recognizer import TableRecognizer
from .md_assembler import MarkdownAssembler

console = Console()


class PDF2MarkdownPipeline:
    """Convert a PDF document to Markdown with LaTeX formulas."""

    def __init__(
        self,
        layout_model_path: str = "models/doclayout_yolo/doclayout_yolo_docstructbench_imgsz1024.pt",
        mfd_model_path: str = "models/mfd/yolo_v8_ft.pt",
        formula_model_dir: str = "models/unimernet_tiny",
        formula_weight_name: str = "unimernet_tiny.pth",
        table_model_path: str = "models/struct_table",
        dpi: int = 216,
        device: str | None = None,
        layout_conf: float = 0.25,
        mfd_conf: float = 0.25,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dpi = dpi

        console.print("[bold]Loading layout detector...[/bold]")
        self.layout_detector = LayoutDetector(
            model_path=layout_model_path,
            conf_threshold=layout_conf,
            device=self.device,
        )

        console.print("[bold]Loading formula detector (MFD)...[/bold]")
        self.inline_detector = InlineFormulaDetector(
            model_path=mfd_model_path,
            conf_threshold=mfd_conf,
            device=self.device,
        )

        console.print("[bold]Loading formula recognizer...[/bold]")
        self.formula_recognizer = FormulaRecognizer(
            model_dir=formula_model_dir,
            weight_name=formula_weight_name,
            device=self.device,
        )

        console.print("[bold]Loading table recognizer (lazy)...[/bold]")
        self.table_recognizer = TableRecognizer(
            model_path=table_model_path,
            device=self.device,
            lazy_load=True,
        )

        self.pdf_renderer = PDFRenderer(dpi=dpi)

    def convert(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
        page_range: tuple[int, int] | None = None,
    ) -> str:
        """Convert a PDF file to Markdown."""
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = output_dir / "images"
        image_dir.mkdir(exist_ok=True)

        doc = self.pdf_renderer.open(pdf_path)
        total_pages = len(doc)

        start = 0
        end = total_pages
        if page_range:
            start, end = page_range
            start = max(0, start)
            end = min(total_pages, end)

        console.print(
            f"Processing [bold]{pdf_path.name}[/bold]: "
            f"pages {start+1}-{end} of {total_pages}"
        )

        all_md_parts: list[str] = []
        assembler = MarkdownAssembler(
            image_dir=str(image_dir),
            image_rel_dir="images",
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Converting pages...", total=end - start)

            for page_idx in range(start, end):
                page = doc[page_idx]
                page_md = self._process_page(page, page_idx, image_dir, assembler)
                all_md_parts.append(page_md)
                progress.advance(task)

                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()

        doc.close()

        full_md = "\n\n---\n\n".join(all_md_parts)

        md_path = output_dir / f"{pdf_path.stem}.md"
        md_path.write_text(full_md, encoding="utf-8")
        console.print(f"[green]Saved:[/green] {md_path}")

        return full_md

    def _process_page(self, page: fitz.Page, page_idx: int, image_dir: Path, assembler: MarkdownAssembler) -> str:
        """Process a single PDF page."""
        rendered = self.pdf_renderer.render_page(page, page_idx)

        blocks = self.layout_detector.detect(rendered.image, page_idx)
        formula_regions = self.inline_detector.detect(rendered.image)

        contents: dict[int, str] = {}
        images: dict[int, Image.Image] = {}

        isolated_formula_indices: list[int] = []
        isolated_formula_images: list[Image.Image] = []
        inline_formula_images: list[Image.Image] = []
        inline_formula_regions: list[FormulaRegion] = []
        table_images_map: dict[int, Image.Image] = {}

        for r in formula_regions:
            if r.category == "inline":
                inline_formula_regions.append(r)
                cropped = self.pdf_renderer.extract_image_in_bbox(
                    rendered.image, r.bbox
                )
                inline_formula_images.append(cropped.convert("RGB"))

        inline_latex_map: dict[int, str] = {}
        if inline_formula_images:
            latex_results = self.formula_recognizer.recognize_batch(inline_formula_images)
            for i, latex in enumerate(latex_results):
                inline_latex_map[i] = latex

        for i, block in enumerate(blocks):
            cat = block.category

            if cat == "isolate_formula":
                isolated_formula_indices.append(i)
                cropped = self.pdf_renderer.extract_image_in_bbox(
                    rendered.image, block.bbox
                )
                isolated_formula_images.append(cropped)

            elif cat == "figure":
                cropped = self.pdf_renderer.extract_image_in_bbox(
                    rendered.image, block.bbox
                )
                images[i] = cropped

            elif cat == "table":
                cropped = self.pdf_renderer.extract_image_in_bbox(
                    rendered.image, block.bbox
                )
                table_images_map[i] = cropped

            elif cat in ("title", "plain_text", "figure_caption",
                         "table_caption", "table_footnote", "formula_caption"):
                raw_text = self.pdf_renderer.extract_text_in_bbox(page, block.bbox)

                if cat == "plain_text" and inline_formula_regions:
                    raw_text = self._replace_inline_formulas(
                        raw_text, block, page,
                        inline_formula_regions, inline_latex_map,
                    )

                contents[i] = normalize_flow_text(raw_text)

        if isolated_formula_images:
            rgb_images = [img.convert("RGB") for img in isolated_formula_images]
            latex_results = self.formula_recognizer.recognize_batch(rgb_images)
            for idx, latex in zip(isolated_formula_indices, latex_results):
                contents[idx] = latex

        if table_images_map:
            table_indices = sorted(table_images_map.keys())
            table_imgs = [table_images_map[k] for k in table_indices]
            table_results = self.table_recognizer.recognize_batch(table_imgs)
            for idx, md in zip(table_indices, table_results):
                if md.strip():
                    contents[idx] = md
                else:
                    contents[idx] = self.pdf_renderer.extract_table_text(
                        page, blocks[idx].bbox
                    )
            self.table_recognizer.unload()

        return assembler.assemble(blocks, contents, images)

    def _replace_inline_formulas(
        self,
        text: str,
        block: LayoutBlock,
        page: fitz.Page,
        all_inline_regions: list[FormulaRegion],
        inline_latex_map: dict[int, str],
    ) -> str:
        """Replace inline formula regions in text with $LaTeX$."""
        matching = self.inline_detector.find_inline_in_bbox(
            all_inline_regions, block.bbox
        )
        if not matching:
            return text

        scale = 72.0 / self.dpi

        for region in matching:
            region_idx = None
            for i, r in enumerate(all_inline_regions):
                if r is region:
                    region_idx = i
                    break
            if region_idx is None or region_idx not in inline_latex_map:
                continue

            latex = inline_latex_map[region_idx]

            pdf_rect = fitz.Rect(
                region.bbox[0] * scale,
                region.bbox[1] * scale,
                region.bbox[2] * scale,
                region.bbox[3] * scale,
            )
            original_text = page.get_text("text", clip=pdf_rect).strip()

            if original_text and original_text in text:
                text = text.replace(original_text, f"${latex}$", 1)
            elif original_text:
                normalized_orig = re.sub(r"\s+", " ", original_text)
                normalized_text = re.sub(r"\s+", " ", text)
                if normalized_orig in normalized_text:
                    text = re.sub(
                        re.escape(normalized_orig).replace(r"\ ", r"\s+"),
                        f"${latex}$",
                        text,
                        count=1,
                    )

        return text
