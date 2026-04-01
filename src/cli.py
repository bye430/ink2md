"""Command-line interface for Ink2MD."""

from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

console = Console()


@click.group()
def cli():
    """Ink2MD - Recognize formulas and convert PDFs to Markdown."""
    pass


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--model-dir", "-m", default="models/unimernet_tiny", help="Model directory.")
@click.option("--device", "-d", default=None, help="Device: cuda / cpu.")
def recognize(image_path: str, model_dir: str, device: str | None):
    """Recognize LaTeX formula from an image file."""
    from .recognizer import FormulaRecognizer

    with console.status("[bold green]Loading model..."):
        recognizer = FormulaRecognizer(model_dir=model_dir, device=device)

    with console.status("[bold green]Recognizing..."):
        result = recognizer.recognize_file(image_path)

    console.print(Panel(result, title="LaTeX Output", border_style="green"))


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output directory (default: same as PDF).")
@click.option("--pages", "-p", default=None, help="Page range, e.g. '1-5' or '3'.")
@click.option("--dpi", default=216, help="Rendering DPI.")
@click.option("--device", "-d", default=None, help="Device: cuda / cpu.")
@click.option("--layout-conf", default=0.25, help="Layout detection confidence threshold.")
def convert(
    pdf_path: str,
    output: str | None,
    pages: str | None,
    dpi: int,
    device: str | None,
    layout_conf: float,
):
    """Convert a PDF document to Markdown with LaTeX formulas."""
    from .pipeline import PDF2MarkdownPipeline

    pdf = Path(pdf_path)
    if output is None:
        output_dir = pdf.parent / pdf.stem
    else:
        output_dir = Path(output)

    page_range = None
    if pages:
        if "-" in pages:
            s, e = pages.split("-", 1)
            page_range = (int(s) - 1, int(e))
        else:
            p = int(pages) - 1
            page_range = (p, p + 1)

    pipeline = PDF2MarkdownPipeline(
        dpi=dpi,
        device=device,
        layout_conf=layout_conf,
    )

    pipeline.convert(pdf, output_dir, page_range=page_range)


@cli.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option("--model-dir", "-m", default="models/unimernet_tiny", help="Model directory.")
@click.option("--device", "-d", default=None, help="Device: cuda / cpu.")
@click.option("--output", "-o", default="output/results.txt", help="Output file.")
def batch(directory: str, model_dir: str, device: str | None, output: str):
    """Recognize LaTeX formulas from all images in a directory."""
    from .recognizer import FormulaRecognizer

    image_dir = Path(directory)
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    image_files = sorted(
        f for f in image_dir.iterdir() if f.suffix.lower() in suffixes
    )

    if not image_files:
        console.print("[red]No image files found in directory.")
        return

    console.print(f"Found [bold]{len(image_files)}[/bold] images.")

    with console.status("[bold green]Loading model..."):
        recognizer = FormulaRecognizer(model_dir=model_dir, device=device)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, img_file in enumerate(image_files, 1):
        console.print(f"[{i}/{len(image_files)}] {img_file.name}...", end=" ")
        latex = recognizer.recognize_file(img_file)
        results.append(f"{img_file.name}\t{latex}")
        console.print("[green]OK[/green]")

    output_path.write_text("\n".join(results), encoding="utf-8")
    console.print(f"\nResults saved to [bold]{output_path}[/bold]")


if __name__ == "__main__":
    cli()
