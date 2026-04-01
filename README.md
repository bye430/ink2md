# Ink2MD — Convert Research PDFs to Markdown

Ink2MD converts academic PDFs into structured Markdown while preserving LaTeX formulas, tables, images, and reading order.

## Features

- **PDF → Markdown**: End-to-end conversion with automatic document layout parsing.
- **Formula recognition**: Converts both display and inline formulas to LaTeX.
- **Table recognition**: Reconstructs complex tables as Markdown, including LaTeX inside cells.
- **Figure extraction**: Crops and saves figures as standalone image files.
- **Single-image OCR**: Recognizes LaTeX directly from formula screenshots.

## Tech Stack

| Module | Model | Purpose |
|------|------|------|
| Layout analysis | [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) | Detects title, text, formulas, tables, figures, etc. |
| Formula detection | [YOLOv8-MFD](https://github.com/opendatalab/PDF-Extract-Kit) | Locates inline and isolated formulas |
| Formula recognition | [UniMERNet](https://github.com/opendatalab/UniMERNet) | Image → LaTeX |
| Table recognition | [StructTable-InternVL2-1B](https://github.com/UniModal4Reasoning/StructEqTable-Deploy) | Table image → Markdown/LaTeX |
| PDF rendering | [PyMuPDF](https://pymupdf.readthedocs.io/) | Page rendering and text extraction |

## Requirements

- Python 3.10+
- CUDA 12.1

## Installation

```bash
conda create -n Ink2MD python=3.10 -y
conda activate Ink2MD

# Install PyTorch (choose by your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

## Model Download

Download 4 models into the `models/` directory:

```bash
# 1. UniMERNet (~500MB)
huggingface-cli download wanderkid/unimernet_tiny --local-dir models/unimernet_tiny

# 2. DocLayout-YOLO (~30MB)
huggingface-cli download juliozhao/DocLayout-YOLO-DocStructBench \
  doclayout_yolo_docstructbench_imgsz1024.pt \
  --local-dir models/doclayout_yolo

# 3. YOLOv8-MFD (~50MB)
huggingface-cli download wanderkid/yolov8-mfd yolo_v8_ft.pt --local-dir models/mfd

# 4. StructTable-InternVL2-1B (~1.8GB)
huggingface-cli download U4R/StructTable-InternVL2-1B --local-dir models/struct_table
```

> For faster downloads in some regions: `export HF_ENDPOINT=https://hf-mirror.com`

Expected directory structure:

```text
models/
├── unimernet_tiny/        # Formula recognition model
├── doclayout_yolo/        # Layout detection model
├── mfd/                   # Formula detector (yolo_v8_ft.pt)
└── struct_table/          # Table recognition model
```

## Usage

### Convert PDF to Markdown

```bash
# Convert a full paper
python -m src.cli convert paper.pdf

# Specify output directory
python -m src.cli convert paper.pdf -o output/

# Convert specific pages
python -m src.cli convert paper.pdf -p 1-5

# Run on CPU
python -m src.cli convert paper.pdf -d cpu
```

Output layout:

```text
output/
├── paper.md              # Markdown file
└── images/               # Extracted figures
    ├── page1_fig0.png
    └── ...
```

### Recognize a Single Formula Image

```bash
python -m src.cli recognize formula.png
```

### Batch Formula Recognition

```bash
python -m src.cli batch images_dir/ -o results.txt
```

## CLI Arguments

### `convert`

| Argument | Default | Description |
|------|--------|------|
| `--output, -o` | PDF sibling directory | Output directory |
| `--pages, -p` | all pages | Page range, e.g. `1-5` or `3` |
| `--dpi` | 216 | Rendering DPI (higher is clearer but slower) |
| `--device, -d` | cuda | Inference device |
| `--layout-conf` | 0.25 | Layout detection confidence threshold |

### `recognize`

| Argument | Default | Description |
|------|--------|------|
| `--model-dir, -m` | models/unimernet_tiny | Model directory |
| `--device, -d` | cuda | Inference device |

## Project Structure

```text
src/
├── cli.py                     # CLI entrypoint
├── pipeline.py                # Main PDF→Markdown pipeline
├── layout_detector.py         # DocLayout-YOLO wrapper
├── inline_formula_detector.py # YOLOv8 inline formula detection
├── recognizer.py              # UniMERNet formula recognition
├── table_recognizer.py        # StructTable recognition
├── pdf_renderer.py            # PDF rendering and text extraction
└── md_assembler.py            # Markdown assembly
```

## License

This project code is released under the MIT License. Model licenses are independent; check each model repository for details.
