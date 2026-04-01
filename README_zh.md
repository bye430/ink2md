# Ink2MD — 论文 PDF 转 Markdown

Ink2MD 将学术论文 PDF 转换为结构化 Markdown，保留 LaTeX 公式、表格、图片和阅读顺序。

## 功能

- **PDF → Markdown**：端到端转换，自动识别文档版面布局
- **公式识别**：独立公式（display math）和行内公式（inline math）均转为 LaTeX
- **表格识别**：复杂表格结构还原为 Markdown 表格，支持表内 LaTeX 公式
- **图片提取**：自动裁切并保存为独立图片文件
- **单图识别**：支持对单张公式截图直接输出 LaTeX

## 技术栈

| 模块 | 模型 | 用途 |
|------|------|------|
| 版面分析 | [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) | 检测标题、正文、公式、表格、图片等区域 |
| 公式检测 | [YOLOv8-MFD](https://github.com/opendatalab/PDF-Extract-Kit) | 定位行内公式和独立公式 |
| 公式识别 | [UniMERNet](https://github.com/opendatalab/UniMERNet) | 图片 → LaTeX |
| 表格识别 | [StructTable-InternVL2-1B](https://github.com/UniModal4Reasoning/StructEqTable-Deploy) | 表格图片 → Markdown/LaTeX |
| PDF 渲染 | [PyMuPDF](https://pymupdf.readthedocs.io/) | 页面渲染与文本提取 |

## 环境要求

- Python 3.10+
- CUDA 12.1

## 安装

```bash
conda create -n Ink2MD python=3.10 -y
conda activate Ink2MD

# PyTorch（根据 CUDA 版本选择）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 其他依赖
pip install -r requirements.txt
```

## 模型下载

需要下载 4 个模型到 `models/` 目录：

```bash
# 1. UniMERNet（公式识别，~500MB）
huggingface-cli download wanderkid/unimernet_tiny --local-dir models/unimernet_tiny

# 2. DocLayout-YOLO（版面分析，~30MB）
huggingface-cli download juliozhao/DocLayout-YOLO-DocStructBench \
  doclayout_yolo_docstructbench_imgsz1024.pt \
  --local-dir models/doclayout_yolo

# 3. YOLOv8-MFD（公式检测，~50MB）
huggingface-cli download wanderkid/yolov8-mfd yolo_v8_ft.pt --local-dir models/mfd

# 4. StructTable-InternVL2-1B（表格识别，~1.8GB）
huggingface-cli download U4R/StructTable-InternVL2-1B --local-dir models/struct_table
```

> 国内用户可设置 HuggingFace 镜像加速：`export HF_ENDPOINT=https://hf-mirror.com`

下载完成后目录结构：

```
models/
├── unimernet_tiny/        # 公式识别模型
├── doclayout_yolo/        # 版面分析模型
├── mfd/                   # 公式检测模型 (yolo_v8_ft.pt)
└── struct_table/          # 表格识别模型
```

## 使用

### PDF 转 Markdown

```bash
# 转换整篇论文
python -m src.cli convert paper.pdf

# 指定输出目录
python -m src.cli convert paper.pdf -o output/

# 只转换指定页
python -m src.cli convert paper.pdf -p 1-5

# 使用 CPU
python -m src.cli convert paper.pdf -d cpu
```

输出结构：

```
output/
├── paper.md              # Markdown 文件
└── images/               # 提取的图片
    ├── page1_fig0.png
    └── ...
```

### 单张公式图片识别

```bash
python -m src.cli recognize formula.png
```

### 批量公式识别

```bash
python -m src.cli batch images_dir/ -o results.txt
```

## 命令参数

### `convert`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output, -o` | PDF 同级目录 | 输出目录 |
| `--pages, -p` | 全部 | 页码范围，如 `1-5` 或 `3` |
| `--dpi` | 216 | 渲染 DPI，越高越清晰但越慢 |
| `--device, -d` | cuda | 推理设备 |
| `--layout-conf` | 0.25 | 版面检测置信度阈值 |

### `recognize`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-dir, -m` | models/unimernet_tiny | 模型目录 |
| `--device, -d` | cuda | 推理设备 |

## 项目结构

```
src/
├── cli.py                    # 命令行入口
├── pipeline.py               # PDF→Markdown 主流程
├── layout_detector.py        # DocLayout-YOLO 版面分析
├── inline_formula_detector.py # YOLOv8 行内公式检测
├── recognizer.py             # UniMERNet 公式识别
├── table_recognizer.py       # StructTable 表格识别
├── pdf_renderer.py           # PDF 页面渲染与文本提取
└── md_assembler.py           # Markdown 组装输出
```

## 许可

本项目代码以 MIT 许可证发布。所使用的模型各有独立许可，请参考对应仓库。
