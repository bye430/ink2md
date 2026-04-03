"""Document layout detection using DocLayout-YOLO."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class LayoutBlock:
    """A detected region in a document page."""
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) in pixels
    category: str
    confidence: float
    page_num: int = 0

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return (x1 - x0) * (y1 - y0)

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2


CATEGORY_MAP = {
    0: "title",
    1: "plain_text",
    2: "abandon",
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}


class LayoutDetector:
    """Detect document layout regions using DocLayout-YOLO."""

    def __init__(
        self,
        model_path: str = "models/doclayout_yolo/doclayout_yolo_docstructbench_imgsz1024.pt",
        conf_threshold: float = 0.25,
        device: str | None = None,
    ):
        from doclayout_yolo import YOLOv10

        self.model = YOLOv10(model_path)
        self.conf_threshold = conf_threshold
        self.device = device or "cuda"

    def detect(self, image: Image.Image | np.ndarray, page_num: int = 0) -> list[LayoutBlock]:
        """Detect layout blocks in a page image."""
        results = self.model.predict(
            image,
            imgsz=1024,
            conf=self.conf_threshold,
            device=self.device,
        )

        blocks = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                category = CATEGORY_MAP.get(cls_id, f"unknown_{cls_id}")
                if category == "abandon":
                    continue
                block = LayoutBlock(
                    bbox=tuple(boxes.xyxy[i].cpu().numpy().tolist()),
                    category=category,
                    confidence=float(boxes.conf[i].item()),
                    page_num=page_num,
                )
                blocks.append(block)

        blocks = self._deduplicate(blocks)
        blocks = self._sort_reading_order(blocks)
        return blocks

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        x0 = max(a[0], b[0])
        y0 = max(a[1], b[1])
        x1 = min(a[2], b[2])
        y1 = min(a[3], b[3])
        inter = max(0, x1 - x0) * max(0, y1 - y0)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    @classmethod
    def _deduplicate(cls, blocks: list[LayoutBlock], iou_threshold: float = 0.8) -> list[LayoutBlock]:
        """Remove near-duplicate blocks (same category, high IoU), keeping the higher-confidence one."""
        if not blocks:
            return blocks
        keep: list[LayoutBlock] = []
        for block in blocks:
            is_dup = False
            for j, kept in enumerate(keep):
                if block.category == kept.category and cls._iou(block.bbox, kept.bbox) > iou_threshold:
                    if block.confidence > kept.confidence:
                        keep[j] = block
                    is_dup = True
                    break
            if not is_dup:
                keep.append(block)
        return keep

    @staticmethod
    def _sort_reading_order(blocks: list[LayoutBlock]) -> list[LayoutBlock]:
        """Sort blocks in reading order.

        - Single-column pages: top-to-bottom, left-to-right.
        - Two-column pages: read left column first, then right column.
        """
        if not blocks:
            return blocks

        page_width = max(b.bbox[2] for b in blocks)
        page_height = max(b.bbox[3] for b in blocks)
        mid_x = page_width / 2.0

        def row_sort(candidates: list[LayoutBlock]) -> list[LayoutBlock]:
            row_threshold = page_height * 0.015
            sorted_blocks = sorted(candidates, key=lambda b: (b.bbox[1], b.bbox[0]))
            rows: list[list[LayoutBlock]] = []
            current_row: list[LayoutBlock] = [sorted_blocks[0]]
            for block in sorted_blocks[1:]:
                if abs(block.bbox[1] - current_row[0].bbox[1]) < row_threshold:
                    current_row.append(block)
                else:
                    rows.append(sorted(current_row, key=lambda b: b.bbox[0]))
                    current_row = [block]
            rows.append(sorted(current_row, key=lambda b: b.bbox[0]))
            result: list[LayoutBlock] = []
            for row in rows:
                result.extend(row)
            return result

        def is_full_width(block: LayoutBlock) -> bool:
            x0, _, x1, _ = block.bbox
            width = x1 - x0
            return width > page_width * 0.6

        full_width = [b for b in blocks if is_full_width(b)]
        column_blocks = [b for b in blocks if not is_full_width(b)]

        def column_of(block: LayoutBlock) -> str:
            """Assign left vs right column. Narrow boxes (equation numbers, etc.) often
            sit in the gutter and cross the midline; use a relaxed threshold only for them.
            """
            x0, _, x1, _ = block.bbox
            cx = (x0 + x1) / 2.0
            w = x1 - x0
            if w < page_width * 0.12:
                return "left" if cx < mid_x + page_width * 0.05 else "right"
            return "left" if cx < mid_x else "right"

        left = [b for b in column_blocks if column_of(b) == "left"]
        right = [b for b in column_blocks if column_of(b) == "right"]

        separation = 0.0
        if left and right:
            separation = min(b.bbox[0] for b in right) - max(b.bbox[2] for b in left)
        is_two_column = bool(left and right) and separation > page_width * 0.03
        if not is_two_column:
            return row_sort(blocks)

        left = sorted(left, key=lambda b: (b.bbox[1], b.bbox[0]))
        right = sorted(right, key=lambda b: (b.bbox[1], b.bbox[0]))
        full_width = sorted(full_width, key=lambda b: (b.bbox[1], b.bbox[0]))

        # Segment by full-width blocks to keep their vertical position.
        result: list[LayoutBlock] = []
        prev_y = float("-inf")
        for fw in full_width:
            left_seg = [b for b in left if prev_y <= b.bbox[1] < fw.bbox[1]]
            right_seg = [b for b in right if prev_y <= b.bbox[1] < fw.bbox[1]]
            result.extend(left_seg)
            result.extend(right_seg)
            result.append(fw)
            prev_y = fw.bbox[1]

        left_tail = [b for b in left if b.bbox[1] >= prev_y]
        right_tail = [b for b in right if b.bbox[1] >= prev_y]
        result.extend(left_tail)
        result.extend(right_tail)
        return result
