"""Inline formula detection using YOLOv8 from PDF-Extract-Kit."""

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class FormulaRegion:
    """A detected formula region."""
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) in pixels
    category: str  # "inline" or "isolated"
    confidence: float


class InlineFormulaDetector:
    """Detect inline and isolated formulas using YOLOv8 MFD model."""

    def __init__(
        self,
        model_path: str = "models/mfd/yolo_v8_ft.pt",
        conf_threshold: float = 0.25,
        device: str | None = None,
    ):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device or "cuda"

    def detect(self, image: Image.Image | np.ndarray) -> list[FormulaRegion]:
        """Detect all formula regions (inline + isolated) in a page image."""
        results = self.model.predict(
            image,
            imgsz=1888,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        regions = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                cat = "inline" if cls_id == 0 else "isolated"
                regions.append(FormulaRegion(
                    bbox=tuple(boxes.xyxy[i].cpu().numpy().tolist()),
                    category=cat,
                    confidence=float(boxes.conf[i].item()),
                ))
        return regions

    def find_inline_in_bbox(
        self,
        all_regions: list[FormulaRegion],
        text_bbox: tuple[float, float, float, float],
        overlap_threshold: float = 0.5,
    ) -> list[FormulaRegion]:
        """Find inline formula regions that fall within a text block bbox."""
        tx0, ty0, tx1, ty1 = text_bbox
        results = []
        for r in all_regions:
            if r.category != "inline":
                continue
            rx0, ry0, rx1, ry1 = r.bbox
            inter_x0 = max(tx0, rx0)
            inter_y0 = max(ty0, ry0)
            inter_x1 = min(tx1, rx1)
            inter_y1 = min(ty1, ry1)
            if inter_x0 >= inter_x1 or inter_y0 >= inter_y1:
                continue
            inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
            formula_area = (rx1 - rx0) * (ry1 - ry0)
            if formula_area > 0 and inter_area / formula_area >= overlap_threshold:
                results.append(r)
        return sorted(results, key=lambda r: (r.bbox[1], r.bbox[0]))
