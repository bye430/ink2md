"""Table recognition using StructTable-InternVL2-1B."""

import gc

import torch
from PIL import Image


class TableRecognizer:
    """Recognize table structure and convert to Markdown using StructTable."""

    def __init__(
        self,
        model_path: str = "models/struct_table",
        device: str | None = None,
        output_format: str = "markdown",
        max_new_tokens: int = 2048,
        lazy_load: bool = True,
    ):
        self.model_path = str(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_format = output_format
        self.max_new_tokens = max_new_tokens
        self._model = None
        if not lazy_load:
            self._load_model()

    def _load_model(self):
        if self._model is not None:
            return
        from struct_eqtable import build_model

        self._model = build_model(
            model_ckpt=self.model_path,
            max_new_tokens=self.max_new_tokens,
            flash_attn=False,
        )
        self._model.to(self.device)

    def unload(self):
        """Release GPU memory."""
        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def recognize(self, image: Image.Image) -> str:
        """Recognize a table image and return Markdown (or other format)."""
        self._load_model()
        image = image.convert("RGB")
        results = self._model([image], output_format=self.output_format)
        if results and results[0]:
            return results[0].strip()
        return ""

    def recognize_batch(self, images: list[Image.Image]) -> list[str]:
        """Recognize multiple table images."""
        if not images:
            return []
        self._load_model()
        rgb_images = [img.convert("RGB") for img in images]
        results = self._model(rgb_images, output_format=self.output_format)
        return [r.strip() if r else "" for r in results]
