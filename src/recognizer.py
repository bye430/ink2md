"""UniMERNet-based LaTeX formula recognizer."""

from pathlib import Path

import torch
from PIL import Image
from omegaconf import OmegaConf

from unimernet.models.unimernet.unimernet import UniMERModel
from unimernet.processors import load_processor


class FormulaRecognizer:
    """Recognize LaTeX formulas from images using UniMERNet."""

    def __init__(
        self,
        model_dir: str = "models/unimernet_tiny",
        weight_name: str = "unimernet_tiny.pth",
        device: str | None = None,
        max_seq_len: int = 1536,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        model_dir = str(Path(model_dir).resolve())
        weight_path = str(Path(model_dir) / weight_name)

        model_cfg = OmegaConf.create({
            "model_name": "unimernet",
            "model_config": {
                "model_name": model_dir,
                "max_seq_len": max_seq_len,
            },
            "tokenizer_name": "nougat",
            "tokenizer_config": {"path": model_dir},
            "load_pretrained": True,
            "pretrained": weight_path,
            "load_finetuned": False,
            "finetuned": "",
        })

        self.model = UniMERModel.from_config(model_cfg)
        self.model.to(self.device)
        self.model.eval()

        vis_cfg = OmegaConf.create({
            "name": "formula_image_eval",
            "image_size": [192, 672],
        })
        self.vis_processor = load_processor("formula_image_eval", vis_cfg)

    def recognize(self, image: Image.Image) -> str:
        """Recognize LaTeX from a single PIL Image."""
        processed = self.vis_processor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model.generate({"image": processed})
        return output["pred_str"][0]

    def recognize_file(self, image_path: str | Path) -> str:
        """Recognize LaTeX from an image file path."""
        image = Image.open(image_path).convert("RGB")
        return self.recognize(image)

    def recognize_batch(self, images: list[Image.Image]) -> list[str]:
        """Recognize LaTeX from a batch of PIL Images."""
        processed = torch.stack(
            [self.vis_processor(img) for img in images]
        ).to(self.device)
        with torch.no_grad():
            output = self.model.generate({"image": processed})
        return output["pred_str"]
