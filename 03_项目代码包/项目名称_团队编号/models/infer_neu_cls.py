"""NEU-CLS 单张图片推理脚本。

运行：
    python models/infer_neu_cls.py --image data/raw/NEU-CLS/valid/valid/images/crazing_1.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from PIL import Image

try:
    from .neu_cls_model import DefectCNN
except ImportError:
    from neu_cls_model import DefectCNN


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


CLASS_NAMES_ZH = {
    "crazing": "龟裂",
    "inclusion": "夹杂",
    "patches": "斑块",
    "pitted_surface": "麻点表面",
    "rolled-in_scale": "轧入氧化皮",
    "scratches": "划痕",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_image(image_source: Path | str | BinaryIO | Image.Image, image_size: int) -> torch.Tensor:
    if isinstance(image_source, Image.Image):
        image = image_source.convert("L")
    else:
        image = Image.open(image_source).convert("L")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[DefectCNN, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    model = DefectCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@lru_cache(maxsize=4)
def load_model_cached(checkpoint_path: str, device_name: str) -> tuple[DefectCNN, dict]:
    return load_model(Path(checkpoint_path), torch.device(device_name))


@torch.no_grad()
def predict_image(
    image_source: Path | str | BinaryIO | Image.Image,
    checkpoint_path: Path | None = None,
    top_k: int = 3,
    device: torch.device | None = None,
) -> dict:
    root = project_root()
    if checkpoint_path is None:
        checkpoint_path = root / "models" / "artifacts" / "neu_cls_cnn" / "best_model.pt"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint = load_model_cached(str(checkpoint_path), str(device))
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint["image_size"])

    image_tensor = load_image(image_source, image_size).to(device)
    start_time = time.perf_counter()
    logits = model(image_tensor)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    top_indices = probabilities.argsort()[::-1][:top_k]

    predicted_class = class_names[int(top_indices[0])]
    result = {
        "predicted_class": predicted_class,
        "predicted_class_zh": CLASS_NAMES_ZH.get(predicted_class, predicted_class),
        "confidence": round(float(probabilities[int(top_indices[0])]), 6),
        "inference_time_ms": round(elapsed_ms, 3),
        "top_k": [
            {
                "class_name": class_names[int(index)],
                "class_name_zh": CLASS_NAMES_ZH.get(class_names[int(index)], class_names[int(index)]),
                "confidence": round(float(probabilities[int(index)]), 6),
            }
            for index in top_indices
        ],
    }
    return result


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="钢材表面缺陷单图推理")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "models" / "artifacts" / "neu_cls_cnn" / "best_model.pt",
    )
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    result = predict_image(args.image, checkpoint_path=args.checkpoint, top_k=args.top_k)
    result["image"] = str(args.image)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
