"""批量推理 NEU-CLS 图片并输出结果 CSV。

运行：
    python models/batch_infer_neu_cls.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from infer_neu_cls import predict_image


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="批量推理钢材表面缺陷图片")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "data" / "processed" / "neu_cls_classification" / "manifests" / "test.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "models" / "artifacts" / "neu_cls_cnn" / "batch_test_predictions.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            image_path = root / row["image_path"]
            prediction = predict_image(image_path)
            correct = prediction["predicted_class"] == row["class_name"]
            rows.append(
                {
                    "image_path": row["image_path"],
                    "true_class": row["class_name"],
                    "true_class_zh": row["class_name_zh"],
                    "predicted_class": prediction["predicted_class"],
                    "predicted_class_zh": prediction["predicted_class_zh"],
                    "confidence": prediction["confidence"],
                    "inference_time_ms": prediction["inference_time_ms"],
                    "correct": str(correct),
                }
            )

    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_path",
                "true_class",
                "true_class_zh",
                "predicted_class",
                "predicted_class_zh",
                "confidence",
                "inference_time_ms",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    correct_count = sum(row["correct"] == "True" for row in rows)
    total_count = len(rows)
    accuracy = correct_count / total_count if total_count else 0
    print(f"输出文件: {args.output}")
    print(f"样本数: {total_count}")
    print(f"正确数: {correct_count}")
    print(f"准确率: {accuracy:.4f}")


if __name__ == "__main__":
    main()
