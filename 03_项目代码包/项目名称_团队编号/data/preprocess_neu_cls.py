"""NEU-CLS 钢材表面缺陷数据预处理脚本。

功能：
1. 扫描 raw/NEU-CLS 下的原始图片。
2. 从文件名前缀提取缺陷类别。
3. 按类别分层划分 train/val/test。
4. 生成 CSV 清单、类别映射和划分统计。

运行方式：
    python data/preprocess_neu_cls.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path


CLASS_NAMES_ZH = {
    "crazing": "龟裂",
    "inclusion": "夹杂",
    "patches": "斑块",
    "pitted_surface": "麻点表面",
    "rolled-in_scale": "轧入氧化皮",
    "scratches": "划痕",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_PATTERN = re.compile(r"^(?P<class_name>.+)_\d+$")


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_class_name(image_path: Path) -> str:
    match = CLASS_PATTERN.match(image_path.stem)
    if not match:
        raise ValueError(f"无法从文件名提取类别: {image_path.name}")
    return match.group("class_name")


def collect_images(raw_root: Path) -> dict[str, list[Path]]:
    by_class: dict[str, list[Path]] = defaultdict(list)
    for path in raw_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            by_class[parse_class_name(path)].append(path)

    for class_name in by_class:
        by_class[class_name] = sorted(by_class[class_name])
    return dict(sorted(by_class.items()))


def split_class_images(
    images: list[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    items = images[:]
    rng = random.Random(seed)
    rng.shuffle(items)

    train_count = int(len(items) * train_ratio)
    val_count = int(len(items) * val_ratio)

    train_items = sorted(items[:train_count])
    val_items = sorted(items[train_count : train_count + val_count])
    test_items = sorted(items[train_count + val_count :])
    return train_items, val_items, test_items


def rel_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def write_manifest(
    path: Path,
    rows: list[dict[str, str | int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_path", "label", "class_name", "class_name_zh", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, summary: dict[str, dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["class_name", "class_name_zh", "train", "val", "test", "total"],
        )
        writer.writeheader()
        for class_name, counts in summary.items():
            writer.writerow(
                {
                    "class_name": class_name,
                    "class_name_zh": CLASS_NAMES_ZH.get(class_name, class_name),
                    "train": counts["train"],
                    "val": counts["val"],
                    "test": counts["test"],
                    "total": counts["total"],
                }
            )


def build_rows(
    split: str,
    images: list[Path],
    class_name: str,
    label: int,
    root: Path,
) -> list[dict[str, str | int]]:
    return [
        {
            "image_path": rel_path(image_path, root),
            "label": label,
            "class_name": class_name,
            "class_name_zh": CLASS_NAMES_ZH.get(class_name, class_name),
            "split": split,
        }
        for image_path in images
    ]


def preprocess(
    raw_root: Path,
    output_root: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, object]:
    root = project_root()
    by_class = collect_images(raw_root)
    if not by_class:
        raise FileNotFoundError(f"未找到图片文件: {raw_root}")

    class_names = sorted(by_class)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    manifests_root = output_root / "manifests"
    rows_by_split: dict[str, list[dict[str, str | int]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    summary: dict[str, dict[str, int]] = {}

    for class_name in class_names:
        train_items, val_items, test_items = split_class_images(
            by_class[class_name],
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed + class_to_idx[class_name],
        )
        label = class_to_idx[class_name]
        rows_by_split["train"].extend(
            build_rows("train", train_items, class_name, label, root)
        )
        rows_by_split["val"].extend(build_rows("val", val_items, class_name, label, root))
        rows_by_split["test"].extend(
            build_rows("test", test_items, class_name, label, root)
        )
        summary[class_name] = {
            "train": len(train_items),
            "val": len(val_items),
            "test": len(test_items),
            "total": len(by_class[class_name]),
        }

    all_rows: list[dict[str, str | int]] = []
    for split in ("train", "val", "test"):
        rows_by_split[split] = sorted(
            rows_by_split[split], key=lambda item: (str(item["class_name"]), str(item["image_path"]))
        )
        write_manifest(manifests_root / f"{split}.csv", rows_by_split[split])
        all_rows.extend(rows_by_split[split])
    write_manifest(manifests_root / "all.csv", sorted(all_rows, key=lambda item: str(item["image_path"])))

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "class_to_idx.json").write_text(
        json.dumps(class_to_idx, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "class_names_zh.json").write_text(
        json.dumps(CLASS_NAMES_ZH, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_csv(output_root / "split_summary.csv", summary)

    return {
        "raw_root": rel_path(raw_root, root),
        "output_root": rel_path(output_root, root),
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": round(1 - train_ratio - val_ratio, 6),
        "class_to_idx": class_to_idx,
        "summary": summary,
        "total_images": sum(item["total"] for item in summary.values()),
    }


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="生成 NEU-CLS 分类任务数据清单")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=root / "data" / "raw" / "NEU-CLS",
        help="NEU-CLS 解压后的原始数据目录",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "data" / "processed" / "neu_cls_classification",
        help="输出目录",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260624)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_ratio <= 0 or args.val_ratio <= 0:
        raise ValueError("train-ratio 和 val-ratio 必须大于 0")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train-ratio + val-ratio 必须小于 1")

    result = preprocess(
        raw_root=args.raw_root,
        output_root=args.output_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
