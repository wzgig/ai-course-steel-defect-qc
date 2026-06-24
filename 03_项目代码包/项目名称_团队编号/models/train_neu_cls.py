"""训练 NEU-CLS 钢材表面缺陷分类模型。

运行：
    python models/train_neu_cls.py --epochs 10
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from neu_cls_model import DefectCNN


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class Sample:
    image_path: Path
    label: int
    class_name: str


class NeuClsDataset(Dataset):
    """从 CSV 清单读取 NEU-CLS 图片。"""

    def __init__(self, manifest_path: Path, root: Path, image_size: int) -> None:
        self.root = root
        self.image_size = image_size
        self.samples = self._load_manifest(manifest_path)

    def _load_manifest(self, manifest_path: Path) -> list[Sample]:
        samples: list[Sample] = []
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                samples.append(
                    Sample(
                        image_path=self.root / row["image_path"],
                        label=int(row["label"]),
                        class_name=row["class_name"],
                    )
                )
        if not samples:
            raise ValueError(f"清单为空: {manifest_path}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("L")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - 0.5) / 0.5
        tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor, sample.label


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(targets, predictions)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(targets, predictions)
    return avg_loss, accuracy, targets, predictions


def save_history(history: list[dict[str, float]], output_dir: Path) -> None:
    with (output_dir / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train_loss")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [row["train_acc"] for row in history], label="train_acc")
    plt.plot(epochs, [row["val_acc"] for row in history], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=160)
    plt.close()


def save_confusion_matrix(
    targets: list[int],
    predictions: list[int],
    class_names: list[str],
    output_dir: Path,
) -> None:
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(class_names))))
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="训练钢材表面缺陷分类模型")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=root / "data" / "processed" / "neu_cls_classification",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "models" / "artifacts" / "neu_cls_cnn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260624)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    class_to_idx = json.loads((args.data_root / "class_to_idx.json").read_text(encoding="utf-8"))
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]

    train_dataset = NeuClsDataset(args.data_root / "manifests" / "train.csv", root, args.image_size)
    val_dataset = NeuClsDataset(args.data_root / "manifests" / "val.csv", root, args.image_size)
    test_dataset = NeuClsDataset(args.data_root / "manifests" / "test.csv", root, args.image_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DefectCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_checkpoint = args.output_dir / "best_model.pt"
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "class_names": class_names,
                    "image_size": args.image_size,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                best_checkpoint,
            )

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, targets, predictions = evaluate(model, test_loader, criterion, device)

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model": "DefectCNN",
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_acc": round(float(checkpoint["val_acc"]), 6),
        "test_loss": round(test_loss, 6),
        "test_acc": round(test_acc, 6),
        "test_macro_precision": round(float(precision), 6),
        "test_macro_recall": round(float(recall), 6),
        "test_macro_f1": round(float(f1), 6),
        "train_count": len(train_dataset),
        "val_count": len(val_dataset),
        "test_count": len(test_dataset),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }

    save_history(history, args.output_dir)
    save_confusion_matrix(targets, predictions, class_names, args.output_dir)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
