# NEU-CLS 数据集说明

本项目当前主推荐数据集为 NEU-CLS 钢材表面缺陷图像数据集。

## 本地文件

```text
data/raw/NEU-CLS.zip
data/raw/NEU-CLS/
```

## 来源

- Figshare：<https://figshare.com/articles/dataset/NEU-CLS/28903550>
- DOI：<https://doi.org/10.6084/m9.figshare.28903550>
- License：CC BY 4.0

## 下载与校验

- 下载日期：2026 年 6 月 24 日。
- 压缩包大小：27,316,154 bytes。
- MD5：`19AF5A1250D8AB0CB0828D31EE2B349D`。

## 数据内容

NEU-CLS 用于钢材或带钢表面缺陷分类，包含 6 类缺陷，每类 300 张图像，总计 1800 张。

| 类别 | 数量 | 中文说明 |
| --- | ---: | --- |
| `crazing` | 300 | 龟裂 |
| `inclusion` | 300 | 夹杂 |
| `patches` | 300 | 斑块 |
| `pitted_surface` | 300 | 麻点表面 |
| `rolled-in_scale` | 300 | 轧入氧化皮 |
| `scratches` | 300 | 划痕 |

## 使用策略

当前建议先将数据集作为图像分类任务使用：

```text
输入：单张钢材表面图像
输出：6 类缺陷之一 + 置信度 + 质检建议
```

数据文件名中包含类别前缀，可直接用于生成分类标签。标签目录中的 `.txt` 文件为坐标标注，可作为后续增强功能，用于缺陷框展示或目标检测。

## 建议重新划分

原始解压结构中训练集和验证集比例不适合课程报告指标展示，建议按类别重新划分：

- 训练集：70%
- 验证集：15%
- 测试集：15%

这样可以在报告中更稳妥地给出准确率、精确率、召回率和 F1 值。

## 当前预处理方案

已新增脚本：

```text
data/preprocess_neu_cls.py
```

运行方式：

```powershell
python data/preprocess_neu_cls.py
```

脚本会从文件名前缀提取类别，并生成：

- `data/processed/neu_cls_classification/manifests/train.csv`
- `data/processed/neu_cls_classification/manifests/val.csv`
- `data/processed/neu_cls_classification/manifests/test.csv`
- `data/processed/neu_cls_classification/class_to_idx.json`
- `data/processed/neu_cls_classification/split_summary.csv`

脚本只生成清单文件，不复制原始图片。
