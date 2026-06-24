# data

用于存放数据集、数据说明和预处理脚本。

当前已收集主推荐数据集：

- `raw/NEU-CLS.zip`
- `raw/NEU-CLS/`
- `数据集说明_NEU-CLS.md`
- `preprocess_neu_cls.py`

后续需要补充：

- 处理后数据的结构说明。
- 数据清洗和预处理记录。

## 预处理

运行：

```powershell
python data/preprocess_neu_cls.py
```

输出：

```text
processed/neu_cls_classification/
├── class_names_zh.json
├── class_to_idx.json
├── manifests/
│   ├── all.csv
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── split_summary.csv
└── split_summary.json
```

预处理脚本不会复制原始图片，只生成可复现实验用的清单文件。
