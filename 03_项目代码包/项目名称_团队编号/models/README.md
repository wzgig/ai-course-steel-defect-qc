# models

用于存放深度学习模型定义、训练脚本、推理脚本和权重文件。

当前已有文件：

- `neu_cls_model.py`：轻量 CNN 模型定义。
- `train_neu_cls.py`：训练脚本，输出权重、指标、训练曲线、混淆矩阵。
- `infer_neu_cls.py`：单张图片推理脚本。
- `batch_infer_neu_cls.py`：批量推理测试集并输出 CSV。
- `训练结果记录.md`：第一版模型结果和报告素材。

## 训练

先确保已运行数据预处理：

```powershell
python data/preprocess_neu_cls.py
```

然后训练模型：

```powershell
python models/train_neu_cls.py --epochs 10
```

训练结果输出到：

```text
models/artifacts/neu_cls_cnn/
```

当前第一版训练结果：

- 测试集准确率：0.944444
- 测试集宏平均精确率：0.946831
- 测试集宏平均召回率：0.944444
- 测试集宏平均 F1：0.944121

## 推理

```powershell
python models/infer_neu_cls.py --image data/raw/NEU-CLS/valid/valid/images/crazing_1.jpg
```

## 批量推理

```powershell
python models/batch_infer_neu_cls.py
```

输出：

```text
models/artifacts/neu_cls_cnn/batch_test_predictions.csv
```

## 后续需要补充

- 将训练指标写入项目主报告。
- 在 `app.py` 中接入推理脚本。
- 接入大模型解释模块，生成质检建议。
