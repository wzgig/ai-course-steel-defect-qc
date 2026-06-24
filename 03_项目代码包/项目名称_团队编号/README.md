# 钢材表面缺陷智能质检与大模型诊断报告系统

这是《人工智能基础 B》期末大作业代码包。项目面向工业钢材表面质检场景，基于 NEU-CLS 数据集训练轻量 CNN 分类模型，并结合本地缺陷知识库与 My Codex 大模型接口生成质检解释报告。

## 当前能力

- NEU-CLS 数据集整理与 train/val/test 清单划分。
- PyTorch 轻量 CNN 六分类模型训练与测试。
- 单图缺陷检测：输出缺陷类别、中文名称、置信度、Top-K 结果和推理耗时。
- 质检解释报告：输出缺陷解释、风险等级、可能成因和处理建议。
- My Codex 增强报告：自动读取本机 Codex 配置，支持 `https://api.9e.lv/v1/responses`。
- Streamlit 质检工作台：支持单图检测、批量检测、缺陷知识库、模型结果、典型案例和网页内使用指南。
- 结果导出：支持单图 Markdown 报告、批量 CSV 明细和批量 Markdown 摘要。
- 轻量演示样例：`data/samples/` 内置 6 张示例图，便于公开仓库克隆后直接演示页面。

## 目录结构

```text
项目名称_团队编号/
├── data/                    # 数据集、预处理脚本、清单文件
├── models/                  # 模型定义、训练、推理、权重和指标
├── llm/                     # 缺陷知识库、大模型配置读取、报告生成
├── app.py                   # Streamlit 网站入口
├── requirements.txt         # 依赖库清单
├── run.bat                  # Windows 一键运行脚本
├── 应用运行说明.md
└── 网站使用指南与案例.md
```

说明：公开 GitHub 仓库不提交完整 `data/raw/` 原始数据集和数据压缩包。需要复现实验训练时，请按 `data/数据集说明_NEU-CLS.md` 下载完整 NEU-CLS 数据集；仅演示网站时可直接使用 `data/samples/` 中的内置样例。

## 环境要求

- Python：3.12.10
- 页面框架：Streamlit 1.58.0
- 深度学习框架：PyTorch 2.12.0

安装依赖：

```powershell
pip install -r requirements.txt
```

## 运行方式

Windows 下双击：

```text
run.bat
```

或在命令行运行：

```powershell
python -m streamlit run app.py
```

浏览器访问：

```text
http://localhost:8501
```

## 关键结果

当前模型在 NEU-CLS 测试集上的结果：

| 指标 | 数值 |
| --- | --- |
| 测试准确率 | 94.44% |
| 宏平均精确率 | 94.68% |
| 宏平均召回率 | 94.44% |
| 宏平均 F1 | 94.41% |
| 测试集规模 | 270 张 |

## 常用命令

```powershell
python data\preprocess_neu_cls.py
python models\train_neu_cls.py --epochs 8 --batch-size 64 --image-size 160
python models\infer_neu_cls.py --image data\raw\NEU-CLS\valid\valid\images\crazing_1.jpg
python models\batch_infer_neu_cls.py
python -m streamlit run app.py
```

## My Codex 接口说明

项目不会把 API Key 写入代码或文档。应用会优先读取：

```text
%USERPROFILE%\.codex\config.toml
%USERPROFILE%\.codex\auth.json
```

已验证配置方向：

```text
base_url = "https://api.9e.lv/v1"
wire_api = "responses"
model = "gpt-5.5"
```

如果接口不可用，系统仍会使用本地缺陷知识库生成稳定报告。

## 使用指南

详细操作流程和三个演示案例见：

```text
网站使用指南与案例.md
```
