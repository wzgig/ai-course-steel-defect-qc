# My Codex 接口测试记录

测试日期：2026 年 6 月 24 日

## 接口来源

- 服务来源：My Codex
- API 地址：`https://api.9e.lv/v1`
- 实际端点：`https://api.9e.lv/v1/responses`
- 模型：`gpt-5.5`
- 配置来源：本机 `%USERPROFILE%\.codex\config.toml` 与 `%USERPROFILE%\.codex\auth.json`

说明：项目代码只在运行时读取本机 Codex 配置，不会将 API Key 写入项目文件。

## 当前应用支持的配置方式

优先读取显式环境变量：

```powershell
$env:CODEX_API_URL="https://api.9e.lv/v1"
$env:CODEX_MODEL="gpt-5.5"
$env:CODEX_API_KEY="your_key"
$env:CODEX_WIRE_API="responses"
```

如果没有设置环境变量，应用会尝试读取本机 Codex 配置：

```text
%USERPROFILE%\.codex\config.toml
%USERPROFILE%\.codex\auth.json
```

## 真实调用测试

测试图片：

```text
data/raw/NEU-CLS/valid/valid/images/crazing_1.jpg
```

模型识别结果：

| 项目 | 结果 |
| --- | --- |
| 缺陷类别 | 龟裂 |
| 英文标签 | `crazing` |
| 风险等级 | 高 |
| 大模型模式 | `llm_enhanced` |

大模型报告节选：

> 模型识别该钢材表面缺陷为龟裂（crazing），预测置信度为 99.65%。龟裂表现为钢材表面细密网状或线状裂纹，会破坏表面连续性，并可能在后续轧制、成形、酸洗或服役过程中进一步扩展。建议对该批次钢材进行重点复检，扩大抽检范围并追溯轧制温度、冷却速度及张力控制等工艺参数。

## 结论

My Codex 接口已可用于本项目的大模型增强报告生成。即使接口不可用，系统仍会自动回退到本地规则知识库报告，保证作业演示和代码复现稳定。
