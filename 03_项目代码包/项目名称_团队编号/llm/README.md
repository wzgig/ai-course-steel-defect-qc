# llm

用于存放大模型交互代码、提示词模板和输出解析逻辑。

当前已有文件：

- `quality_report.py`：钢材表面缺陷解释、风险判断和质检报告生成。

## 应用场景

深度学习模型输出缺陷类别和置信度后，本模块生成：

- 缺陷类型解释。
- 可能成因。
- 风险等级。
- 处理建议。
- 质检报告摘要。

## 运行模式

默认使用本地知识库规则生成报告，不需要联网，也不需要密钥。

模块会优先读取本机 Codex 配置：

```text
%USERPROFILE%\.codex\config.toml
%USERPROFILE%\.codex\auth.json
```

当前已验证的 My Codex 配置：

```powershell
$env:CODEX_API_URL="https://api.9e.lv/v1"
$env:CODEX_MODEL="gpt-5.5"
$env:CODEX_WIRE_API="responses"
```

如果需要手动覆盖接口密钥，再补充：

```powershell
$env:CODEX_API_KEY="your_key"
```

也支持通用 LLM 配置：

```powershell
$env:LLM_API_URL="https://your-api-host/v1/chat/completions"
$env:LLM_API_KEY="your_key"
$env:LLM_MODEL="your_model"
```

还支持 OpenAI Responses API：

```powershell
$env:OPENAI_API_KEY="your_key"
$env:OPENAI_MODEL="gpt-4.1-mini"
```

不要把真实密钥写入代码或提交材料。

## 后续需要补充

- 根据最终报告措辞微调提示词。
- 在报告中说明大模型模块的输入、输出和兜底机制。
