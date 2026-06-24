"""钢材表面缺陷智能质检系统主入口。

运行：
    streamlit run app.py
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from urllib.parse import quote, unquote

import pandas as pd
import streamlit as st
from PIL import Image

from llm.quality_report import DEFECT_KNOWLEDGE, generate_quality_report, llm_config_status
from models.infer_neu_cls import predict_image


ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "models" / "artifacts" / "neu_cls_cnn" / "best_model.pt"
MANIFEST_PATH = ROOT / "data" / "processed" / "neu_cls_classification" / "manifests" / "test.csv"
SAMPLE_MANIFEST_PATH = ROOT / "data" / "samples" / "samples.csv"
METRICS_PATH = ROOT / "models" / "artifacts" / "neu_cls_cnn" / "metrics.json"
CONFUSION_MATRIX_PATH = ROOT / "models" / "artifacts" / "neu_cls_cnn" / "confusion_matrix.png"
ACCURACY_CURVE_PATH = ROOT / "models" / "artifacts" / "neu_cls_cnn" / "accuracy_curve.png"
LOSS_CURVE_PATH = ROOT / "models" / "artifacts" / "neu_cls_cnn" / "loss_curve.png"

DEMO_CASES = [
    {
        "title": "案例一：生产线龟裂抽检",
        "class_name": "crazing",
        "scene": "热轧或冷轧产线抽检发现疑似裂纹纹理，需要快速判断是否扩大抽检。",
        "operation": "选择龟裂样例，运行单图检测，重点查看风险等级、Top-K 置信度和处理建议。",
        "highlight": "若系统识别为龟裂且置信度较高，可在报告中记录“重点复检、检查温度/张力/冷却记录”。",
        "llm": True,
    },
    {
        "title": "案例二：异常批次快速复核",
        "class_name": "rolled-in_scale",
        "scene": "同一批次多张表面图像存在氧化皮压入疑似缺陷，需要批量统计缺陷分布。",
        "operation": "进入批量检测页，选择测试集样本数，运行检测后导出 CSV 和批量摘要。",
        "highlight": "根据预测类别分布、高风险样本数和准确率，判断是否需要追溯除鳞水压、喷嘴或加热参数。",
        "llm": False,
    },
    {
        "title": "案例三：质检报告辅助撰写",
        "class_name": "scratches",
        "scene": "外观要求较高的产品出现划痕，需要形成可提交给工艺或质量人员的文字说明。",
        "operation": "选择划痕样例，开启 My Codex，下载 Markdown 报告并作为质检记录初稿。",
        "highlight": "报告会同时包含模型结论、可能成因、处理建议和提示词，便于复核与追溯。",
        "llm": True,
    },
]


@st.cache_data(show_spinner=False)
def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


@st.cache_data(show_spinner=False)
def load_manifest_rows() -> list[dict[str, str]]:
    return load_csv_rows(MANIFEST_PATH)


@st.cache_data(show_spinner=False)
def load_sample_rows() -> list[dict[str, str]]:
    bundled_rows = [row for row in load_csv_rows(SAMPLE_MANIFEST_PATH) if (ROOT / row["image_path"]).exists()]
    if bundled_rows:
        return bundled_rows

    selected: list[dict[str, str]] = []
    seen_classes: set[str] = set()
    for row in load_manifest_rows():
        if row["class_name"] not in seen_classes and (ROOT / row["image_path"]).exists():
            selected.append(row)
            seen_classes.add(row["class_name"])
        if len(selected) >= 6:
            break
    return selected


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.Series | None:
    if not METRICS_PATH.exists():
        return None
    return pd.read_json(METRICS_PATH, typ="series")


def find_sample_for_class(class_name: str) -> dict[str, str] | None:
    for row in load_sample_rows():
        if row.get("class_name") == class_name:
            return row
    for row in load_manifest_rows():
        if row.get("class_name") == class_name and (ROOT / row["image_path"]).exists():
            return row
    return None


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


def build_report_markdown(image_label: str, prediction: dict, report: dict) -> str:
    top_k_lines = "\n".join(
        f"- {item['class_name_zh']} ({item['class_name']}): {item['confidence']:.2%}"
        for item in prediction["top_k"]
    )
    cause_lines = "\n".join(f"- {item}" for item in report["possible_causes"])
    recommendation_lines = "\n".join(f"- {item}" for item in report["recommendations"])
    llm_section = ""
    if report.get("llm_report"):
        llm_section = f"\n## 大模型生成报告\n\n{report['llm_report']}\n"

    return f"""# 钢材表面缺陷质检报告

## 样本信息

- 图片：{image_label}
- 预测缺陷：{prediction['predicted_class_zh']} ({prediction['predicted_class']})
- 置信度：{prediction['confidence']:.2%}
- 推理耗时：{prediction['inference_time_ms']:.2f} ms
- 风险等级：{report['risk_level']}
- 报告模式：{report['mode']}

## Top-K 预测

{top_k_lines}

## 缺陷解释

{report['summary']}

## 可能成因

{cause_lines}

## 处理建议

{recommendation_lines}
{llm_section}
"""


def build_batch_summary_markdown(result_df: pd.DataFrame) -> str:
    total = len(result_df)
    avg_confidence = float(result_df["置信度"].mean()) if total else 0.0
    high_risk_count = int(result_df["风险等级"].astype(str).str.contains("高").sum()) if total else 0
    distribution_lines = "\n".join(
        f"- {label}：{count} 张"
        for label, count in result_df["预测类别中文"].value_counts().items()
    )

    accuracy_line = "未提供真实类别，无法计算准确率。"
    if "是否正确" in result_df.columns and (result_df["是否正确"] != "").any():
        correct = int((result_df["是否正确"] == "True").sum())
        labeled_total = int((result_df["是否正确"] != "").sum())
        accuracy_line = f"{correct}/{labeled_total}，准确率 {correct / labeled_total:.2%}。"

    return f"""# 批量检测摘要

- 检测图片数量：{total}
- 平均置信度：{avg_confidence:.2%}
- 高风险相关样本数：{high_risk_count}
- 带真实标签样本准确率：{accuracy_line}

## 预测类别分布

{distribution_lines}

## 使用建议

- 若同一缺陷类别集中出现，应优先追溯对应批次的工艺参数和设备状态。
- 对置信度低或高风险样本，应安排人工复检，不直接作为最终质量判定。
- CSV 明细可用于课程报告中的功能测试、批量推理结果统计和演示视频讲解。
"""


def predict_with_report(image_path: Path, image_label: str, enable_llm: bool) -> tuple[dict, dict, str]:
    prediction = predict_image(image_path, checkpoint_path=CHECKPOINT_PATH, top_k=3)
    report = generate_quality_report(prediction, enable_llm=enable_llm)
    report_markdown = build_report_markdown(image_label, prediction, report)
    return prediction, report, report_markdown


def show_prediction(prediction: dict) -> None:
    confidence = float(prediction["confidence"])
    col1, col2, col3 = st.columns(3)
    col1.metric("预测缺陷", prediction["predicted_class_zh"])
    col2.metric("置信度", f"{confidence:.2%}")
    col3.metric("推理耗时", f"{prediction['inference_time_ms']:.2f} ms")

    top_k_df = pd.DataFrame(
        [
            {
                "缺陷类别": item["class_name_zh"],
                "英文标签": item["class_name"],
                "置信度": item["confidence"],
            }
            for item in prediction["top_k"]
        ]
    )
    st.subheader("Top-K 预测结果")
    st.dataframe(top_k_df, use_container_width=True, hide_index=True)
    st.bar_chart(top_k_df.set_index("缺陷类别")["置信度"])


def show_quality_report(report: dict, report_markdown: str, filename: str) -> None:
    st.subheader("质检解释与处理建议")
    st.info(report["summary"])

    col1, col2 = st.columns([1, 2])
    col1.metric("风险等级", report["risk_level"])
    col2.write(f"报告模式：`{report['mode']}`")

    st.markdown("**可能成因**")
    for cause in report["possible_causes"]:
        st.write(f"- {cause}")

    st.markdown("**处理建议**")
    for recommendation in report["recommendations"]:
        st.write(f"- {recommendation}")

    if report.get("llm_report"):
        st.markdown("**大模型生成报告**")
        st.write(report["llm_report"])

    st.download_button(
        "下载本次质检报告 Markdown",
        data=report_markdown.encode("utf-8"),
        file_name=filename,
        mime="text/markdown",
        use_container_width=True,
    )

    with st.expander("查看大模型提示词"):
        st.code(report["llm_prompt"], language="text")


def show_model_metrics() -> None:
    st.subheader("模型训练与测试结果")
    metrics = load_metrics()
    if metrics is not None:
        metric_cols = st.columns(4)
        metric_cols[0].metric("测试准确率", f"{float(metrics['test_acc']):.2%}")
        metric_cols[1].metric("宏平均精确率", f"{float(metrics['test_macro_precision']):.2%}")
        metric_cols[2].metric("宏平均召回率", f"{float(metrics['test_macro_recall']):.2%}")
        metric_cols[3].metric("宏平均 F1", f"{float(metrics['test_macro_f1']):.2%}")
    else:
        st.warning("尚未找到模型指标文件，请先运行训练脚本。")

    image_cols = st.columns(3)
    if CONFUSION_MATRIX_PATH.exists():
        image_cols[0].image(str(CONFUSION_MATRIX_PATH), caption="混淆矩阵")
    if ACCURACY_CURVE_PATH.exists():
        image_cols[1].image(str(ACCURACY_CURVE_PATH), caption="准确率曲线")
    if LOSS_CURVE_PATH.exists():
        image_cols[2].image(str(LOSS_CURVE_PATH), caption="损失曲线")


def show_project_overview() -> None:
    metrics = load_metrics()
    rows = load_manifest_rows()
    status = llm_config_status()

    st.caption("面向钢材表面缺陷分类、批量复核和质检报告生成的课程项目工作台。")
    col1, col2, col3, col4 = st.columns(4)
    if metrics is not None:
        col1.metric("测试集准确率", f"{float(metrics['test_acc']):.2%}")
        col2.metric("宏平均 F1", f"{float(metrics['test_macro_f1']):.2%}")
    else:
        col1.metric("测试集准确率", "未生成")
        col2.metric("宏平均 F1", "未生成")
    col3.metric("测试样本数", str(len(rows)))
    col4.metric("报告增强", "My Codex 可用" if status["available"] else "本地规则")


def resolve_input_image(uploaded_file, selected_sample: str, sample_rows: list[dict[str, str]]) -> tuple[Path, str] | None:
    if uploaded_file is not None:
        return save_uploaded_file(uploaded_file), uploaded_file.name

    if selected_sample:
        selected_row = next((row for row in sample_rows if row["image_path"] == selected_sample), None)
        if selected_row:
            return ROOT / selected_row["image_path"], selected_row["image_path"]
    return None


def sidebar_controls() -> tuple[object, str, bool, bool, list[dict[str, str]]]:
    with st.sidebar:
        st.header("检测输入")
        uploaded_file = st.file_uploader("上传钢材表面图片", type=["jpg", "jpeg", "png", "bmp"])
        sample_rows = load_sample_rows()
        sample_options = [""] + [row["image_path"] for row in sample_rows]
        requested_sample = unquote(st.query_params.get("sample", ""))
        default_sample_index = sample_options.index(requested_sample) if requested_sample in sample_options else 0
        selected_sample = st.selectbox(
            "或选择测试样例",
            sample_options,
            index=default_sample_index,
            format_func=lambda item: "请选择样例" if not item else item,
        )

        status = llm_config_status()
        if status["available"]:
            st.success(status["message"])
            st.caption(
                f"来源：{status.get('source', 'unknown')}；端点类型：{status['endpoint_type']}；"
                f"密钥：{'已配置' if status['has_api_key'] else '本地免密'}"
            )
        else:
            st.warning(status["message"])
        llm_default = st.query_params.get("llm", "0") == "1" and bool(status["available"])
        enable_llm = st.checkbox("单图检测时调用 My Codex", value=llm_default, disabled=not bool(status["available"]))
        run_button = st.button("开始检测", type="primary", use_container_width=True)
    return uploaded_file, selected_sample, enable_llm, run_button, sample_rows


def single_detection_tab(uploaded_file, selected_sample: str, enable_llm: bool, run_button: bool, sample_rows: list[dict[str, str]]) -> None:
    image_input = resolve_input_image(uploaded_file, selected_sample, sample_rows)
    if image_input is None:
        st.write("请上传图片，或从左侧选择一个测试样例。")
        st.caption("建议先选择一个内置样例熟悉流程，再上传自己的图片。")
        return

    image_path, image_label = image_input
    col_image, col_result = st.columns([1, 1.35])
    with col_image:
        st.subheader("待检测图像")
        st.image(Image.open(image_path), caption=image_label, use_container_width=True)

    if run_button or selected_sample:
        with st.spinner("正在执行模型推理并生成质检报告..."):
            prediction, report, report_markdown = predict_with_report(image_path, image_label, enable_llm=enable_llm)
        with col_result:
            show_prediction(prediction)
        show_quality_report(report, report_markdown, "steel_defect_quality_report.md")


def batch_detection_tab() -> None:
    st.subheader("批量检测与结果导出")
    st.write("批量检测用于快速复核多张钢材表面图片。为保证速度和稳定性，批量模式默认使用模型推理和规则报告，不逐张调用 My Codex。")

    source = st.radio("批量来源", ["测试集前 30 张", "上传多张图片"], horizontal=True)
    rows_for_batch: list[tuple[Path, str, str | None]] = []

    if source == "测试集前 30 张":
        limit = st.slider("测试集样本数量", min_value=6, max_value=60, value=30, step=6)
        for row in load_manifest_rows()[:limit]:
            rows_for_batch.append((ROOT / row["image_path"], row["image_path"], row.get("class_name")))
    else:
        uploaded_files = st.file_uploader(
            "上传多张图片",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
        )
        for uploaded_file in uploaded_files:
            rows_for_batch.append((save_uploaded_file(uploaded_file), uploaded_file.name, None))

    if not rows_for_batch:
        st.info("请上传图片，或选择测试集样本数量。")
        return

    if st.button("运行批量检测", type="primary", use_container_width=True):
        results = []
        progress = st.progress(0)
        for index, (image_path, image_label, true_class) in enumerate(rows_for_batch, start=1):
            prediction = predict_image(image_path, checkpoint_path=CHECKPOINT_PATH, top_k=3)
            report = generate_quality_report(prediction, enable_llm=False)
            results.append(
                {
                    "图片": image_label,
                    "真实类别": true_class or "",
                    "预测类别": prediction["predicted_class"],
                    "预测类别中文": prediction["predicted_class_zh"],
                    "置信度": prediction["confidence"],
                    "风险等级": report["risk_level"],
                    "推理耗时ms": prediction["inference_time_ms"],
                    "是否正确": "" if true_class is None else str(true_class == prediction["predicted_class"]),
                }
            )
            progress.progress(index / len(rows_for_batch))

        result_df = pd.DataFrame(results)
        st.session_state["batch_results"] = result_df

    result_df = st.session_state.get("batch_results")
    if isinstance(result_df, pd.DataFrame) and not result_df.empty:
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        summary = build_batch_summary_markdown(result_df)
        metric_cols = st.columns(4)
        metric_cols[0].metric("检测图片数", str(len(result_df)))
        metric_cols[1].metric("平均置信度", f"{float(result_df['置信度'].mean()):.2%}")
        metric_cols[2].metric("高风险相关样本", str(int(result_df["风险等级"].astype(str).str.contains("高").sum())))
        if "是否正确" in result_df.columns and (result_df["是否正确"] != "").any():
            correct = (result_df["是否正确"] == "True").sum()
            total = (result_df["是否正确"] != "").sum()
            metric_cols[3].metric("批量样本准确率", f"{correct / total:.2%}", f"{correct}/{total}")
        else:
            metric_cols[3].metric("批量样本准确率", "无标签")

        distribution = result_df["预测类别中文"].value_counts().rename_axis("预测类别").reset_index(name="数量")
        st.subheader("批量预测分布")
        st.bar_chart(distribution.set_index("预测类别")["数量"])
        st.download_button(
            "下载批量检测结果 CSV",
            data=result_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="batch_detection_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "下载批量检测摘要 Markdown",
            data=summary.encode("utf-8"),
            file_name="batch_detection_summary.md",
            mime="text/markdown",
            use_container_width=True,
        )


def knowledge_tab() -> None:
    st.subheader("缺陷知识库")
    rows = []
    for item in DEFECT_KNOWLEDGE.values():
        rows.append(
            {
                "英文标签": item.class_name,
                "中文名称": item.class_name_zh,
                "基础风险": item.base_risk,
                "缺陷说明": item.description,
                "典型成因": "；".join(item.possible_causes),
                "处理建议": "；".join(item.recommendations),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### 使用场景示例")
    case_cols = st.columns(3)
    case_cols[0].markdown("**案例 1：生产线抽检**\n\n质检员从生产线上抽取样片，上传后系统给出缺陷类别和风险等级，用于快速判断是否扩大抽检。")
    case_cols[1].markdown("**案例 2：异常批次复核**\n\n对同一批次多张图片运行批量检测，导出 CSV 后统计高风险缺陷占比。")
    case_cols[2].markdown("**案例 3：质检报告辅助**\n\n对关键样本启用 My Codex，生成更完整的文字报告，辅助报告撰写和问题追溯。")


def cases_tab() -> None:
    st.subheader("典型案例演示")
    st.write("以下案例可直接用于课堂展示、答辩演示或测试记录补充。")

    for case in DEMO_CASES:
        sample = find_sample_for_class(case["class_name"])
        sample_path = sample["image_path"] if sample else ""
        with st.container(border=True):
            col_image, col_text = st.columns([1, 2])
            with col_image:
                if sample_path:
                    st.image(Image.open(ROOT / sample_path), caption=sample_path, use_container_width=True)
                    st.link_button(
                        "在单图检测中打开",
                        f"?sample={quote(sample_path)}&llm={'1' if case['llm'] else '0'}",
                        use_container_width=True,
                    )
                else:
                    st.warning("未找到对应类别样例。")
            with col_text:
                st.markdown(f"**{case['title']}**")
                st.write(case["scene"])
                st.markdown("**操作路径**")
                st.write(case["operation"])
                st.markdown("**演示重点**")
                st.write(case["highlight"])
                st.code(
                    "\n".join(
                        [
                            f"样例类别：{case['class_name']}",
                            f"样例路径：{sample_path or '未找到'}",
                            f"My Codex：{'建议开启' if case['llm'] else '不需要开启'}",
                        ]
                    ),
                    language="text",
                )


def guide_tab() -> None:
    st.subheader("网站使用指南")
    st.markdown(
        """
        ### 1. 启动网站

        在项目代码包目录运行 `python -m streamlit run app.py`，浏览器打开 `http://localhost:8501`。

        ### 2. 单图检测

        在左侧选择内置样例或上传图片；需要更完整文字报告时勾选“单图检测时调用 My Codex”；点击“开始检测”后查看预测类别、置信度、Top-K 结果、风险等级和处理建议。

        ### 3. 批量检测

        进入“批量检测”页，选择测试集样本数或上传多张图片；运行后查看明细表、类别分布、平均置信度和高风险样本数；下载 CSV 或 Markdown 摘要。

        ### 4. 缺陷知识库

        进入“缺陷知识库”页查看六类缺陷的中文解释、典型成因和处理建议，可作为报告撰写与演示讲解依据。

        ### 5. 模型结果

        进入“模型结果”页查看测试准确率、宏平均 F1、混淆矩阵、准确率曲线和损失曲线，用于证明模型训练效果。

        ### 6. 典型案例

        进入“典型案例”页，按案例给出的操作路径演示生产线抽检、异常批次复核和质检报告辅助三个场景。

        ### 常见问题

        - 如果页面提示未找到模型权重，先运行 `python models/train_neu_cls.py --epochs 8`。
        - 如果 My Codex 不可用，系统仍会使用本地缺陷知识库生成规则报告。
        - 批量检测默认不逐张调用大模型，避免等待时间过长和接口消耗过高。
        """
    )
    st.markdown("### 推荐课堂演示顺序")
    st.write("先展示首页指标和单图检测，再展示模型结果，随后运行批量检测并导出结果，最后用典型案例说明工程应用价值。")


def main() -> None:
    st.set_page_config(page_title="钢材表面缺陷智能质检系统", layout="wide")
    st.title("钢材表面缺陷智能质检系统")
    show_project_overview()

    if not CHECKPOINT_PATH.exists():
        st.error("未找到模型权重，请先运行：python models/train_neu_cls.py --epochs 8")
        return

    uploaded_file, selected_sample, enable_llm, run_button, sample_rows = sidebar_controls()

    tab_detect, tab_batch, tab_knowledge, tab_metrics, tab_cases, tab_guide = st.tabs(
        ["单图检测", "批量检测", "缺陷知识库", "模型结果", "典型案例", "使用指南"]
    )

    with tab_detect:
        single_detection_tab(uploaded_file, selected_sample, enable_llm, run_button, sample_rows)
    with tab_batch:
        batch_detection_tab()
    with tab_knowledge:
        knowledge_tab()
    with tab_metrics:
        show_model_metrics()
    with tab_cases:
        cases_tab()
    with tab_guide:
        guide_tab()


if __name__ == "__main__":
    main()
