"""生成报告用系统架构图和流程图。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "05_补充材料" / "图表"
FONT_PATHS = [
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/simsun.ttc"),
]


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_PATHS:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_round_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    text: str,
    fill: str,
    outline: str,
    font: ImageFont.ImageFont,
) -> None:
    draw.rounded_rectangle(xy, radius=18, fill=fill, outline=outline, width=2)
    left, top, right, bottom = xy
    lines = text.split("\n")
    line_heights = [draw.textbbox((0, 0), line, font=font)[3] for line in lines]
    total_height = sum(line_heights) + 8 * (len(lines) - 1)
    y = top + (bottom - top - total_height) / 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        x = left + (right - left - (bbox[2] - bbox[0])) / 2
        draw.text((x, y), line, fill="#111827", font=font)
        y += (bbox[3] - bbox[1]) + 8


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    draw.line([start, end], fill="#334155", width=3)
    ex, ey = end
    sx, sy = start
    if ex >= sx:
        points = [(ex, ey), (ex - 12, ey - 7), (ex - 12, ey + 7)]
    else:
        points = [(ex, ey), (ex + 12, ey - 7), (ex + 12, ey + 7)]
    draw.polygon(points, fill="#334155")


def architecture_diagram() -> None:
    image = Image.new("RGB", (1800, 960), "#f8fafc")
    draw = ImageDraw.Draw(image)
    title_font = get_font(46)
    box_font = get_font(27)
    small_font = get_font(22)
    draw.text((60, 42), "钢材表面缺陷智能质检系统总体架构", fill="#0f172a", font=title_font)

    boxes = [
        (80, 180, 340, 310, "NEU-CLS\n原始数据集", "#dbeafe", "#2563eb"),
        (430, 180, 690, 310, "数据预处理\ntrain/val/test", "#e0f2fe", "#0284c7"),
        (780, 180, 1040, 310, "DefectCNN\n模型训练", "#dcfce7", "#16a34a"),
        (1130, 180, 1390, 310, "模型权重\nbest_model.pt", "#fef3c7", "#d97706"),
        (1480, 180, 1740, 310, "Streamlit\n质检工作台", "#ede9fe", "#7c3aed"),
        (430, 520, 690, 650, "单图/批量推理\n类别+置信度", "#fce7f3", "#db2777"),
        (780, 520, 1040, 650, "缺陷知识库\n规则兜底", "#f1f5f9", "#475569"),
        (1130, 520, 1390, 650, "My Codex\n大模型增强", "#ccfbf1", "#0d9488"),
        (1480, 520, 1740, 650, "报告与导出\n案例+指南", "#fee2e2", "#dc2626"),
    ]
    for xy in boxes:
        draw_round_box(draw, xy[:4], xy[4], xy[5], xy[6], box_font)

    arrow(draw, (340, 245), (430, 245))
    arrow(draw, (690, 245), (780, 245))
    arrow(draw, (1040, 245), (1130, 245))
    arrow(draw, (1390, 245), (1480, 245))
    arrow(draw, (1260, 310), (560, 520))
    arrow(draw, (690, 585), (780, 585))
    arrow(draw, (1040, 585), (1130, 585))
    arrow(draw, (1390, 585), (1480, 585))
    arrow(draw, (1610, 520), (1610, 310))

    draw.text((78, 800), "核心优势：数据可复现、模型轻量可运行、接口有规则兜底，支持单图报告、批量导出、典型案例和 My Codex 增强说明。", fill="#334155", font=small_font)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_DIR / "系统总体架构图.png")


def flow_diagram() -> None:
    image = Image.new("RGB", (1600, 1050), "#ffffff")
    draw = ImageDraw.Draw(image)
    title_font = get_font(44)
    box_font = get_font(27)
    note_font = get_font(22)
    draw.text((60, 42), "应用运行流程图", fill="#0f172a", font=title_font)

    boxes = [
        (90, 170, 370, 290, "用户上传图片\n或选择样例", "#dbeafe", "#2563eb"),
        (500, 170, 780, 290, "图像预处理\n灰度化+缩放", "#e0f2fe", "#0284c7"),
        (910, 170, 1190, 290, "DefectCNN\n缺陷分类", "#dcfce7", "#16a34a"),
        (1230, 420, 1510, 540, "My Codex\n生成质检报告", "#ccfbf1", "#0d9488"),
        (910, 420, 1190, 540, "本地知识库\n规则兜底", "#f1f5f9", "#475569"),
        (500, 670, 780, 790, "结果整合\nTop-K+风险+摘要", "#fef3c7", "#d97706"),
        (90, 670, 370, 790, "页面展示\n检测+导出+案例", "#ede9fe", "#7c3aed"),
    ]
    for xy in boxes:
        draw_round_box(draw, xy[:4], xy[4], xy[5], xy[6], box_font)

    arrow(draw, (370, 230), (500, 230))
    arrow(draw, (780, 230), (910, 230))
    arrow(draw, (1050, 290), (1050, 420))
    arrow(draw, (1190, 480), (1230, 480))
    arrow(draw, (1050, 540), (640, 670))
    arrow(draw, (1370, 540), (720, 670))
    arrow(draw, (500, 730), (370, 730))

    draw.text((1220, 575), "接口不可用时不影响主流程", fill="#475569", font=note_font)
    draw.text((80, 900), "运行入口：run.bat 或 python -m streamlit run app.py；页面包含单图检测、批量检测、知识库、模型结果、案例和指南。", fill="#334155", font=note_font)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_DIR / "应用运行流程图.png")


def main() -> None:
    architecture_diagram()
    flow_diagram()
    print(OUTPUT_DIR / "系统总体架构图.png")
    print(OUTPUT_DIR / "应用运行流程图.png")


if __name__ == "__main__":
    main()
