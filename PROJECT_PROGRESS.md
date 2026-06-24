# PROJECT_PROGRESS

## 2026-06-24 本轮项目提升

操作者：Codex

目标：在已有钢材表面缺陷智能质检项目基础上，进一步提升网站和交付材料，不止满足基础题目要求，并补充网站使用指南和案例。

主要变更：

- 升级 `03_项目代码包/项目名称_团队编号/app.py`：新增首页指标、批量检测统计、批量 CSV/Markdown 导出、典型案例页和网页内使用指南。
- 新增 `03_项目代码包/项目名称_团队编号/网站使用指南与案例.md`。
- 新增 `05_补充材料/网站使用指南与案例.md`，作为补充材料摘要。
- 更新根目录 `README.md`、代码包 `README.md`、`应用运行说明.md`。
- 更新 `05_补充材料/功能测试表.md`、`系统架构与流程图.md`、`补充材料清单.md`。
- 更新 `01_任务管理/当前交付检查清单.md` 和 `01_任务管理/下一阶段完成作业规划.md`。
- 更新 `02_项目主报告/项目主报告初稿.md` 与 `02_项目主报告/项目主报告成稿.md`，并重新导出 DOCX/PDF。
- 重新生成架构图和流程图 PNG，新增工作台运行截图。

验证结果：

- `python -m py_compile app.py llm\quality_report.py models\infer_neu_cls.py models\batch_infer_neu_cls.py models\neu_cls_model.py models\train_neu_cls.py data\preprocess_neu_cls.py` 通过。
- `python models\infer_neu_cls.py --image data\raw\NEU-CLS\valid\valid\images\crazing_1.jpg` 通过，输出龟裂分类结果。
- `build_batch_summary_markdown()` 构造样例测试通过。
- Streamlit 服务 `http://localhost:8501` 返回 200 OK。
- 已生成 `05_补充材料/运行截图/streamlit_detection_workbench.png`。
- 已重新导出 `02_项目主报告/output/钢材表面缺陷智能质检与大模型诊断报告系统.docx`。
- 已重新导出 `02_项目主报告/output/钢材表面缺陷智能质检与大模型诊断报告系统.pdf`。
- PDF 共 10 页，已渲染检查第 1、4、5、6、7、10 页，无明显空白、乱码或截图截断。

遗留事项：

- 团队成员、学号、班级、团队编号和提交日期仍需人工补充。
- 最终提交压缩包和演示视频仍未处理，按用户要求暂不进入该阶段。

## 2026-06-24 本地 Skill 与 GitHub 发布

操作者：Codex

目标：将当前人工智能基础课程大作业工作流沉淀为本地 Codex skill，并把项目推送到 GitHub 公开仓库，同时启用 GitHub Pages 静态展示页。

主要变更：

- 新增本地 skill：`C:\Users\Zicheng Wang\.codex\skills\ai-course-project-workflow`。
- skill 包含 `SKILL.md`、`references/assignment_workflow.md`、`references/github_pages.md`、`scripts/init_ai_course_project.py`。
- 已用 `quick_validate.py` 校验 skill，通过。
- 新增 `data/samples/` 六张轻量示例图和 `samples.csv`，便于公开仓库克隆后直接演示。
- 新增 `.gitignore`，排除完整原始数据集、压缩包、缓存、本地 agent/Codex 目录和原始课程通知。
- 新增 `docs/index.html` 与 `docs/assets/`，用于 GitHub Pages 静态展示。

GitHub：

- 公开仓库：`https://github.com/wzgig/ai-course-steel-defect-qc`
- GitHub Pages：`https://wzgig.github.io/ai-course-steel-defect-qc/`
- Pages 来源：`main` 分支 `/docs` 目录。
- Pages 状态：`built`，访问返回 200 OK。

说明：

- GitHub Pages 只能托管静态展示页，不能运行 Streamlit 服务。
- 交互式网站仍需本地运行 `python -m streamlit run app.py`。
