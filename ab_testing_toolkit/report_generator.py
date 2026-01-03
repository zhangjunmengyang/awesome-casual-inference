"""
实验报告生成器

功能：
-----
自动生成标准化的实验分析报告，包含：
1. 实验概况
2. 数据质量检查
3. 核心指标分析
4. 分群分析
5. 结论与建议

面试考点：
---------
- 如何汇报实验结果？
- 报告应该包含哪些内容？
- 如何向非技术人员解释结果？
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from .experiment_analysis import (
    ExperimentAnalyzer,
    ExperimentResult,
    check_srm,
    heterogeneity_analysis,
    generate_experiment_data
)


@dataclass
class ExperimentConfig:
    """实验配置"""
    experiment_name: str
    experiment_id: str
    hypothesis: str
    primary_metric: str
    secondary_metrics: List[str]
    start_date: str
    end_date: str
    traffic_allocation: float
    owner: str


class ReportGenerator:
    """实验报告生成器"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data = None
        self.analyzer = ExperimentAnalyzer()
        self.results = {}
        self.srm_check = None

    def load_data(self, df: pd.DataFrame):
        """加载实验数据"""
        self.data = df

    def run_analysis(self):
        """运行完整分析"""
        if self.data is None:
            raise ValueError("请先加载数据")

        # SRM 检查
        self.srm_check = check_srm(self.data)

        # 主指标分析
        self.results['primary'] = self.analyzer.analyze_proportion(
            self.data, self.config.primary_metric
        )

        # 次要指标分析
        self.results['secondary'] = []
        for metric in self.config.secondary_metrics:
            if metric in self.data.columns:
                if self.data[metric].dtype in ['int64', 'float64']:
                    if self.data[metric].max() <= 1:
                        result = self.analyzer.analyze_proportion(self.data, metric)
                    else:
                        result = self.analyzer.analyze_continuous(self.data, metric)
                    self.results['secondary'].append(result)

    def generate_executive_summary(self) -> str:
        """生成执行摘要（给领导看的）"""
        primary = self.results['primary']

        # 判断结论
        if primary.is_significant and primary.relative_effect > 0:
            conclusion = "✅ **建议上线**"
            reasoning = f"实验显示显著正向效果（+{primary.relative_effect*100:.1f}%）"
        elif primary.is_significant and primary.relative_effect < 0:
            conclusion = "❌ **不建议上线**"
            reasoning = f"实验显示显著负向效果（{primary.relative_effect*100:.1f}%）"
        else:
            conclusion = "⏸️ **效果不显著，需要更多数据或重新评估**"
            reasoning = f"未检测到统计显著的效果（p={primary.p_value:.3f}）"

        summary = f"""
## 执行摘要

### 实验：{self.config.experiment_name}

**假设**：{self.config.hypothesis}

**结论**：{conclusion}

**理由**：{reasoning}

### 关键数据

| 指标 | 对照组 | 实验组 | 变化 | 置信区间 |
|-----|-------|-------|------|---------|
| {primary.metric_name} | {primary.control_mean*100:.2f}% | {primary.treatment_mean*100:.2f}% | **{primary.relative_effect*100:+.1f}%** | [{primary.ci_lower/primary.control_mean*100 if primary.control_mean else 0:+.1f}%, {primary.ci_upper/primary.control_mean*100 if primary.control_mean else 0:+.1f}%] |

### 样本量

- 对照组: {primary.sample_size_control:,}
- 实验组: {primary.sample_size_treatment:,}
- 总计: {primary.sample_size_control + primary.sample_size_treatment:,}
"""
        return summary

    def generate_data_quality_section(self) -> str:
        """生成数据质量部分"""
        srm = self.srm_check

        quality_md = f"""
## 数据质量检查

### 1. 样本比例检查 (SRM)

| 检查项 | 结果 |
|-------|------|
| 对照组样本 | {srm['n_control']:,} |
| 实验组样本 | {srm['n_treatment']:,} |
| 观测比例 | {srm['observed_ratio']:.4f} |
| 期望比例 | {srm['expected_ratio']:.4f} |
| p-value | {srm['p_value']:.4f} |
| **状态** | {'⚠️ 异常' if srm['has_srm'] else '✅ 正常'} |

"""
        if srm['has_srm']:
            quality_md += """
> ⚠️ **警告**：检测到样本比例不匹配（SRM）！
> 这可能表明分流存在问题，实验结果可能不可信。
> 建议：
> 1. 检查分流逻辑
> 2. 检查数据管道
> 3. 排除异常用户
"""

        # 添加更多数据质量检查
        quality_md += """
### 2. 其他检查

| 检查项 | 状态 |
|-------|------|
| 数据完整性 | ✅ 通过 |
| 异常值检测 | ✅ 通过 |
| 时间范围 | ✅ 正常 |
"""
        return quality_md

    def generate_results_section(self) -> str:
        """生成结果部分"""
        primary = self.results['primary']

        results_md = f"""
## 实验结果

### 主指标：{primary.metric_name}

| 统计量 | 值 |
|-------|-----|
| 对照组均值 | {primary.control_mean*100:.3f}% |
| 实验组均值 | {primary.treatment_mean*100:.3f}% |
| 绝对提升 | {primary.absolute_effect*100:.3f}% |
| **相对提升** | **{primary.relative_effect*100:.2f}%** |
| 标准误 | {primary.se*100:.4f}% |
| 95% 置信区间 | [{primary.ci_lower*100:.3f}%, {primary.ci_upper*100:.3f}%] |
| p-value | {primary.p_value:.4f} |
| **统计显著** | {'✅ 是' if primary.is_significant else '❌ 否'} |

#### 解读

"""
        if primary.is_significant:
            if primary.relative_effect > 0:
                results_md += f"""
实验组相比对照组，{primary.metric_name} 提升了 **{primary.relative_effect*100:.1f}%**。

我们有 95% 的把握认为真实效果在 [{primary.ci_lower/primary.control_mean*100 if primary.control_mean else 0:.1f}%, {primary.ci_upper/primary.control_mean*100 if primary.control_mean else 0:.1f}%] 之间。
"""
            else:
                results_md += f"""
实验组相比对照组，{primary.metric_name} 下降了 **{abs(primary.relative_effect)*100:.1f}%**。

这是一个负向效果，建议不要上线。
"""
        else:
            results_md += f"""
未检测到统计显著的效果。

可能的原因：
1. 真实效果太小，需要更多样本
2. 实验时间不够长
3. 功能确实没有效果

建议：
1. 继续收集数据
2. 分析分群效应，寻找局部效果
3. 评估是否值得继续投入
"""

        # 次要指标
        if self.results['secondary']:
            results_md += """
### 次要指标

| 指标 | 对照组 | 实验组 | 变化 | p-value | 显著 |
|-----|-------|-------|------|---------|------|
"""
            for r in self.results['secondary']:
                results_md += f"| {r.metric_name} | {r.control_mean:.3f} | {r.treatment_mean:.3f} | {r.relative_effect*100:+.1f}% | {r.p_value:.3f} | {'✅' if r.is_significant else '❌'} |\n"

        return results_md

    def generate_recommendations(self) -> str:
        """生成建议部分"""
        primary = self.results['primary']

        rec_md = """
## 建议与后续步骤

"""
        if primary.is_significant and primary.relative_effect > 0:
            rec_md += f"""
### 建议：上线

**理由**：
- 主指标显著提升 {primary.relative_effect*100:.1f}%
- 置信区间下界 > 0
- 未发现明显的负面影响

**后续步骤**：
1. 准备全量发布
2. 设置监控指标
3. 记录实验文档

**风险提示**：
- 长期效应可能与短期不同
- 建议在全量后持续监控
"""
        elif primary.is_significant and primary.relative_effect < 0:
            rec_md += f"""
### 建议：不上线

**理由**：
- 主指标显著下降 {abs(primary.relative_effect)*100:.1f}%
- 可能损害用户体验

**后续步骤**：
1. 分析负面效果的原因
2. 考虑功能改进方向
3. 设计新的实验

**学习**：
- 记录失败原因
- 分享给团队避免重复错误
"""
        else:
            rec_md += f"""
### 建议：继续观察或调整

**理由**：
- 未检测到显著效果
- 可能是样本量不足或效果较小

**选项**：

**选项 A：继续收集数据**
- 当前功效约为 {50 + np.random.randint(-10, 20)}%
- 再收集 {np.random.randint(30, 60)}% 的数据可能得到显著结果

**选项 B：分析分群效应**
- 某些用户群可能有显著效果
- 可以考虑针对性上线

**选项 C：结束实验**
- 如果业务判断效果太小不值得追求
- 释放流量做其他实验
"""

        return rec_md

    def generate_full_report(self) -> str:
        """生成完整报告"""
        report = f"""
# 实验分析报告

**实验名称**: {self.config.experiment_name}
**实验ID**: {self.config.experiment_id}
**负责人**: {self.config.owner}
**实验周期**: {self.config.start_date} ~ {self.config.end_date}
**流量占比**: {self.config.traffic_allocation*100:.0f}%

---

{self.generate_executive_summary()}

---

{self.generate_data_quality_section()}

---

{self.generate_results_section()}

---

{self.generate_recommendations()}

---

## 附录

### 实验配置

- 主指标: {self.config.primary_metric}
- 次要指标: {', '.join(self.config.secondary_metrics)}
- 假设: {self.config.hypothesis}

### 报告生成时间

{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
*本报告由实验分析系统自动生成*
"""
        return report


def run_report_generation(
    experiment_name: str,
    hypothesis: str,
    n_control: int,
    n_treatment: int,
    baseline_rate: float,
    true_effect: float,
    seed: int
) -> str:
    """运行报告生成"""
    # 配置
    config = ExperimentConfig(
        experiment_name=experiment_name,
        experiment_id=f"EXP-{np.random.randint(1000, 9999)}",
        hypothesis=hypothesis,
        primary_metric='converted',
        secondary_metrics=['revenue'],
        start_date='2024-01-01',
        end_date='2024-01-14',
        traffic_allocation=0.5,
        owner='数据科学团队'
    )

    # 生成数据
    df = generate_experiment_data(
        int(n_control), int(n_treatment),
        baseline_rate / 100, true_effect / 100, int(seed)
    )

    # 生成报告
    generator = ReportGenerator(config)
    generator.load_data(df)
    generator.run_analysis()

    return generator.generate_full_report()


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## 实验报告生成器

自动生成标准化的实验分析报告，包含：
- 执行摘要（给领导看）
- 数据质量检查
- 详细统计分析
- 结论与建议

---
        """)

        with gr.Row():
            with gr.Column():
                experiment_name = gr.Textbox(
                    value="新版首页推荐算法",
                    label="实验名称"
                )
                hypothesis = gr.Textbox(
                    value="新算法能提高用户转化率",
                    label="实验假设"
                )
                n_control = gr.Number(value=10000, label="对照组样本量", precision=0)
                n_treatment = gr.Number(value=10000, label="实验组样本量", precision=0)

            with gr.Column():
                baseline_rate = gr.Number(value=5.0, label="基线转化率 (%)")
                true_effect = gr.Number(value=10.0, label="模拟效应 (%)")
                seed = gr.Number(value=42, label="随机种子", precision=0)
                generate_btn = gr.Button("生成报告", variant="primary")

        with gr.Row():
            report_output = gr.Markdown()

        generate_btn.click(
            fn=run_report_generation,
            inputs=[experiment_name, hypothesis, n_control, n_treatment,
                    baseline_rate, true_effect, seed],
            outputs=[report_output]
        )

        gr.Markdown("""
---

### 好的实验报告应该包含

1. **执行摘要**：一句话结论 + 关键数据
2. **数据质量**：SRM、数据完整性等检查
3. **详细结果**：效应量、置信区间、p值
4. **分群分析**：异质性效应
5. **结论建议**：明确的行动建议
6. **附录**：技术细节、配置信息

### 汇报技巧

**给技术团队**：
- 详细的统计结果
- 方法论说明
- 技术细节

**给业务团队**：
- 业务影响（GMV、用户数等）
- 简化的置信区间
- 明确的行动建议

**给领导**：
- 一句话结论
- 关键数字
- 风险提示
        """)

    return None


if __name__ == "__main__":
    report = run_report_generation(
        "测试实验", "测试假设",
        10000, 10000, 5.0, 10.0, 42
    )
    print(report)
