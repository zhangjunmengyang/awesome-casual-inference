"""
实验结果分析模块

功能：
-----
1. 统计检验：t检验、z检验、卡方检验
2. 置信区间：效应估计的不确定性
3. 方差缩减：CUPED、分层等技术
4. 多重检验：Bonferroni、FDR 校正
5. 异质性分析：分群效应分析

面试考点：
---------
- t检验和z检验的区别？
- 什么是 CUPED？
- 如何处理多重检验？
- 什么是 SRM (Sample Ratio Mismatch)？
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class ExperimentResult:
    """实验结果"""
    metric_name: str
    control_mean: float
    treatment_mean: float
    absolute_effect: float
    relative_effect: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    is_significant: bool
    sample_size_control: int
    sample_size_treatment: int


def generate_experiment_data(
    n_control: int = 10000,
    n_treatment: int = 10000,
    baseline_rate: float = 0.05,
    true_effect: float = 0.10,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成实验数据

    Parameters:
    -----------
    n_control: 对照组样本量
    n_treatment: 实验组样本量
    baseline_rate: 基线转化率
    true_effect: 真实效应（相对提升）
    seed: 随机种子

    Returns:
    --------
    实验数据 DataFrame
    """
    np.random.seed(seed)

    # 用户特征
    n_total = n_control + n_treatment

    user_activity = np.random.beta(2, 5, n_total)
    historical_conversion = np.random.beta(2, 8, n_total)
    age = np.random.normal(35, 10, n_total).clip(18, 65)
    is_mobile = np.random.binomial(1, 0.6, n_total)

    # 分组（随机分配）
    group = np.array(['control'] * n_control + ['treatment'] * n_treatment)
    np.random.shuffle(group)
    is_treatment = (group == 'treatment').astype(int)

    # 基线转化概率
    base_prob = 1 / (1 + np.exp(-(
        -3 + 2 * user_activity + 1.5 * historical_conversion + 0.01 * (age - 35)
    )))

    # 实验效应
    treatment_effect = true_effect * baseline_rate

    # 实际转化概率
    prob = base_prob + is_treatment * treatment_effect
    prob = np.clip(prob, 0, 1)

    # 转化
    converted = np.random.binomial(1, prob)

    # 收入（条件于转化）
    revenue = np.where(
        converted == 1,
        np.random.lognormal(4, 0.8, n_total),
        0
    )

    return pd.DataFrame({
        'user_id': range(n_total),
        'group': group,
        'is_treatment': is_treatment,
        'user_activity': user_activity,
        'historical_conversion': historical_conversion,
        'age': age,
        'is_mobile': is_mobile,
        'converted': converted,
        'revenue': revenue,
        '_base_prob': base_prob,
    })


class ExperimentAnalyzer:
    """实验分析器"""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.results = {}

    def analyze_proportion(
        self,
        df: pd.DataFrame,
        metric_col: str,
        group_col: str = 'group'
    ) -> ExperimentResult:
        """
        分析比例指标（如转化率）

        使用两比例 z 检验
        """
        control = df[df[group_col] == 'control'][metric_col]
        treatment = df[df[group_col] == 'treatment'][metric_col]

        n_c, n_t = len(control), len(treatment)
        p_c, p_t = control.mean(), treatment.mean()

        # 效应
        absolute_effect = p_t - p_c
        relative_effect = absolute_effect / p_c if p_c > 0 else 0

        # 标准误（使用分组方差）
        se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)

        # z 检验
        z_stat = absolute_effect / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # 置信区间
        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = absolute_effect - z_crit * se
        ci_upper = absolute_effect + z_crit * se

        result = ExperimentResult(
            metric_name=metric_col,
            control_mean=p_c,
            treatment_mean=p_t,
            absolute_effect=absolute_effect,
            relative_effect=relative_effect,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            sample_size_control=n_c,
            sample_size_treatment=n_t
        )

        self.results[metric_col] = result
        return result

    def analyze_continuous(
        self,
        df: pd.DataFrame,
        metric_col: str,
        group_col: str = 'group'
    ) -> ExperimentResult:
        """
        分析连续指标（如收入）

        使用 Welch's t 检验
        """
        control = df[df[group_col] == 'control'][metric_col]
        treatment = df[df[group_col] == 'treatment'][metric_col]

        n_c, n_t = len(control), len(treatment)
        mean_c, mean_t = control.mean(), treatment.mean()
        var_c, var_t = control.var(), treatment.var()

        # 效应
        absolute_effect = mean_t - mean_c
        relative_effect = absolute_effect / mean_c if mean_c > 0 else 0

        # 标准误
        se = np.sqrt(var_c / n_c + var_t / n_t)

        # Welch's t 检验
        t_stat = absolute_effect / se
        # 自由度（Welch-Satterthwaite）
        df_welch = (var_c / n_c + var_t / n_t) ** 2 / (
            (var_c / n_c) ** 2 / (n_c - 1) + (var_t / n_t) ** 2 / (n_t - 1)
        )
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_welch))

        # 置信区间
        t_crit = stats.t.ppf(1 - self.alpha / 2, df_welch)
        ci_lower = absolute_effect - t_crit * se
        ci_upper = absolute_effect + t_crit * se

        result = ExperimentResult(
            metric_name=metric_col,
            control_mean=mean_c,
            treatment_mean=mean_t,
            absolute_effect=absolute_effect,
            relative_effect=relative_effect,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            sample_size_control=n_c,
            sample_size_treatment=n_t
        )

        self.results[metric_col] = result
        return result

    def analyze_with_cuped(
        self,
        df: pd.DataFrame,
        metric_col: str,
        covariate_col: str,
        group_col: str = 'group'
    ) -> Tuple[ExperimentResult, float]:
        """
        使用 CUPED 进行方差缩减

        CUPED (Controlled-experiment Using Pre-Experiment Data)
        Y_adj = Y - θ(X - X̄)
        其中 θ = Cov(Y, X) / Var(X)

        Parameters:
        -----------
        metric_col: 目标指标
        covariate_col: 协变量（如实验前数据）

        Returns:
        --------
        (调整后的结果, 方差缩减比例)
        """
        df = df.copy()

        # 计算 CUPED 调整
        Y = df[metric_col].values
        X = df[covariate_col].values

        theta = np.cov(Y, X)[0, 1] / np.var(X)
        Y_adjusted = Y - theta * (X - X.mean())

        df['_cuped_adjusted'] = Y_adjusted

        # 原始方差
        original_var = Y.var()

        # 调整后方差
        adjusted_var = Y_adjusted.var()

        # 方差缩减比例
        variance_reduction = 1 - adjusted_var / original_var

        # 分析调整后的指标
        result = self.analyze_continuous(df, '_cuped_adjusted', group_col)
        result.metric_name = f"{metric_col} (CUPED)"

        return result, variance_reduction


def check_srm(df: pd.DataFrame, group_col: str = 'group',
              expected_ratio: float = 0.5) -> Dict:
    """
    检查样本比例不匹配 (Sample Ratio Mismatch)

    SRM 是实验分流问题的信号，可能导致结果偏差

    Parameters:
    -----------
    df: 实验数据
    group_col: 分组列
    expected_ratio: 期望的实验组比例

    Returns:
    --------
    SRM 检查结果
    """
    n_treatment = (df[group_col] == 'treatment').sum()
    n_control = (df[group_col] == 'control').sum()
    n_total = n_treatment + n_control

    observed_ratio = n_treatment / n_total

    # 二项检验
    p_value = stats.binom_test(n_treatment, n_total, expected_ratio)

    # 或使用卡方检验
    expected_treatment = n_total * expected_ratio
    expected_control = n_total * (1 - expected_ratio)
    chi2, chi2_p = stats.chisquare([n_treatment, n_control],
                                   [expected_treatment, expected_control])

    has_srm = p_value < 0.001  # 使用严格阈值

    return {
        'n_treatment': n_treatment,
        'n_control': n_control,
        'observed_ratio': observed_ratio,
        'expected_ratio': expected_ratio,
        'p_value': p_value,
        'chi2_p_value': chi2_p,
        'has_srm': has_srm,
        'warning': "检测到 SRM！实验结果可能不可信。请检查分流逻辑。" if has_srm else None
    }


def multiple_testing_correction(
    p_values: List[float],
    method: str = 'bonferroni'
) -> Tuple[List[float], List[bool]]:
    """
    多重检验校正

    Parameters:
    -----------
    p_values: 原始 p 值列表
    method: 校正方法
        - 'bonferroni': Bonferroni 校正
        - 'holm': Holm-Bonferroni 校正
        - 'fdr': Benjamini-Hochberg FDR 控制

    Returns:
    --------
    (校正后的 p 值或阈值, 是否显著)
    """
    n = len(p_values)
    alpha = 0.05

    if method == 'bonferroni':
        # 简单校正：α' = α/n
        adjusted_alpha = alpha / n
        adjusted_p = [min(p * n, 1.0) for p in p_values]
        significant = [p < adjusted_alpha for p in p_values]

    elif method == 'holm':
        # Holm-Bonferroni: 阶梯校正
        sorted_indices = np.argsort(p_values)
        adjusted_p = [0.0] * n
        significant = [False] * n

        for rank, idx in enumerate(sorted_indices):
            threshold = alpha / (n - rank)
            if p_values[idx] < threshold:
                significant[idx] = True
                adjusted_p[idx] = p_values[idx] * (n - rank)
            else:
                adjusted_p[idx] = min(p_values[idx] * (n - rank), 1.0)
                # 一旦不显著，后续都不显著
                for later_idx in sorted_indices[rank:]:
                    significant[later_idx] = False
                break

    elif method == 'fdr':
        # Benjamini-Hochberg FDR
        sorted_indices = np.argsort(p_values)
        adjusted_p = [0.0] * n
        significant = [False] * n

        for rank, idx in enumerate(sorted_indices):
            threshold = (rank + 1) * alpha / n
            adjusted_p[idx] = p_values[idx] * n / (rank + 1)
            significant[idx] = p_values[idx] <= threshold

        # 调整：确保单调性
        for i in range(n - 2, -1, -1):
            idx = sorted_indices[i]
            next_idx = sorted_indices[i + 1]
            adjusted_p[idx] = min(adjusted_p[idx], adjusted_p[next_idx])

    else:
        raise ValueError(f"Unknown method: {method}")

    return adjusted_p, significant


def heterogeneity_analysis(
    df: pd.DataFrame,
    metric_col: str,
    segment_col: str,
    group_col: str = 'group'
) -> pd.DataFrame:
    """
    异质性分析：分群效应分析

    Parameters:
    -----------
    df: 实验数据
    metric_col: 指标列
    segment_col: 分群列
    group_col: 分组列

    Returns:
    --------
    各分群的效应
    """
    results = []

    for segment in df[segment_col].unique():
        segment_df = df[df[segment_col] == segment]

        control = segment_df[segment_df[group_col] == 'control'][metric_col]
        treatment = segment_df[segment_df[group_col] == 'treatment'][metric_col]

        if len(control) < 30 or len(treatment) < 30:
            continue

        mean_c, mean_t = control.mean(), treatment.mean()
        effect = mean_t - mean_c
        relative_effect = effect / mean_c if mean_c > 0 else 0

        # 简化的标准误
        se = np.sqrt(control.var() / len(control) + treatment.var() / len(treatment))

        results.append({
            'segment': segment,
            'n_control': len(control),
            'n_treatment': len(treatment),
            'control_mean': mean_c,
            'treatment_mean': mean_t,
            'absolute_effect': effect,
            'relative_effect': relative_effect,
            'se': se,
            'ci_lower': effect - 1.96 * se,
            'ci_upper': effect + 1.96 * se
        })

    return pd.DataFrame(results)


def plot_experiment_results(results: List[ExperimentResult]) -> go.Figure:
    """绘制实验结果"""
    fig = go.Figure()

    metrics = [r.metric_name for r in results]
    effects = [r.relative_effect * 100 for r in results]
    ci_lower = [(r.ci_lower / r.control_mean * 100 if r.control_mean > 0 else 0) for r in results]
    ci_upper = [(r.ci_upper / r.control_mean * 100 if r.control_mean > 0 else 0) for r in results]
    colors = ['#27AE60' if r.is_significant and r.relative_effect > 0
              else '#EB5757' if r.is_significant
              else '#6B7280' for r in results]

    fig.add_trace(go.Bar(
        x=metrics,
        y=effects,
        error_y=dict(
            type='data',
            symmetric=False,
            array=[u - e for u, e in zip(ci_upper, effects)],
            arrayminus=[e - l for e, l in zip(effects, ci_lower)]
        ),
        marker_color=colors,
        text=[f'{e:.1f}%' for e in effects],
        textposition='outside'
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title='实验结果概览',
        xaxis_title='指标',
        yaxis_title='相对变化 (%)',
        template='plotly_white',
        height=400
    )

    return fig


def render():
    """渲染 Gradio 界面"""
    import gradio as gr

    with gr.Blocks() as block:
        gr.Markdown("""
## 实验结果分析

### 分析流程

1. **SRM 检查**：验证分流是否正常
2. **统计检验**：计算效应和显著性
3. **置信区间**：量化不确定性
4. **多重检验校正**：控制假阳性
5. **异质性分析**：分群效应

---
        """)

        with gr.Row():
            with gr.Column():
                n_control = gr.Number(value=10000, label="对照组样本量", precision=0)
                n_treatment = gr.Number(value=10000, label="实验组样本量", precision=0)
                baseline_rate = gr.Number(value=5.0, label="基线转化率 (%)")
                true_effect = gr.Number(value=10.0, label="模拟效应 (相对提升 %)")
                seed = gr.Number(value=42, label="随机种子", precision=0)

            with gr.Column():
                run_btn = gr.Button("生成数据并分析", variant="primary")
                srm_output = gr.Markdown(label="SRM 检查")

        with gr.Row():
            plot_output = gr.Plot()

        with gr.Row():
            result_output = gr.Markdown()

        def run_analysis(n_control, n_treatment, baseline_rate, true_effect, seed):
            # 生成数据
            df = generate_experiment_data(
                int(n_control), int(n_treatment),
                baseline_rate / 100, true_effect / 100, int(seed)
            )

            # SRM 检查
            srm = check_srm(df)
            srm_md = f"""
### SRM 检查结果

| 指标 | 值 |
|-----|-----|
| 对照组 | {srm['n_control']:,} |
| 实验组 | {srm['n_treatment']:,} |
| 观测比例 | {srm['observed_ratio']:.4f} |
| 期望比例 | {srm['expected_ratio']:.4f} |
| p-value | {srm['p_value']:.4f} |
| **状态** | {'⚠️ 检测到 SRM!' if srm['has_srm'] else '✓ 正常'} |
"""

            # 分析
            analyzer = ExperimentAnalyzer(alpha=0.05)
            conversion_result = analyzer.analyze_proportion(df, 'converted')
            revenue_result = analyzer.analyze_continuous(df, 'revenue')

            # CUPED
            cuped_result, var_reduction = analyzer.analyze_with_cuped(
                df, 'converted', 'historical_conversion'
            )

            results = [conversion_result, revenue_result]

            # 可视化
            fig = plot_experiment_results(results)

            # 结果报告
            report = f"""
### 实验结果

#### 转化率
| 指标 | 值 |
|-----|-----|
| 对照组 | {conversion_result.control_mean*100:.2f}% |
| 实验组 | {conversion_result.treatment_mean*100:.2f}% |
| 绝对提升 | {conversion_result.absolute_effect*100:.2f}% |
| 相对提升 | {conversion_result.relative_effect*100:.1f}% |
| 95% CI | [{conversion_result.ci_lower*100:.2f}%, {conversion_result.ci_upper*100:.2f}%] |
| p-value | {conversion_result.p_value:.4f} |
| **显著性** | {'✓ 显著' if conversion_result.is_significant else '✗ 不显著'} |

#### 人均收入
| 指标 | 值 |
|-----|-----|
| 对照组 | ¥{revenue_result.control_mean:.2f} |
| 实验组 | ¥{revenue_result.treatment_mean:.2f} |
| 绝对提升 | ¥{revenue_result.absolute_effect:.2f} |
| 相对提升 | {revenue_result.relative_effect*100:.1f}% |
| p-value | {revenue_result.p_value:.4f} |
| **显著性** | {'✓ 显著' if revenue_result.is_significant else '✗ 不显著'} |

#### CUPED 方差缩减
- 使用历史转化率作为协变量
- 方差缩减比例: **{var_reduction*100:.1f}%**
- 这意味着达到同样功效只需要 **{(1-var_reduction)*100:.0f}%** 的样本量
"""

            return srm_md, fig, report

        run_btn.click(
            fn=run_analysis,
            inputs=[n_control, n_treatment, baseline_rate, true_effect, seed],
            outputs=[srm_output, plot_output, result_output]
        )

        gr.Markdown("""
---

### 面试常见问题

**Q1: 什么是 SRM？为什么重要？**
> Sample Ratio Mismatch：实际分流比例与期望不符
> 原因：分流 bug、用户行为差异、数据管道问题
> 危害：结果有偏，不可信

**Q2: t检验和z检验区别？**
> - z检验：已知总体方差或大样本（n>30）
> - t检验：未知总体方差，用样本方差估计
> 实践中：大样本时两者结果接近

**Q3: 什么是 CUPED？**
> Controlled-experiment Using Pre-Experiment Data
> 用实验前数据减少方差：Y_adj = Y - θ(X - X̄)
> 效果：减少 20-50% 方差，等效增加样本量

**Q4: 多重检验问题？**
> 看 k 个指标，整体假阳性率 = 1-(1-α)^k
> 看 20 个指标，假阳性率 > 60%！
> 解决：Bonferroni、FDR 校正
        """)

    return None


if __name__ == "__main__":
    # 测试
    df = generate_experiment_data()
    analyzer = ExperimentAnalyzer()
    result = analyzer.analyze_proportion(df, 'converted')
    print(f"转化率提升: {result.relative_effect*100:.1f}%")
    print(f"p-value: {result.p_value:.4f}")
    print(f"显著: {result.is_significant}")
