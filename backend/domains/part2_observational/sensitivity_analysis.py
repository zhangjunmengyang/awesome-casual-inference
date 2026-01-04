"""
敏感性分析模块

评估未测量混淆对因果估计的影响:
- Rosenbaum Bounds (匹配数据)
- E-value (一般因果估计)
- 混淆偏差分析
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, Optional


class RosenbaumBounds:
    """
    Rosenbaum Bounds 敏感性分析

    用于评估匹配后的因果估计对未测量混淆的敏感性

    Rosenbaum (2002): "Observational Studies"

    核心思想:
    - Γ (Gamma): 隐藏偏差参数，表示两个样本的处理概率比值的上界
    - Γ = 1: 无隐藏偏差 (完美匹配)
    - Γ > 1: 允许一定程度的隐藏偏差
    """

    def __init__(self):
        self.gamma_values = None
        self.p_values_plus = None
        self.p_values_minus = None

    def analyze(
        self,
        matched_pairs_diff: np.ndarray,
        gamma_range: Tuple[float, float] = (1.0, 3.0),
        n_gamma: int = 20
    ) -> Dict:
        """
        执行 Rosenbaum Bounds 分析

        Parameters:
        -----------
        matched_pairs_diff: 匹配对的差异 (处理组 - 控制组)
        gamma_range: Γ 的范围
        n_gamma: Γ 的取值数量

        Returns:
        --------
        dict with sensitivity analysis results
        """
        n_pairs = len(matched_pairs_diff)

        # 计算符号秩 (Wilcoxon signed-rank test 的统计量)
        abs_diff = np.abs(matched_pairs_diff)
        ranks = stats.rankdata(abs_diff)
        signs = np.sign(matched_pairs_diff)

        T_plus = np.sum(ranks[signs > 0])  # 正差异的秩和
        T_minus = np.sum(ranks[signs < 0])  # 负差异的秩和

        # Γ 值
        self.gamma_values = np.linspace(gamma_range[0], gamma_range[1], n_gamma)

        self.p_values_plus = []
        self.p_values_minus = []

        for gamma in self.gamma_values:
            # 在 Γ 假设下，计算 p-value 的上界和下界

            # 期望和方差 (在无效假设下)
            # 当 Γ = 1 时，退化为标准 Wilcoxon 检验
            n_total = n_pairs * (n_pairs + 1) / 2
            p_i_plus = gamma / (1 + gamma)  # 上界概率
            p_i_minus = 1 / (1 + gamma)     # 下界概率

            # 上界
            E_plus = n_total * p_i_plus
            var_plus = n_total * p_i_plus * (1 - p_i_plus)
            z_plus = (T_plus - E_plus) / np.sqrt(var_plus)
            p_plus = 1 - stats.norm.cdf(z_plus)

            # 下界
            E_minus = n_total * p_i_minus
            var_minus = n_total * p_i_minus * (1 - p_i_minus)
            z_minus = (T_plus - E_minus) / np.sqrt(var_minus)
            p_minus = 1 - stats.norm.cdf(z_minus)

            self.p_values_plus.append(p_plus)
            self.p_values_minus.append(p_minus)

        self.p_values_plus = np.array(self.p_values_plus)
        self.p_values_minus = np.array(self.p_values_minus)

        # 找到临界 Γ 值 (p-value 上界 = 0.05)
        critical_gamma_idx = np.where(self.p_values_plus >= 0.05)[0]
        if len(critical_gamma_idx) > 0:
            critical_gamma = self.gamma_values[critical_gamma_idx[0]]
        else:
            critical_gamma = self.gamma_values[-1]

        return {
            'gamma_values': self.gamma_values.tolist(),
            'p_values_plus': self.p_values_plus.tolist(),
            'p_values_minus': self.p_values_minus.tolist(),
            'critical_gamma': float(critical_gamma),
            'interpretation': self._interpret_gamma(critical_gamma)
        }

    def _interpret_gamma(self, gamma: float) -> str:
        """解释 Γ 值"""
        if gamma < 1.5:
            return "结果对未测量混淆非常敏感"
        elif gamma < 2.0:
            return "结果对未测量混淆较敏感"
        elif gamma < 3.0:
            return "结果对未测量混淆有一定鲁棒性"
        else:
            return "结果对未测量混淆具有强鲁棒性"


class EValueAnalysis:
    """
    E-value 分析

    VanderWeele & Ding (2017): "Sensitivity Analysis in Observational Research"

    E-value: 使观测到的关联消失所需的最小混淆强度

    适用于任何因果估计 (不限于匹配)
    """

    def __init__(self):
        pass

    def compute_evalue(
        self,
        effect_estimate: float,
        effect_type: str = 'rate_ratio',
        ci_lower: Optional[float] = None
    ) -> Dict:
        """
        计算 E-value

        Parameters:
        -----------
        effect_estimate: 效应估计 (比率或差异)
        effect_type: 效应类型 ('rate_ratio', 'odds_ratio', 'mean_difference')
        ci_lower: 置信区间下界 (可选)

        Returns:
        --------
        dict with E-value results
        """
        if effect_type in ['rate_ratio', 'odds_ratio']:
            # 对于比率
            RR = np.abs(effect_estimate)

            if RR < 1:
                RR = 1 / RR

            # E-value 公式
            e_value = RR + np.sqrt(RR * (RR - 1))

            # 如果有置信区间，计算 CI 的 E-value
            if ci_lower is not None:
                RR_ci = np.abs(ci_lower)
                if RR_ci < 1:
                    RR_ci = 1 / RR_ci

                e_value_ci = RR_ci + np.sqrt(RR_ci * (RR_ci - 1))
            else:
                e_value_ci = None

        elif effect_type == 'mean_difference':
            # 对于均值差异，需要转换为标准化效应
            # 这里假设已经标准化
            standardized_diff = np.abs(effect_estimate)

            # 近似转换为 RR
            RR = np.exp(0.91 * standardized_diff)
            e_value = RR + np.sqrt(RR * (RR - 1))

            if ci_lower is not None:
                standardized_diff_ci = np.abs(ci_lower)
                RR_ci = np.exp(0.91 * standardized_diff_ci)
                e_value_ci = RR_ci + np.sqrt(RR_ci * (RR_ci - 1))
            else:
                e_value_ci = None

        else:
            raise ValueError(f"Unknown effect_type: {effect_type}")

        return {
            'e_value': float(e_value),
            'e_value_ci': float(e_value_ci) if e_value_ci is not None else None,
            'interpretation': self._interpret_evalue(e_value),
            'required_confounder_strength': self._describe_confounder(e_value)
        }

    def _interpret_evalue(self, e_value: float) -> str:
        """解释 E-value"""
        if e_value < 1.5:
            return "结果对未测量混淆非常敏感，需要很弱的混淆就能消除效应"
        elif e_value < 2.0:
            return "结果对未测量混淆较敏感"
        elif e_value < 3.0:
            return "结果对未测量混淆有一定鲁棒性"
        elif e_value < 5.0:
            return "结果对未测量混淆具有较强鲁棒性"
        else:
            return "结果对未测量混淆具有很强鲁棒性"

    def _describe_confounder(self, e_value: float) -> str:
        """描述所需的混淆强度"""
        return (
            f"未测量混淆需要同时与处理和结果有 {e_value:.2f} 倍的关联强度，"
            f"才能完全解释观测到的效应"
        )


class ConfoundingBiasAnalysis:
    """
    混淆偏差分析

    使用简化的偏差公式分析未测量混淆的潜在影响
    """

    def __init__(self):
        pass

    def analyze_omitted_confounder(
        self,
        observed_ate: float,
        treatment_confounder_corr: float,
        outcome_confounder_corr: float,
        confounder_prevalence: float = 0.5
    ) -> Dict:
        """
        分析遗漏混淆变量的影响

        简化的偏差公式:
        Bias ≈ β_UT * (E[U|T=1] - E[U|T=0])

        其中:
        - β_UT: 混淆变量对结果的效应
        - E[U|T=1] - E[U|T=0]: 混淆变量在处理组和控制组的差异

        Parameters:
        -----------
        observed_ate: 观测到的 ATE
        treatment_confounder_corr: 处理与混淆变量的相关性
        outcome_confounder_corr: 结果与混淆变量的相关性
        confounder_prevalence: 混淆变量的患病率

        Returns:
        --------
        dict with bias analysis
        """
        # 简化假设: 混淆变量为二元变量
        # P(U=1|T=1) = p + ρ_TU
        # P(U=1|T=0) = p - ρ_TU

        p = confounder_prevalence
        rho_TU = treatment_confounder_corr

        # 混淆变量在两组的均值差
        confounder_diff = 2 * rho_TU

        # 混淆变量对结果的效应
        beta_U = outcome_confounder_corr

        # 估计偏差
        estimated_bias = beta_U * confounder_diff

        # 调整后的 ATE
        adjusted_ate = observed_ate - estimated_bias

        return {
            'observed_ate': float(observed_ate),
            'estimated_bias': float(estimated_bias),
            'adjusted_ate': float(adjusted_ate),
            'bias_percentage': float(estimated_bias / observed_ate * 100) if observed_ate != 0 else np.inf,
            'confounder_params': {
                'treatment_corr': float(treatment_confounder_corr),
                'outcome_corr': float(outcome_confounder_corr),
                'prevalence': float(confounder_prevalence)
            }
        }

    def sensitivity_contour(
        self,
        observed_ate: float,
        treatment_corr_range: Tuple[float, float] = (-0.5, 0.5),
        outcome_corr_range: Tuple[float, float] = (-2.0, 2.0),
        n_points: int = 20
    ) -> Dict:
        """
        生成敏感性等高线图数据

        Parameters:
        -----------
        observed_ate: 观测到的 ATE
        treatment_corr_range: 处理相关性范围
        outcome_corr_range: 结果相关性范围
        n_points: 网格点数量

        Returns:
        --------
        dict with contour data
        """
        treatment_corrs = np.linspace(*treatment_corr_range, n_points)
        outcome_corrs = np.linspace(*outcome_corr_range, n_points)

        adjusted_ates = np.zeros((n_points, n_points))

        for i, tc in enumerate(treatment_corrs):
            for j, oc in enumerate(outcome_corrs):
                result = self.analyze_omitted_confounder(
                    observed_ate, tc, oc
                )
                adjusted_ates[i, j] = result['adjusted_ate']

        return {
            'treatment_corrs': treatment_corrs.tolist(),
            'outcome_corrs': outcome_corrs.tolist(),
            'adjusted_ates': adjusted_ates.tolist(),
            'zero_effect_curve': self._find_zero_effect_curve(
                observed_ate, treatment_corrs, outcome_corrs
            )
        }

    def _find_zero_effect_curve(
        self,
        observed_ate: float,
        treatment_corrs: np.ndarray,
        outcome_corrs: np.ndarray
    ) -> Dict:
        """
        找到使调整后 ATE = 0 的混淆参数组合

        Bias = β_U * 2 * ρ_TU = observed_ate
        => β_U = observed_ate / (2 * ρ_TU)
        """
        zero_effect_outcomes = []

        for tc in treatment_corrs:
            if tc != 0:
                oc_required = observed_ate / (2 * tc)
                if outcome_corrs.min() <= oc_required <= outcome_corrs.max():
                    zero_effect_outcomes.append({
                        'treatment_corr': float(tc),
                        'outcome_corr': float(oc_required)
                    })

        return zero_effect_outcomes
