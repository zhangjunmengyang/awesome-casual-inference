"""预算分配优化模块

实现基于响应曲线的多渠道预算优化：
- 响应曲线建模（Hill Equation）
- 边际 ROI 优化
- 约束优化
- 稳健优化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize, LinearConstraint
from itertools import chain, combinations


class ResponseCurveModel:
    """响应曲线模型

    使用 Hill Equation 建模渠道响应曲线。
    """

    @staticmethod
    def response(x: np.ndarray, a: float, c: float, alpha: float) -> np.ndarray:
        """计算响应曲线值

        R(x) = a * x^alpha / (c^alpha + x^alpha)

        Args:
            x: 投入预算
            a: 饱和收益（最大收益）
            c: 半饱和点（达到 50% 最大收益时的投入）
            alpha: 形状参数

        Returns:
            响应值
        """
        return a * (x ** alpha) / (c ** alpha + x ** alpha)

    @staticmethod
    def marginal_response(
        x: np.ndarray,
        a: float,
        c: float,
        alpha: float
    ) -> np.ndarray:
        """计算边际响应（导数）

        R'(x) = a * alpha * c^alpha * x^(alpha-1) / (c^alpha + x^alpha)^2

        Args:
            x: 投入预算
            a: 饱和收益
            c: 半饱和点
            alpha: 形状参数

        Returns:
            边际响应值
        """
        numerator = a * alpha * (c ** alpha) * (x ** (alpha - 1))
        denominator = (c ** alpha + x ** alpha) ** 2
        return numerator / denominator


class BudgetOptimizer:
    """预算优化器

    封装完整的预算优化流程。
    """

    def __init__(self, channels_params: Dict[str, Dict[str, float]]):
        """
        Args:
            channels_params: 渠道参数
                {'channel_name': {'a': ..., 'c': ..., 'alpha': ...}}
        """
        self.channels = channels_params
        self.channel_names = list(channels_params.keys())
        self.n_channels = len(channels_params)
        self.history = []

    def optimize(
        self,
        total_budget: float,
        constraints: Optional[Dict] = None,
        interaction_matrix: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, float], float]:
        """执行预算优化

        Args:
            total_budget: 总预算
            constraints: 约束条件字典
                {
                    'min_budget': {channel: min_value},
                    'max_budget': {channel: max_value},
                    'max_share': {channel: max_ratio}
                }
            interaction_matrix: 渠道交互矩阵（可选）

        Returns:
            (allocation, total_response)
            - allocation: 最优预算分配
            - total_response: 总收益
        """
        n = self.n_channels

        def objective(x):
            """目标函数：最大化总收益"""
            total = 0
            for i, name in enumerate(self.channel_names):
                total += ResponseCurveModel.response(
                    x[i],
                    **self.channels[name]
                )

            # 添加交互效应
            if interaction_matrix is not None:
                for i in range(n):
                    for j in range(i + 1, n):
                        gamma = interaction_matrix[i, j]
                        if gamma != 0:
                            # 使用几何平均建模交互
                            interaction_term = gamma * np.sqrt(x[i] * x[j] + 1e-10)
                            total += interaction_term

            return -total  # 负号：最小化转最大化

        # 构建约束条件
        cons = [LinearConstraint(np.ones(n), total_budget, total_budget)]

        # 构建边界
        bounds = []
        for i, name in enumerate(self.channel_names):
            min_b = 0
            max_b = total_budget

            if constraints:
                if 'min_budget' in constraints and name in constraints['min_budget']:
                    min_b = constraints['min_budget'][name]
                if 'max_budget' in constraints and name in constraints['max_budget']:
                    max_b = min(max_b, constraints['max_budget'][name])
                if 'max_share' in constraints and name in constraints['max_share']:
                    max_share = constraints['max_share'][name]
                    max_b = min(max_b, total_budget * max_share)

            bounds.append((min_b, max_b))

        # 初始猜测
        x0 = np.ones(n) * total_budget / n

        # 优化求解
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000}
        )

        allocation = dict(zip(self.channel_names, result.x))
        total_response = -result.fun

        # 记录历史
        self.history.append({
            'budget': total_budget,
            'allocation': allocation,
            'response': total_response
        })

        return allocation, total_response

    def compute_marginal_rois(
        self,
        allocation: Dict[str, float]
    ) -> Dict[str, float]:
        """计算各渠道的边际 ROI

        Args:
            allocation: 预算分配

        Returns:
            各渠道的边际 ROI
        """
        marginal_rois = {}
        for name, budget in allocation.items():
            marginal_rois[name] = ResponseCurveModel.marginal_response(
                budget,
                **self.channels[name]
            )
        return marginal_rois

    def sensitivity_analysis(
        self,
        budget_range: np.ndarray
    ) -> pd.DataFrame:
        """敏感性分析：不同预算下的最优分配

        Args:
            budget_range: 预算范围

        Returns:
            敏感性分析结果
        """
        results = []
        for B in budget_range:
            alloc, resp = self.optimize(B)
            marginal_rois = self.compute_marginal_rois(alloc)

            result = {
                'budget': B,
                'response': resp,
                'shadow_price': np.mean(list(marginal_rois.values()))
            }
            # 添加每个渠道的分配
            for ch, budget in alloc.items():
                result[f'{ch}_budget'] = budget

            results.append(result)

        return pd.DataFrame(results)

    def robust_optimization(
        self,
        total_budget: float,
        param_uncertainty: Dict[str, Dict[str, float]],
        n_scenarios: int = 500
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """稳健优化（蒙特卡洛方法）

        考虑参数不确定性的优化。

        Args:
            total_budget: 总预算
            param_uncertainty: 参数不确定性（标准差）
                {'channel': {'a_std': ..., 'c_std': ..., 'alpha_std': ...}}
            n_scenarios: 场景数量

        Returns:
            (allocation, stats)
            - allocation: 稳健最优分配
            - stats: 性能统计
        """
        n = self.n_channels

        # 生成参数场景
        scenarios = []
        for _ in range(n_scenarios):
            scenario = {}
            for name in self.channel_names:
                base = self.channels[name]
                unc = param_uncertainty.get(
                    name,
                    {'a_std': 0, 'c_std': 0, 'alpha_std': 0}
                )

                # 采样参数（确保 > 0）
                scenario[name] = {
                    'a': max(0.1, np.random.normal(base['a'], unc.get('a_std', 0))),
                    'c': max(1, np.random.normal(base['c'], unc.get('c_std', 0))),
                    'alpha': max(0.1, np.random.normal(base['alpha'], unc.get('alpha_std', 0)))
                }
            scenarios.append(scenario)

        # 优化：期望最大化
        def objective_expected(x):
            expected_response = 0
            for scenario in scenarios:
                scenario_response = 0
                for i, name in enumerate(self.channel_names):
                    scenario_response += ResponseCurveModel.response(
                        x[i],
                        **scenario[name]
                    )
                expected_response += scenario_response
            expected_response /= n_scenarios
            return -expected_response

        # 求解
        constraints = [LinearConstraint(np.ones(n), total_budget, total_budget)]
        bounds = [(0, total_budget) for _ in range(n)]
        x0 = np.ones(n) * total_budget / n

        result = minimize(
            objective_expected,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        allocation = dict(zip(self.channel_names, result.x))

        # 评估该分配在所有场景下的表现
        performances = []
        for scenario in scenarios:
            perf = sum(
                ResponseCurveModel.response(allocation[name], **scenario[name])
                for name in self.channel_names
            )
            performances.append(perf)

        performances = np.array(performances)

        stats = {
            'mean': float(np.mean(performances)),
            'std': float(np.std(performances)),
            'min': float(np.min(performances)),
            'percentile_5': float(np.percentile(performances, 5)),
            'percentile_95': float(np.percentile(performances, 95)),
            'max': float(np.max(performances))
        }

        return allocation, stats

    def what_if(
        self,
        scenario_name: str,
        **kwargs
    ) -> Tuple[Dict[str, float], float]:
        """What-if 分析

        Args:
            scenario_name: 场景名称
            **kwargs: 传递给 optimize 的参数

        Returns:
            (allocation, response)
        """
        allocation, response = self.optimize(**kwargs)
        return allocation, response
