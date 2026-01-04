"""用户定向模块

实现基于 CATE 估计的用户定向：
- T-Learner: 分别训练处理组和控制组的模型
- X-Learner: 考虑伪处理效应的高级方法
- Policy Learning: 学习最优干预策略
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression


class TLearner:
    """T-Learner: 分别训练处理组和控制组的模型"""

    def __init__(self, random_state: int = 42):
        """
        Args:
            random_state: 随机种子
        """
        self.model_control = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=4,
            random_state=random_state
        )
        self.model_treatment = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=4,
            random_state=random_state + 1
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """训练 T-Learner

        Args:
            X: 特征矩阵
            T: 处理指示变量
            Y: 结果变量

        Returns:
            self
        """
        # 分离控制组和处理组
        mask_control = (T == 0)
        mask_treatment = (T == 1)

        # 训练两个模型
        self.model_control.fit(X[mask_control], Y[mask_control])
        self.model_treatment.fit(X[mask_treatment], Y[mask_treatment])

        self.is_fitted = True
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """预测 CATE (Conditional Average Treatment Effect)

        Args:
            X: 特征矩阵

        Returns:
            CATE 估计
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")

        # 分别预测
        mu1 = self.model_treatment.predict(X)
        mu0 = self.model_control.predict(X)

        # CATE = mu1 - mu0
        cate = mu1 - mu0

        return cate


class XLearner:
    """X-Learner: 额外学习伪处理效应的高级方法"""

    def __init__(self, random_state: int = 42):
        """
        Args:
            random_state: 随机种子
        """
        self.model_control = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=4,
            random_state=random_state
        )
        self.model_treatment = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=4,
            random_state=random_state + 1
        )
        self.tau_control = GradientBoostingRegressor(
            n_estimators=30,
            max_depth=3,
            random_state=random_state + 2
        )
        self.tau_treatment = GradientBoostingRegressor(
            n_estimators=30,
            max_depth=3,
            random_state=random_state + 3
        )
        self.propensity_model = LogisticRegression(
            random_state=random_state + 4
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """训练 X-Learner (三阶段)

        Args:
            X: 特征矩阵
            T: 处理指示变量
            Y: 结果变量

        Returns:
            self
        """
        mask_control = (T == 0)
        mask_treatment = (T == 1)

        # Stage 1: 训练 mu_0 和 mu_1
        self.model_control.fit(X[mask_control], Y[mask_control])
        self.model_treatment.fit(X[mask_treatment], Y[mask_treatment])

        # Stage 2: 计算伪处理效应
        # 处理组: D = Y - mu_0(X) (实际 - 反事实)
        mu0_treatment = self.model_control.predict(X[mask_treatment])
        D_treatment = Y[mask_treatment] - mu0_treatment

        # 控制组: D = mu_1(X) - Y (反事实 - 实际)
        mu1_control = self.model_treatment.predict(X[mask_control])
        D_control = mu1_control - Y[mask_control]

        # 训练伪效应模型
        self.tau_treatment.fit(X[mask_treatment], D_treatment)
        self.tau_control.fit(X[mask_control], D_control)

        # Stage 3: 估计倾向得分
        self.propensity_model.fit(X, T)

        self.is_fitted = True
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """预测 CATE (使用倾向得分加权)

        Args:
            X: 特征矩阵

        Returns:
            CATE 估计
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")

        # 预测两个 tau
        tau0 = self.tau_control.predict(X)
        tau1 = self.tau_treatment.predict(X)

        # 估计倾向得分
        propensity = self.propensity_model.predict_proba(X)[:, 1]

        # 加权组合 (使用倾向得分)
        cate = propensity * tau0 + (1 - propensity) * tau1

        return cate


class PolicyLearner:
    """策略学习器

    学习最优干预策略：基于 CATE 和成本-收益权衡决定是否干预。
    """

    @staticmethod
    def learn_optimal_policy(
        cate: np.ndarray,
        cost_per_treatment: float = 100,
        value_per_unit: float = 30
    ) -> Tuple[np.ndarray, Dict]:
        """学习最优干预策略

        决策规则: 当 CATE * value > cost 时进行干预

        Args:
            cate: 估计的 CATE
            cost_per_treatment: 每次处理的成本
            value_per_unit: 每单位结果的价值

        Returns:
            (optimal_policy, metrics)
            - optimal_policy: 最优策略掩码（1=干预，0=不干预）
            - metrics: 策略性能指标
        """
        # 计算阈值：多少增量才划算
        threshold = cost_per_treatment / value_per_unit

        # 最优策略
        optimal_policy = (cate > threshold).astype(int)

        # 计算指标
        n_treated = optimal_policy.sum()
        expected_effect = cate[optimal_policy == 1].sum() if n_treated > 0 else 0
        total_cost = n_treated * cost_per_treatment
        total_value = expected_effect * value_per_unit
        net_benefit = total_value - total_cost
        roi = net_benefit / total_cost if total_cost > 0 else 0

        metrics = {
            'n_treated': int(n_treated),
            'treatment_rate': float(n_treated / len(cate)),
            'expected_effect': float(expected_effect),
            'total_cost': float(total_cost),
            'total_value': float(total_value),
            'net_benefit': float(net_benefit),
            'roi': float(roi),
            'threshold': float(threshold)
        }

        return optimal_policy, metrics

    @staticmethod
    def segment_by_cate(
        cate: np.ndarray,
        percentiles: Tuple[float, float] = (25, 75)
    ) -> np.ndarray:
        """根据 CATE 将用户分层

        Args:
            cate: CATE 分数
            percentiles: 分位数阈值 (低, 高)

        Returns:
            分层标签
        """
        p_low, p_high = percentiles
        threshold_low = np.percentile(cate, p_low)
        threshold_high = np.percentile(cate, p_high)

        segments = []
        for c in cate:
            if c >= threshold_high:
                segments.append('High Impact')
            elif c >= threshold_low:
                segments.append('Medium Impact')
            elif c > 0:
                segments.append('Low Impact')
            else:
                segments.append('Negative Impact')

        return np.array(segments)

    @staticmethod
    def compare_targeting_strategies(
        df: pd.DataFrame,
        cate: np.ndarray,
        cost: float = 100,
        value: float = 30
    ) -> pd.DataFrame:
        """对比不同干预策略

        Args:
            df: 数据
            cate: CATE 分数
            cost: 处理成本
            value: 单位价值

        Returns:
            策略对比结果
        """
        results = []

        # 策略 1: No Treatment
        results.append({
            'strategy': 'No Treatment',
            'n_treated': 0,
            'cost': 0,
            'value': 0,
            'net_benefit': 0,
            'roi': 0
        })

        # 策略 2: Treat All
        n_all = len(df)
        expected_effect_all = cate.sum()
        cost_all = n_all * cost
        value_all = expected_effect_all * value
        results.append({
            'strategy': 'Treat All',
            'n_treated': n_all,
            'cost': cost_all,
            'value': value_all,
            'net_benefit': value_all - cost_all,
            'roi': (value_all - cost_all) / cost_all if cost_all > 0 else 0
        })

        # 策略 3: Treat Part-time Only (如果有相关特征)
        if 'is_fulltime' in df.columns:
            parttime_mask = df['is_fulltime'] == 0
            n_parttime = parttime_mask.sum()
            expected_effect_parttime = cate[parttime_mask].sum()
            cost_parttime = n_parttime * cost
            value_parttime = expected_effect_parttime * value
            results.append({
                'strategy': 'Treat Part-time',
                'n_treated': n_parttime,
                'cost': cost_parttime,
                'value': value_parttime,
                'net_benefit': value_parttime - cost_parttime,
                'roi': (value_parttime - cost_parttime) / cost_parttime if cost_parttime > 0 else 0
            })

        # 策略 4: Optimal Policy
        optimal_policy, optimal_metrics = PolicyLearner.learn_optimal_policy(
            cate, cost, value
        )
        results.append({
            'strategy': 'Optimal Policy',
            'n_treated': optimal_metrics['n_treated'],
            'cost': optimal_metrics['total_cost'],
            'value': optimal_metrics['total_value'],
            'net_benefit': optimal_metrics['net_benefit'],
            'roi': optimal_metrics['roi']
        })

        return pd.DataFrame(results)
