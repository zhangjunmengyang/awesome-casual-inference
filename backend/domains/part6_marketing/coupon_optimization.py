"""智能发券优化模块

实现基于 Uplift 建模的智能发券优化：
- 用户分群（Persuadables, Sure Things, Lost Causes, Sleeping Dogs）
- Uplift 建模（T-Learner）
- ROI 优化
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.ensemble import GradientBoostingClassifier


class UserSegmentation:
    """用户分群

    根据 Uplift 分数将用户分为不同的群体。
    """

    @staticmethod
    def segment_by_uplift(
        uplift_scores: np.ndarray,
        high_threshold: float = 0.1,
        low_threshold: float = 0.02
    ) -> np.ndarray:
        """根据 Uplift 分数分群

        Args:
            uplift_scores: Uplift 分数
            high_threshold: 高 Uplift 阈值
            low_threshold: 低 Uplift 阈值

        Returns:
            分群标签数组
        """
        segments = []
        for uplift in uplift_scores:
            if uplift >= high_threshold:
                segments.append('High Uplift (Target!)')
            elif uplift >= low_threshold:
                segments.append('Medium Uplift')
            elif uplift > 0:
                segments.append('Low Uplift')
            else:
                segments.append('Negative Uplift (Avoid!)')

        return np.array(segments)


class CouponOptimizer:
    """智能发券优化器

    使用 T-Learner 方法估计 Uplift，并基于 ROI 优化发券策略。
    """

    def __init__(self, random_state: int = 42):
        """
        Args:
            random_state: 随机种子
        """
        self.model_control = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=4,
            random_state=random_state
        )
        self.model_treatment = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=4,
            random_state=random_state + 1
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """训练 T-Learner 模型

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

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        """预测 Uplift

        Args:
            X: 特征矩阵

        Returns:
            Uplift 分数
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")

        # 分别预测转化概率
        prob_treatment = self.model_treatment.predict_proba(X)[:, 1]
        prob_control = self.model_control.predict_proba(X)[:, 1]

        # Uplift = 处理组概率 - 控制组概率
        uplift = prob_treatment - prob_control

        return uplift

    def calculate_roi(
        self,
        df: pd.DataFrame,
        treatment_mask: np.ndarray,
        revenue_per_conversion: float = 100,
        cost_per_coupon: float = 15
    ) -> Dict[str, float]:
        """计算发券策略的 ROI

        Args:
            df: 数据
            treatment_mask: 哪些用户应该发券（1=发券，0=不发）
            revenue_per_conversion: 每次转化的收入（元）
            cost_per_coupon: 每张券的成本（元）

        Returns:
            ROI 指标字典
        """
        # 发券的用户
        treated_users = df[treatment_mask == 1]

        if len(treated_users) == 0:
            return {
                'roi': 0,
                'revenue': 0,
                'cost': 0,
                'n_coupons': 0,
                'expected_conversions': 0,
                'actual_uplift': 0
            }

        # 在这些用户中，计算实际 uplift（基于实验数据）
        treat_mask = treated_users['T'] == 1
        control_mask = treated_users['T'] == 0

        if treat_mask.sum() > 0 and control_mask.sum() > 0:
            treat_conv = treated_users.loc[treat_mask, 'conversion'].mean()
            control_conv = treated_users.loc[control_mask, 'conversion'].mean()
            actual_uplift = treat_conv - control_conv
        else:
            actual_uplift = 0

        # 计算 ROI
        n_coupons = treatment_mask.sum()
        expected_conversions = n_coupons * actual_uplift
        revenue = expected_conversions * revenue_per_conversion
        cost = n_coupons * cost_per_coupon
        roi = (revenue - cost) / cost if cost > 0 else 0

        return {
            'roi': roi,
            'revenue': revenue,
            'cost': cost,
            'n_coupons': int(n_coupons),
            'expected_conversions': expected_conversions,
            'actual_uplift': actual_uplift
        }

    def optimize_strategy(
        self,
        X: np.ndarray,
        uplift_threshold: float = 0.05
    ) -> np.ndarray:
        """优化发券策略

        Args:
            X: 特征矩阵
            uplift_threshold: Uplift 阈值（只给 Uplift 高于此值的用户发券）

        Returns:
            发券掩码（1=发券，0=不发）
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before optimizing")

        uplift = self.predict_uplift(X)
        return (uplift > uplift_threshold).astype(int)

    def compare_strategies(
        self,
        df: pd.DataFrame,
        X: np.ndarray,
        budget_fraction: float = 0.3,
        revenue_per_conversion: float = 100,
        cost_per_coupon: float = 15
    ) -> pd.DataFrame:
        """对比不同发券策略

        Args:
            df: 数据
            X: 特征矩阵
            budget_fraction: 预算比例（可以发券给多少比例的用户）
            revenue_per_conversion: 每次转化的收入
            cost_per_coupon: 每张券的成本

        Returns:
            策略对比结果 DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before comparing strategies")

        n_target = int(budget_fraction * len(df))
        results = []

        # 策略 1: Random
        np.random.seed(42)
        random_idx = np.random.choice(len(df), n_target, replace=False)
        random_mask = np.zeros(len(df))
        random_mask[random_idx] = 1
        random_roi = self.calculate_roi(
            df, random_mask, revenue_per_conversion, cost_per_coupon
        )
        results.append({'strategy': 'Random', **random_roi})

        # 策略 2: High Frequency
        if 'order_freq' in df.columns:
            high_freq_idx = np.argsort(df['order_freq'].values)[-n_target:]
            high_freq_mask = np.zeros(len(df))
            high_freq_mask[high_freq_idx] = 1
            high_freq_roi = self.calculate_roi(
                df, high_freq_mask, revenue_per_conversion, cost_per_coupon
            )
            results.append({'strategy': 'High Frequency', **high_freq_roi})

        # 策略 3: Uplift Model
        uplift_scores = self.predict_uplift(X)
        uplift_idx = np.argsort(uplift_scores)[-n_target:]
        uplift_mask = np.zeros(len(df))
        uplift_mask[uplift_idx] = 1
        uplift_roi = self.calculate_roi(
            df, uplift_mask, revenue_per_conversion, cost_per_coupon
        )
        results.append({'strategy': 'Uplift Model', **uplift_roi})

        return pd.DataFrame(results)
