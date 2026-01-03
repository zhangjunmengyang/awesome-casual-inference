"""
挑战 3: Uplift 排序
对用户按处理效应排序，用于精准营销
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple

from .challenge_base import Challenge, ChallengeResult, ChallengeDataGenerator


class UpliftRankingChallenge(Challenge):
    """
    Uplift 排序挑战

    任务: 对用户按 uplift 排序，识别最应该接受干预的用户

    场景: 营销优惠券发放
    - 处理: 发放优惠券
    - 结果: 是否购买 (转化)
    - 挑战: 存在四类用户 (Persuadables, Sure things, Lost causes, Sleeping dogs)

    评估指标:
    - 主要: Qini 系数 / AUUC
    - 次要: Top-K Uplift, 业务价值 (ROI)

    难度: Advanced
    """

    def __init__(self):
        super().__init__(
            name="Uplift Ranking Challenge",
            description="Rank users by treatment effect for optimal targeting",
            difficulty="advanced"
        )

        self.true_ate = None
        self.true_uplift_test = None

    def generate_data(self, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        生成训练和测试数据

        训练集: 5000 样本
        测试集: 2000 样本
        """
        # 训练数据
        train_df, train_uplift, train_ate = (
            ChallengeDataGenerator.generate_marketing_data(
                n=5000,
                seed=seed,
                n_features=8
            )
        )

        # 测试数据
        test_df, test_uplift, test_ate = (
            ChallengeDataGenerator.generate_marketing_data(
                n=2000,
                seed=seed + 3000,
                n_features=8
            )
        )

        # 存储真实值
        self.true_ate = test_ate
        self.true_uplift_test = test_uplift
        self.true_targets = test_uplift

        self.train_data = train_df
        self.test_data = test_df

        return train_df, test_df

    def evaluate(
        self,
        predictions: np.ndarray,
        user_name: str = "Anonymous"
    ) -> ChallengeResult:
        """
        评估 Uplift 排序

        predictions: 测试集上的 uplift 预测 (shape: (n_test,))
        评估基于排序质量，而非绝对值准确性
        """
        # 验证格式
        is_valid, msg = self.validate_predictions(predictions)
        if not is_valid:
            raise ValueError(f"Invalid predictions: {msg}")

        Y = self.test_data['Y'].values
        T = self.test_data['T'].values
        true_uplift = self.true_uplift_test

        # 主要指标: AUUC (Qini)
        auuc = self.calculate_auuc(Y, T, predictions)
        auuc_perfect = self.calculate_auuc(Y, T, true_uplift)
        auuc_random = self.calculate_auuc(Y, T, np.random.randn(len(predictions)))

        # 归一化 AUUC
        normalized_auuc = (auuc - auuc_random) / (auuc_perfect - auuc_random) if auuc_perfect != auuc_random else 0

        # Top-K uplift
        k_values = [0.1, 0.2, 0.3]
        top_k_uplifts = {}

        for k in k_values:
            n_top = int(k * len(predictions))
            top_indices = np.argsort(predictions)[-n_top:]
            avg_true_uplift = true_uplift[top_indices].mean()
            top_k_uplifts[f'top_{int(k*100)}%'] = avg_true_uplift

        # 业务价值指标
        cost_per_treatment = 1.0  # 每次干预成本
        revenue_per_conversion = 10.0  # 每次转化收益

        # 如果对 top 30% 用户干预
        n_target = int(0.3 * len(predictions))
        top_30_indices = np.argsort(predictions)[-n_target:]

        expected_conversions = true_uplift[top_30_indices].sum()
        revenue = expected_conversions * revenue_per_conversion
        cost = n_target * cost_per_treatment
        roi = (revenue - cost) / cost if cost > 0 else 0

        # Qini 系数 (曲线下最大值)
        fraction, qini_curve = self.calculate_qini_curve(Y, T, predictions)
        max_qini = qini_curve.max()

        # 排序质量: Kendall's Tau
        from scipy.stats import kendalltau
        tau, _ = kendalltau(predictions, true_uplift)

        secondary_metrics = {
            'auuc': auuc,
            'normalized_auuc': normalized_auuc,
            'max_qini': max_qini,
            'kendall_tau': tau,
            'roi_30%': roi,
            **top_k_uplifts,
            'true_ate': self.true_ate,
            'estimated_ate': predictions.mean()
        }

        # 计算得分 (0-100)
        # 基于 normalized AUUC
        score = max(0, 100 * normalized_auuc)

        # 奖励高 ROI
        if roi > 2.0:
            score += 10
        elif roi > 1.0:
            score += 5

        score = min(100, score)

        result = ChallengeResult(
            challenge_name=self.name,
            user_name=user_name,
            submission_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            primary_metric=auuc,
            secondary_metrics=secondary_metrics,
            score=score
        )

        return result

    def calculate_qini_curve(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        predictions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算 Qini 曲线"""
        # 按预测 uplift 降序排序
        sorted_idx = np.argsort(predictions)[::-1]
        Y_sorted = Y[sorted_idx]
        T_sorted = T[sorted_idx]

        n = len(Y)
        fractions = np.linspace(0, 1, 100)
        qini_values = []

        for frac in fractions:
            k = int(frac * n)
            if k == 0:
                qini_values.append(0)
                continue

            Y_k = Y_sorted[:k]
            T_k = T_sorted[:k]

            n_t = T_k.sum()
            n_c = k - n_t

            if n_t > 0 and n_c > 0:
                qini = Y_k[T_k == 1].sum() - Y_k[T_k == 0].sum() * (n_t / n_c)
            else:
                qini = 0

            qini_values.append(qini)

        return fractions, np.array(qini_values)

    def calculate_auuc(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        predictions: np.ndarray
    ) -> float:
        """计算 AUUC (Area Under Uplift Curve)"""
        fraction, qini = self.calculate_qini_curve(Y, T, predictions)
        auuc = np.trapz(qini, fraction)
        return auuc

    def get_baseline_predictions(self, method: str = 't_learner') -> np.ndarray:
        """
        获取基线方法的 Uplift 预测

        Methods:
        - 't_learner': T-Learner
        - 'uplift_tree': Uplift Tree
        - 'class_transformation': Class Transformation
        """
        if self.train_data is None or self.test_data is None:
            raise ValueError("Data not generated. Call generate_data() first.")

        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        X_cols = [col for col in self.train_data.columns if col.startswith('feature_')]
        X_train = self.train_data[X_cols].values
        T_train = self.train_data['T'].values
        Y_train = self.train_data['Y'].values
        X_test = self.test_data[X_cols].values

        if method == 't_learner':
            # T-Learner for binary outcome
            model_0 = RandomForestClassifier(n_estimators=100, random_state=42)
            model_1 = RandomForestClassifier(n_estimators=100, random_state=43)

            model_0.fit(X_train[T_train == 0], Y_train[T_train == 0])
            model_1.fit(X_train[T_train == 1], Y_train[T_train == 1])

            p0 = model_0.predict_proba(X_test)[:, 1]
            p1 = model_1.predict_proba(X_test)[:, 1]

            uplift = p1 - p0

        elif method == 'class_transformation':
            # Class Transformation approach
            # Transform to 4-class problem
            Z = np.zeros(len(Y_train), dtype=int)
            Z[(T_train == 1) & (Y_train == 1)] = 0  # Treated & Responded
            Z[(T_train == 1) & (Y_train == 0)] = 1  # Treated & Not Responded
            Z[(T_train == 0) & (Y_train == 1)] = 2  # Control & Responded
            Z[(T_train == 0) & (Y_train == 0)] = 3  # Control & Not Responded

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, Z)

            proba = model.predict_proba(X_test)

            # P(Y=1|T=1) - P(Y=1|T=0)
            if proba.shape[1] == 4:
                p_y1_t1 = proba[:, 0] / (proba[:, 0] + proba[:, 1] + 1e-10)
                p_y1_t0 = proba[:, 2] / (proba[:, 2] + proba[:, 3] + 1e-10)
                uplift = p_y1_t1 - p_y1_t0
            else:
                uplift = np.zeros(len(X_test))

        else:
            raise ValueError(f"Unknown method: {method}")

        return uplift

    def get_starter_code(self) -> str:
        """返回 starter code"""
        return """
# Uplift Ranking Starter Code
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def predict_uplift(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    \"\"\"
    预测 Uplift 用于排序

    Parameters
    ----------
    train_df : pd.DataFrame
        训练数据，包含 T (处理), Y (转化), feature_1 - feature_8
    test_df : pd.DataFrame
        测试数据

    Returns
    -------
    uplift_predictions : np.ndarray
        测试集上的 uplift 预测 (排序用)，shape: (n_test,)
    \"\"\"

    X_cols = [f'feature_{i}' for i in range(1, 9)]

    X_train = train_df[X_cols].values
    T_train = train_df['T'].values
    Y_train = train_df['Y'].values

    X_test = test_df[X_cols].values

    # TODO: 实现你的 Uplift 预测方法

    # 注意: Y 是二分类 (0/1)，需要使用分类模型

    # 方法 1: T-Learner (分类版本)
    # 1. 训练两个分类器预测转化概率
    # 2. uplift = P(Y=1|T=1, X) - P(Y=1|T=0, X)

    # 方法 2: S-Learner
    # 1. 训练单一模型
    # 2. 预测 uplift

    # 方法 3: Class Transformation
    # 1. 转换为4类问题
    # 2. 计算 uplift

    # 方法 4: 直接建模 Uplift (Uplift Tree, Uplift RF)

    # 示例: T-Learner
    model_0 = RandomForestClassifier(n_estimators=100, random_state=42)
    model_1 = RandomForestClassifier(n_estimators=100, random_state=43)

    model_0.fit(X_train[T_train == 0], Y_train[T_train == 0])
    model_1.fit(X_train[T_train == 1], Y_train[T_train == 1])

    p0 = model_0.predict_proba(X_test)[:, 1]
    p1 = model_1.predict_proba(X_test)[:, 1]

    uplift_predictions = p1 - p0

    return uplift_predictions

# 示例用法
# uplift_preds = predict_uplift(train_data, test_data)
# submit(uplift_preds)

# 业务场景:
# - 对 uplift_preds 降序排序
# - 对 top 30% 用户发放优惠券
# - 最大化 ROI
"""

    def get_metric_description(self) -> dict:
        """返回评估指标说明"""
        return {
            'auuc': 'AUUC (Area Under Uplift Curve) - Qini 曲线下面积',
            'normalized_auuc': '归一化 AUUC，范围 [0, 1]',
            'max_qini': 'Qini 曲线的最大值',
            'kendall_tau': 'Kendall Tau 排序相关系数',
            'roi_30%': '对 top 30% 用户干预的投资回报率',
            'top_k%': 'Top K% 用户的平均真实 uplift',
            'score': '综合得分 (0-100), 基于 AUUC 和 ROI'
        }

    def get_detailed_info(self) -> str:
        """返回详细挑战说明"""
        baseline_t = self.get_baseline_predictions('t_learner')

        Y = self.test_data['Y'].values
        T = self.test_data['T'].values

        auuc_t = self.calculate_auuc(Y, T, baseline_t)

        return f"""
### 挑战说明

**场景**: 营销优惠券发放优化

你是一家电商公司的数据科学家，需要决定向哪些用户发放优惠券。

**数据描述**:
- **处理 (T)**: 是否收到优惠券 (0/1)
- **结果 (Y)**: 是否购买/转化 (0/1)
- **特征 (feature_1 - feature_8)**: 用户行为特征

**四类用户**:
1. **Persuadables**: 发券会购买，不发券不买 (正 uplift)
2. **Sure Things**: 无论发不发券都会买 (零 uplift)
3. **Lost Causes**: 无论发不发券都不买 (零 uplift)
4. **Sleeping Dogs**: 发券反而不买 (负 uplift)

**业务目标**:
- 优惠券成本: 1元/张
- 转化收益: 10元/次
- 最大化 ROI = (收益 - 成本) / 成本

**真实 ATE**: {self.true_ate:.4f}

**基线方法**:
- T-Learner: AUUC = {auuc_t:.4f}

**评估指标**:
- **AUUC** (主要): Qini 曲线下面积，衡量排序质量
- **Kendall Tau**: 与真实 uplift 排序的一致性
- **Top-K Uplift**: Top K% 用户的平均 uplift
- **ROI**: 业务投资回报率

**关键洞察**:
1. 不要向所有人发券! (浪费在 Sure Things 和 Lost Causes)
2. 避免向 Sleeping Dogs 发券 (负效应)
3. 识别并重点针对 Persuadables

**提示**:
1. 专注于排序，而非绝对值预测
2. T-Learner 在 Uplift 场景中表现通常不错
3. 考虑 Class Transformation 方法
4. 可以尝试 causalml 库的 Uplift Tree
5. 特征工程: 用户历史行为、偏好等

**目标**: Normalized AUUC > 0.7 即可获得高分!

**业务决策**: 根据你的模型，应该对 top X% 用户发券来最大化 ROI
"""

    def calculate_optimal_targeting_fraction(self) -> dict:
        """
        计算最优干预比例

        Returns
        -------
        info : dict
            包含最优比例、ROI 等信息
        """
        true_uplift = self.true_uplift_test

        sorted_idx = np.argsort(true_uplift)[::-1]
        sorted_uplift = true_uplift[sorted_idx]

        cost_per_treatment = 1.0
        revenue_per_conversion = 10.0

        n = len(true_uplift)
        fractions = np.linspace(0.05, 1.0, 20)
        roi_values = []

        for frac in fractions:
            k = int(frac * n)
            cumulative_uplift = sorted_uplift[:k].sum()
            revenue = cumulative_uplift * revenue_per_conversion
            cost = k * cost_per_treatment
            roi = (revenue - cost) / cost if cost > 0 else 0
            roi_values.append(roi)

        best_idx = np.argmax(roi_values)
        optimal_fraction = fractions[best_idx]
        optimal_roi = roi_values[best_idx]

        return {
            'optimal_fraction': optimal_fraction,
            'optimal_roi': optimal_roi,
            'fractions': fractions,
            'roi_values': roi_values
        }
