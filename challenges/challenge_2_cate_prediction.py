"""
挑战 2: CATE 预测
预测个体条件平均处理效应
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple

from .challenge_base import Challenge, ChallengeResult, ChallengeDataGenerator


class CATEPredictionChallenge(Challenge):
    """
    CATE 预测挑战

    任务: 给定协变量，预测每个个体的条件平均处理效应

    场景: IHDP 婴儿健康发展干预项目
    - 处理: 早期教育干预
    - 结果: 认知测试得分
    - 挑战: 效应具有复杂的异质性 (不同孩子效果差异大)

    评估指标:
    - 主要: PEHE (Precision in Estimation of Heterogeneous Effect)
    - 次要: ATE bias, Correlation, R²

    难度: Intermediate
    """

    def __init__(self):
        super().__init__(
            name="CATE Prediction Challenge",
            description="Predict individual-level treatment effects (CATE)",
            difficulty="intermediate"
        )

        self.true_ate = None
        self.true_cate_test = None

    def generate_data(self, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        生成训练和测试数据

        训练集: 3000 样本
        测试集: 1000 样本
        """
        # 训练数据
        train_df, train_cate, train_ate = (
            ChallengeDataGenerator.generate_ihdp_style_data(
                n=3000,
                seed=seed,
                n_features=10
            )
        )

        # 测试数据
        test_df, test_cate, test_ate = (
            ChallengeDataGenerator.generate_ihdp_style_data(
                n=1000,
                seed=seed + 2000,
                n_features=10
            )
        )

        # 存储真实值
        self.true_ate = test_ate
        self.true_cate_test = test_cate
        self.true_targets = test_cate

        self.train_data = train_df
        self.test_data = test_df

        return train_df, test_df

    def evaluate(
        self,
        predictions: np.ndarray,
        user_name: str = "Anonymous"
    ) -> ChallengeResult:
        """
        评估 CATE 预测

        predictions: 测试集上的 CATE 预测 (shape: (n_test,))
        """
        # 验证格式
        is_valid, msg = self.validate_predictions(predictions)
        if not is_valid:
            raise ValueError(f"Invalid predictions: {msg}")

        true_cate = self.true_cate_test

        # 主要指标: PEHE
        pehe = np.sqrt(np.mean((predictions - true_cate) ** 2))

        # 次要指标
        ate_bias = abs(predictions.mean() - true_cate.mean())

        # 计算相关系数，处理零方差情况
        pred_std = np.std(predictions)
        true_std = np.std(true_cate)
        if pred_std > 1e-10 and true_std > 1e-10:
            correlation = np.corrcoef(predictions, true_cate)[0, 1]
            # 处理可能的 NaN（如数值不稳定）
            if np.isnan(correlation):
                correlation = 0.0
        else:
            # 方差为零时相关系数无意义
            correlation = 0.0

        # R² score
        ss_res = np.sum((true_cate - predictions) ** 2)
        ss_tot = np.sum((true_cate - true_cate.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        # MAE
        mae = np.mean(np.abs(predictions - true_cate))

        secondary_metrics = {
            'pehe': pehe,
            'ate_bias': ate_bias,
            'correlation': correlation,
            'r2': r2,
            'mae': mae,
            'true_ate': self.true_ate,
            'estimated_ate': predictions.mean()
        }

        # 计算得分 (0-100)
        # 基于 PEHE 的归一化得分
        # PEHE = 0 -> 100分, PEHE = 5 -> 0分
        max_pehe = 5.0
        score = max(0, 100 * (1 - pehe / max_pehe))

        # 奖励高相关性
        if correlation > 0.8:
            score += 10
        elif correlation > 0.6:
            score += 5

        score = min(100, score)

        result = ChallengeResult(
            challenge_name=self.name,
            user_name=user_name,
            submission_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            primary_metric=pehe,
            secondary_metrics=secondary_metrics,
            score=score
        )

        return result

    def get_baseline_predictions(self, method: str = 's_learner') -> np.ndarray:
        """
        获取基线方法的 CATE 预测

        Methods:
        - 's_learner': S-Learner
        - 't_learner': T-Learner
        - 'x_learner': X-Learner
        """
        if self.train_data is None or self.test_data is None:
            raise ValueError("Data not generated. Call generate_data() first.")

        from sklearn.ensemble import RandomForestRegressor

        X_cols = [col for col in self.train_data.columns if col.startswith('X')]
        X_train = self.train_data[X_cols].values
        T_train = self.train_data['T'].values
        Y_train = self.train_data['Y'].values
        X_test = self.test_data[X_cols].values

        if method == 's_learner':
            # S-Learner
            X_with_T = np.column_stack([X_train, T_train])
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_with_T, Y_train)

            X_test_1 = np.column_stack([X_test, np.ones(len(X_test))])
            X_test_0 = np.column_stack([X_test, np.zeros(len(X_test))])

            cate = model.predict(X_test_1) - model.predict(X_test_0)

        elif method == 't_learner':
            # T-Learner
            model_0 = RandomForestRegressor(n_estimators=100, random_state=42)
            model_1 = RandomForestRegressor(n_estimators=100, random_state=43)

            model_0.fit(X_train[T_train == 0], Y_train[T_train == 0])
            model_1.fit(X_train[T_train == 1], Y_train[T_train == 1])

            cate = model_1.predict(X_test) - model_0.predict(X_test)

        elif method == 'x_learner':
            # 简化的 X-Learner
            from sklearn.linear_model import LogisticRegression

            # Stage 1: 训练结果模型
            model_0 = RandomForestRegressor(n_estimators=100, random_state=42)
            model_1 = RandomForestRegressor(n_estimators=100, random_state=43)

            mask_0 = T_train == 0
            mask_1 = T_train == 1

            model_0.fit(X_train[mask_0], Y_train[mask_0])
            model_1.fit(X_train[mask_1], Y_train[mask_1])

            # Stage 2: 伪处理效应
            D_1 = Y_train[mask_1] - model_0.predict(X_train[mask_1])
            D_0 = model_1.predict(X_train[mask_0]) - Y_train[mask_0]

            tau_1_model = RandomForestRegressor(n_estimators=100, random_state=44)
            tau_0_model = RandomForestRegressor(n_estimators=100, random_state=45)

            tau_1_model.fit(X_train[mask_1], D_1)
            tau_0_model.fit(X_train[mask_0], D_0)

            # 倾向得分
            ps_model = LogisticRegression(max_iter=1000, random_state=42)
            ps_model.fit(X_train, T_train)
            g_test = ps_model.predict_proba(X_test)[:, 1]

            # 加权组合
            tau_0_pred = tau_0_model.predict(X_test)
            tau_1_pred = tau_1_model.predict(X_test)

            cate = g_test * tau_0_pred + (1 - g_test) * tau_1_pred

        else:
            raise ValueError(f"Unknown method: {method}")

        return cate

    def get_starter_code(self) -> str:
        """返回 starter code"""
        return """
# CATE Prediction Starter Code
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

def predict_cate(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    \"\"\"
    预测个体处理效应 (CATE)

    Parameters
    ----------
    train_df : pd.DataFrame
        训练数据，包含 T (处理), Y (结果), X1-X10 (协变量)
    test_df : pd.DataFrame
        测试数据，包含 T, Y, X1-X10

    Returns
    -------
    cate_predictions : np.ndarray
        测试集上的 CATE 预测，shape: (n_test,)
    \"\"\"

    X_cols = [f'X{i}' for i in range(1, 11)]

    X_train = train_df[X_cols].values
    T_train = train_df['T'].values
    Y_train = train_df['Y'].values

    X_test = test_df[X_cols].values

    # TODO: 实现你的 CATE 预测方法

    # 方法 1: S-Learner
    # 1. 训练 Y ~ X, T
    # 2. 预测 CATE = f(X, T=1) - f(X, T=0)

    # 方法 2: T-Learner
    # 1. 分别训练控制组和处理组模型
    # 2. CATE = model_1(X) - model_0(X)

    # 方法 3: X-Learner (推荐)
    # 1. 训练结果模型
    # 2. 计算伪处理效应
    # 3. 训练效应模型

    # 方法 4: Causal Forest (高级)
    # 使用 econml.dml.CausalForestDML

    # 示例: T-Learner
    model_0 = RandomForestRegressor(n_estimators=100, random_state=42)
    model_1 = RandomForestRegressor(n_estimators=100, random_state=43)

    model_0.fit(X_train[T_train == 0], Y_train[T_train == 0])
    model_1.fit(X_train[T_train == 1], Y_train[T_train == 1])

    cate_predictions = model_1.predict(X_test) - model_0.predict(X_test)

    return cate_predictions

# 示例用法
# cate_preds = predict_cate(train_data, test_data)
# submit(cate_preds)
"""

    def get_metric_description(self) -> dict:
        """返回评估指标说明"""
        return {
            'pehe': 'PEHE (Precision in Estimation of Heterogeneous Effect) = sqrt(MSE(CATE))',
            'ate_bias': 'ATE 偏差 = |平均预测 CATE - 真实 ATE|',
            'correlation': 'CATE 预测与真实值的相关系数',
            'r2': 'R² 决定系数',
            'mae': 'MAE 平均绝对误差',
            'score': '综合得分 (0-100), 基于 PEHE 和相关性'
        }

    def get_detailed_info(self) -> str:
        """返回详细挑战说明"""
        baseline_s = self.get_baseline_predictions('s_learner')
        baseline_t = self.get_baseline_predictions('t_learner')

        pehe_s = np.sqrt(np.mean((baseline_s - self.true_cate_test) ** 2))
        pehe_t = np.sqrt(np.mean((baseline_t - self.true_cate_test) ** 2))

        return f"""
### 挑战说明

**场景**: IHDP 婴儿健康发展项目

预测早期教育干预对每个孩子认知测试得分的影响。

**数据描述**:
- **处理 (T)**: 是否接受早期教育干预 (0/1)
- **结果 (Y)**: 认知测试得分
- **协变量 (X1-X10)**: 孩子和家庭特征

**挑战**:
处理效应具有复杂的异质性! 不同孩子的效应差异很大。
需要准确预测每个个体的 CATE，而不仅仅是平均效应。

**真实 ATE**: {self.true_ate:.4f}

**基线方法性能**:
- S-Learner: PEHE = {pehe_s:.4f}
- T-Learner: PEHE = {pehe_t:.4f}

**评估指标**:
- **PEHE** (主要): 衡量 CATE 预测精度
- **Correlation**: 预测与真实 CATE 的相关性
- **R²**: 解释方差比例
- **ATE Bias**: 平均效应估计偏差

**提示**:
1. 探索特征与处理效应的关系
2. 尝试不同的 Meta-Learner (S/T/X-Learner)
3. 使用非线性模型 (RF, GBM) 捕捉复杂交互
4. 考虑特征工程 (交互项、多项式)
5. 可以尝试 EconML 库的高级方法

**目标**: PEHE < 2.0 即可获得高分!

**Leaderboard**: 查看其他参赛者的成绩和方法
"""
