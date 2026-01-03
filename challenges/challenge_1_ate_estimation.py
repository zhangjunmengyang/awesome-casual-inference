"""
挑战 1: ATE 估计
从观察性数据中估计平均处理效应
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple

from .challenge_base import Challenge, ChallengeResult, ChallengeDataGenerator


class ATEEstimationChallenge(Challenge):
    """
    ATE 估计挑战

    任务: 给定观察性数据，估计平均处理效应 (ATE)

    场景: LaLonde 职业培训数据
    - 处理: 参加职业培训
    - 结果: 培训后的收入
    - 挑战: 存在混淆偏差 (低收入、低教育者更可能参加培训)

    评估指标:
    - 主要: ATE 相对误差
    - 次要: 绝对误差、置信区间覆盖率

    难度: Beginner
    """

    def __init__(self):
        super().__init__(
            name="ATE Estimation Challenge",
            description="Estimate the Average Treatment Effect from observational data",
            difficulty="beginner"
        )

        self.confounding_strength = 0.5
        self.true_ate = None
        self.Y0 = None
        self.Y1 = None

    def generate_data(self, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        生成训练和测试数据

        训练集: 1500 样本
        测试集: 500 样本
        """
        # 训练数据
        train_df, train_Y0, train_Y1, train_ate = (
            ChallengeDataGenerator.generate_lalonde_style_data(
                n=1500,
                seed=seed,
                confounding_strength=self.confounding_strength
            )
        )

        # 测试数据
        test_df, test_Y0, test_Y1, test_ate = (
            ChallengeDataGenerator.generate_lalonde_style_data(
                n=500,
                seed=seed + 1000,
                confounding_strength=self.confounding_strength
            )
        )

        # 存储真实值
        self.true_ate = test_ate
        self.Y0 = test_Y0
        self.Y1 = test_Y1
        self.true_targets = test_Y1 - test_Y0  # ITE

        self.train_data = train_df
        self.test_data = test_df

        return train_df, test_df

    def evaluate(
        self,
        predictions: np.ndarray,
        user_name: str = "Anonymous"
    ) -> ChallengeResult:
        """
        评估 ATE 估计

        predictions: 单一值或数组
        - 如果是单一值: 直接作为 ATE 估计
        - 如果是数组: 计算均值作为 ATE 估计
        """
        # 验证输入
        if isinstance(predictions, (int, float)):
            ate_estimate = float(predictions)
        elif isinstance(predictions, np.ndarray):
            if predictions.size == 1:
                ate_estimate = float(predictions)
            else:
                # 假设是 CATE 预测，计算平均值
                ate_estimate = float(predictions.mean())
        else:
            raise ValueError("Predictions must be a number or numpy array")

        # 计算误差
        absolute_error = abs(ate_estimate - self.true_ate)
        relative_error = absolute_error / abs(self.true_ate) if self.true_ate != 0 else float('inf')

        # 次要指标
        secondary_metrics = {
            'absolute_error': absolute_error,
            'relative_error': relative_error,
            'estimated_ate': ate_estimate,
            'true_ate': self.true_ate,
            'bias': ate_estimate - self.true_ate
        }

        # 计算得分 (0-100)
        # 得分基于相对误差: 0% 误差 = 100分, 50% 误差 = 0分
        score = max(0, 100 * (1 - relative_error / 0.5))

        result = ChallengeResult(
            challenge_name=self.name,
            user_name=user_name,
            submission_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            primary_metric=relative_error,
            secondary_metrics=secondary_metrics,
            score=score
        )

        return result

    def get_baseline_predictions(self, method: str = 'naive') -> float:
        """
        获取基线方法的 ATE 估计

        Methods:
        - 'naive': E[Y|T=1] - E[Y|T=0]
        - 'ipw': Inverse Propensity Weighting
        - 'matching': Simple matching estimator
        """
        if self.train_data is None:
            raise ValueError("Data not generated. Call generate_data() first.")

        X_cols = ['age', 'education', 're74', 're75', 'black', 'hispanic', 'married']
        T = self.train_data['T'].values
        Y = self.train_data['Y'].values

        if method == 'naive':
            # 朴素估计
            ate = Y[T == 1].mean() - Y[T == 0].mean()

        elif method == 'ipw':
            # IPW 估计
            from sklearn.linear_model import LogisticRegression

            X = self.train_data[X_cols].values
            ps_model = LogisticRegression(max_iter=1000, random_state=42)
            ps_model.fit(X, T)
            ps = ps_model.predict_proba(X)[:, 1]

            # 避免极端值
            ps = np.clip(ps, 0.01, 0.99)

            # IPW
            weights_1 = T / ps
            weights_0 = (1 - T) / (1 - ps)

            ate = (Y * weights_1).sum() / weights_1.sum() - (Y * weights_0).sum() / weights_0.sum()

        elif method == 'matching':
            # 简单的最近邻匹配
            from sklearn.neighbors import NearestNeighbors

            X = self.train_data[X_cols].values
            X_treated = X[T == 1]
            Y_treated = Y[T == 1]
            X_control = X[T == 0]
            Y_control = Y[T == 0]

            # 为每个处理组样本找最近的控制组样本
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(X_control)
            distances, indices = nn.kneighbors(X_treated)

            matched_control = Y_control[indices.flatten()]
            ate = (Y_treated - matched_control).mean()

        else:
            raise ValueError(f"Unknown method: {method}")

        return ate

    def get_starter_code(self) -> str:
        """返回 starter code"""
        return """
# ATE Estimation Starter Code
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

def estimate_ate(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    \"\"\"
    估计平均处理效应

    Parameters
    ----------
    train_df : pd.DataFrame
        训练数据，包含 T (处理), Y (结果), 以及协变量
    test_df : pd.DataFrame
        测试数据

    Returns
    -------
    ate : float
        ATE 估计值
    \"\"\"

    # TODO: 实现你的 ATE 估计方法

    # 方法 1: 朴素估计 (会有偏差!)
    # ate = train_df[train_df['T']==1]['Y'].mean() - train_df[train_df['T']==0]['Y'].mean()

    # 方法 2: IPW
    # 1. 估计倾向得分
    # 2. 使用 IPW 估计 ATE

    # 方法 3: Matching
    # 1. 为处理组找匹配的控制组
    # 2. 计算匹配对的差异

    # 方法 4: 双重稳健 (推荐)
    # 1. 估计倾向得分和结果模型
    # 2. 结合两者进行估计

    ate = 0.0  # 替换为你的估计
    return ate

# 示例用法
# ate_estimate = estimate_ate(train_data, test_data)
# submit(ate_estimate)
"""

    def get_metric_description(self) -> dict:
        """返回评估指标说明"""
        return {
            'relative_error': 'ATE 相对误差 = |估计值 - 真实值| / |真实值|',
            'absolute_error': 'ATE 绝对误差 = |估计值 - 真实值|',
            'bias': '偏差 = 估计值 - 真实值',
            'score': '综合得分 (0-100), 基于相对误差计算'
        }

    def get_detailed_info(self) -> str:
        """返回详细挑战说明"""
        return f"""
### 挑战说明

**场景**: LaLonde 职业培训项目

你需要从观察性数据中估计职业培训对收入的平均因果效应。

**数据描述**:
- **处理 (T)**: 是否参加职业培训 (0/1)
- **结果 (Y)**: 培训后的年收入 (美元)
- **协变量**:
  - age: 年龄
  - education: 受教育年限
  - re74: 1974年收入
  - re75: 1975年收入
  - black: 是否黑人
  - hispanic: 是否西班牙裔
  - married: 是否已婚

**挑战**:
数据存在混淆偏差! 低收入、低教育背景的人更可能参加培训。
朴素的组间均值差会高估培训效果。

**真实 ATE**: {self.true_ate:.2f} 美元/年

**基线方法**:
- 朴素估计 (有偏): {self.get_baseline_predictions('naive'):.2f}
- IPW 估计: {self.get_baseline_predictions('ipw'):.2f}
- Matching 估计: {self.get_baseline_predictions('matching'):.2f}

**评估指标**:
- 主要: 相对误差 (越小越好)
- 次要: 绝对误差、偏差

**提示**:
1. 先探索数据，可视化处理组和控制组的差异
2. 考虑倾向得分方法 (PSM, IPW)
3. 考虑结果模型方法 (回归调整)
4. 双重稳健方法结合两者优势
5. 交叉验证可以帮助选择模型

**目标**: 相对误差 < 10% 即可获得高分!
"""
