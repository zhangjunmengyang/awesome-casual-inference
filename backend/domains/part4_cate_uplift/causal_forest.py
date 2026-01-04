"""
Causal Forest - 因果森林

基于 Wager & Athey (2018) 的因果森林方法，用于估计异质性处理效应。

核心思想:
- 修改的随机森林，专门用于估计 CATE
- 诚实分裂 (Honest Splitting): 训练和估计使用不同数据
- 自适应 Neighborhood: 使用树结构定义相似样本

注意: 完整的因果森林实现需要 econml 库。
这里提供一个简化版本的 T-Learner 作为备选方案。
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

try:
    from econml.grf import CausalForest
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False


class SimpleTLearner:
    """
    简化版 T-Learner (用于因果森林场景)

    当 econml 不可用时的备选方案。
    使用两个独立的随机森林分别拟合处理组和控制组。

    注意: 这不是因果森林，只是基础的 T-Learner 实现。
    真正的因果森林需要诚实分裂和自适应邻域等特性。
    """

    def __init__(self, n_estimators: int = 100, min_samples_leaf: int = 10, random_state: int = 42):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.model_0 = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        self.model_1 = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state + 1
        )

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """训练模型"""
        mask_0 = T == 0
        mask_1 = T == 1

        self.model_0.fit(X[mask_0], Y[mask_0])
        self.model_1.fit(X[mask_1], Y[mask_1])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测 CATE"""
        Y0 = self.model_0.predict(X)
        Y1 = self.model_1.predict(X)
        return Y1 - Y0

    def feature_importances_(self) -> np.ndarray:
        """特征重要性 (平均两个模型)"""
        return (self.model_0.feature_importances_ + self.model_1.feature_importances_) / 2


def get_causal_forest_model(n_trees: int = 100, min_samples_leaf: int = 10, random_state: int = 42):
    """
    获取因果森林模型

    如果 econml 可用，返回真正的 CausalForest
    否则返回 SimpleTLearner 作为备选

    Parameters:
    -----------
    n_trees: 树的数量
    min_samples_leaf: 叶节点最小样本数
    random_state: 随机种子

    Returns:
    --------
    模型实例
    """
    if ECONML_AVAILABLE:
        return CausalForest(
            n_estimators=n_trees,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            verbose=0
        )
    else:
        return SimpleTLearner(
            n_estimators=n_trees,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
