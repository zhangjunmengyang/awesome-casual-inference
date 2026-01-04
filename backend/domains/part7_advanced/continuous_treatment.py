"""连续处理效应模块

实现连续处理的剂量响应函数估计
- 广义倾向得分（GPS）
- 剂量响应函数（DRF）估计
"""

import numpy as np
from typing import Callable, Tuple
from scipy import stats
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class GeneralizedPropensityScore:
    """
    广义倾向得分估计器

    假设: T | X ~ N(X'α, σ²)
    """

    def __init__(self):
        self.model = LinearRegression()
        self.sigma = None

    def fit(self, T: np.ndarray, X: np.ndarray):
        """
        拟合 GPS 模型

        Args:
            T: 处理变量 (n,)
            X: 协变量 (n, p)
        """
        self.model.fit(X, T)
        residuals = T - self.model.predict(X)
        self.sigma = np.std(residuals)
        return self

    def predict(self, T: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        计算 GPS: r(t, X)

        Args:
            T: 处理变量 (n,)
            X: 协变量 (n, p)

        Returns:
            GPS 值 (n,)
        """
        T_pred = self.model.predict(X)
        gps = stats.norm.pdf(T, loc=T_pred, scale=self.sigma)
        return gps


class DoseResponseEstimator:
    """
    剂量响应函数估计器

    实现 Hirano & Imbens (2004) 方法
    """

    def __init__(self, treatment_degree: int = 2, gps_degree: int = 2):
        """
        Args:
            treatment_degree: 处理变量的多项式阶数
            gps_degree: GPS 的多项式阶数
        """
        self.treatment_degree = treatment_degree
        self.gps_degree = gps_degree
        self.gps_estimator = GeneralizedPropensityScore()
        self.outcome_model = None

    def fit(self, T: np.ndarray, X: np.ndarray, Y: np.ndarray):
        """
        拟合剂量响应函数

        Args:
            T: 处理变量 (n,)
            X: 协变量 (n, p)
            Y: 结果变量 (n,)
        """
        # 步骤 1: 估计 GPS
        self.gps_estimator.fit(T, X)
        gps = self.gps_estimator.predict(T, X)

        # 步骤 2: 构造特征 (T, T², r, r², T*r)
        features = []

        # 处理变量的多项式
        for d in range(1, self.treatment_degree + 1):
            features.append(T ** d)

        # GPS 的多项式
        for d in range(1, self.gps_degree + 1):
            features.append(gps ** d)

        # 交互项
        features.append(T * gps)

        Z = np.column_stack(features)

        # 拟合结果模型
        self.outcome_model = LinearRegression()
        self.outcome_model.fit(Z, Y)

        return self

    def predict_drf(self, t_grid: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        估计剂量响应函数 μ(t)

        Args:
            t_grid: 处理水平网格 (m,)
            X: 协变量（用于计算 GPS） (n, p)

        Returns:
            每个 t 的估计值 (m,)
        """
        drf_estimates = []

        for t in t_grid:
            # 对每个个体计算 E[Y|T=t, r(t,Xi)]
            T_temp = np.full(len(X), t)
            gps_temp = self.gps_estimator.predict(T_temp, X)

            # 构造特征
            features = []
            for d in range(1, self.treatment_degree + 1):
                features.append(T_temp ** d)
            for d in range(1, self.gps_degree + 1):
                features.append(gps_temp ** d)
            features.append(T_temp * gps_temp)

            Z_temp = np.column_stack(features)

            # 预测并边际化
            Y_pred = self.outcome_model.predict(Z_temp)
            drf_estimates.append(Y_pred.mean())

        return np.array(drf_estimates)


def estimate_drf_spline(
    T: np.ndarray,
    Y: np.ndarray,
    gps: np.ndarray,
    t_grid: np.ndarray,
    smoothing: float = 1000
) -> np.ndarray:
    """
    使用样条回归估计剂量响应函数

    Args:
        T: 处理变量
        Y: 结果变量
        gps: 广义倾向得分
        t_grid: 预测网格
        smoothing: 平滑参数

    Returns:
        DRF 估计值
    """
    # GPS 权重
    weights = 1.0 / np.clip(gps, 1e-6, np.inf)

    # 排序（样条需要有序输入）
    sorted_idx = np.argsort(T)
    T_sorted = T[sorted_idx]
    Y_sorted = Y[sorted_idx]
    weights_sorted = weights[sorted_idx]

    # 拟合样条
    spline = UnivariateSpline(
        T_sorted, Y_sorted, w=weights_sorted, s=smoothing, k=3
    )

    # 预测
    drf_estimates = spline(t_grid)

    return drf_estimates


def compute_marginal_effect(t_grid: np.ndarray, drf_values: np.ndarray) -> np.ndarray:
    """
    计算边际效应: dμ/dt

    Args:
        t_grid: 处理网格
        drf_values: DRF 值

    Returns:
        边际效应
    """
    marginal = np.gradient(drf_values, t_grid)
    return marginal


def find_optimal_treatment(
    t_grid: np.ndarray,
    drf_values: np.ndarray
) -> Tuple[float, float]:
    """
    找到最优处理水平

    Args:
        t_grid: 处理网格
        drf_values: DRF 值

    Returns:
        (最优处理水平, 最大结果值)
    """
    optimal_idx = np.argmax(drf_values)
    optimal_t = t_grid[optimal_idx]
    max_value = drf_values[optimal_idx]

    return optimal_t, max_value
