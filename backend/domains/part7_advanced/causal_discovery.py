"""因果发现模块

实现基于约束和评分的因果发现算法
- PC 算法（基于条件独立性）
- 简化的结构学习
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
from itertools import combinations
from scipy import stats
from sklearn.linear_model import LinearRegression


class SimplifiedPCAlgorithm:
    """
    简化的 PC 算法实现

    基于条件独立性检验学习因果图骨架
    """

    def __init__(self, alpha: float = 0.05, max_cond_size: int = 3):
        """
        Args:
            alpha: 显著性水平
            max_cond_size: 最大条件集大小
        """
        self.alpha = alpha
        self.max_cond_size = max_cond_size
        self.adjacency = {}
        self.separation_sets = {}

    def _independence_test(
        self,
        data: pd.DataFrame,
        X: str,
        Y: str,
        Z: List[str] = None
    ) -> bool:
        """
        条件独立性检验: X ⊥ Y | Z

        Args:
            data: 数据框
            X, Y: 变量名
            Z: 条件变量列表

        Returns:
            是否独立（True 表示独立）
        """
        if Z is None or len(Z) == 0:
            # 边缘独立性
            corr, pval = stats.pearsonr(data[X], data[Y])
            return pval > self.alpha
        else:
            # 条件独立性（使用偏相关）
            Z_data = data[Z].values

            # 回归残差化
            reg_X = LinearRegression().fit(Z_data, data[X])
            reg_Y = LinearRegression().fit(Z_data, data[Y])

            res_X = data[X] - reg_X.predict(Z_data)
            res_Y = data[Y] - reg_Y.predict(Z_data)

            corr, pval = stats.pearsonr(res_X, res_Y)
            return pval > self.alpha

    def fit(self, data: pd.DataFrame) -> Dict[str, Set[str]]:
        """
        学习因果图骨架

        Args:
            data: 数据框

        Returns:
            邻接字典
        """
        variables = list(data.columns)
        n_vars = len(variables)

        # 初始化完全图
        self.adjacency = {var: set(variables) - {var} for var in variables}
        self.separation_sets = {}

        # 逐步增加条件集大小
        for cond_size in range(self.max_cond_size + 1):
            for X in variables:
                for Y in list(self.adjacency[X]):
                    if Y not in self.adjacency[X]:
                        continue

                    # 候选条件集
                    candidates = self.adjacency[X] - {Y}

                    if len(candidates) < cond_size:
                        continue

                    # 检验所有大小为 cond_size 的条件集
                    for Z in combinations(candidates, cond_size):
                        Z_list = list(Z)
                        if self._independence_test(data, X, Y, Z_list):
                            # 找到分离集，删边
                            self.adjacency[X].discard(Y)
                            self.adjacency[Y].discard(X)
                            self.separation_sets[(X, Y)] = Z_list
                            self.separation_sets[(Y, X)] = Z_list
                            break

        return self.adjacency

    def get_edges(self) -> List[Tuple[str, str]]:
        """
        获取无向边列表

        Returns:
            边列表（每条边只出现一次）
        """
        edges = []
        seen = set()

        for src, neighbors in self.adjacency.items():
            for dst in neighbors:
                edge = tuple(sorted([src, dst]))
                if edge not in seen:
                    edges.append(edge)
                    seen.add(edge)

        return edges


def discover_causal_structure(
    data: pd.DataFrame,
    alpha: float = 0.05,
    max_cond_size: int = 2
) -> Dict:
    """
    发现数据的因果结构

    Args:
        data: 数据框
        alpha: 显著性水平
        max_cond_size: 最大条件集大小

    Returns:
        包含邻接关系、边和分离集的字典
    """
    pc = SimplifiedPCAlgorithm(alpha=alpha, max_cond_size=max_cond_size)
    adjacency = pc.fit(data)
    edges = pc.get_edges()

    return {
        'adjacency': adjacency,
        'edges': edges,
        'separation_sets': pc.separation_sets,
        'n_edges': len(edges)
    }


def evaluate_discovery_performance(
    discovered_edges: List[Tuple[str, str]],
    true_edges: List[Tuple[str, str]]
) -> Dict[str, float]:
    """
    评估因果发现性能

    Args:
        discovered_edges: 发现的边
        true_edges: 真实的边

    Returns:
        包含 precision, recall, f1 的字典
    """
    # 转换为集合（边无向化）
    discovered_set = {tuple(sorted(edge)) for edge in discovered_edges}
    true_set = {tuple(sorted(edge)) for edge in true_edges}

    # 计算指标
    tp = len(discovered_set & true_set)
    fp = len(discovered_set - true_set)
    fn = len(true_set - discovered_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positive': tp,
        'false_positive': fp,
        'false_negative': fn
    }


def compute_structural_hamming_distance(
    discovered_edges: List[Tuple[str, str]],
    true_edges: List[Tuple[str, str]]
) -> int:
    """
    计算结构汉明距离（SHD）

    Args:
        discovered_edges: 发现的边
        true_edges: 真实的边

    Returns:
        SHD 距离
    """
    discovered_set = {tuple(sorted(edge)) for edge in discovered_edges}
    true_set = {tuple(sorted(edge)) for edge in true_edges}

    # SHD = 缺失的边 + 额外的边
    shd = len(true_set - discovered_set) + len(discovered_set - true_set)

    return shd
