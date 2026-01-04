"""营销归因模块

实现多种营销归因方法：
- 规则归因（Last-touch, First-touch, Linear, Time-decay, Position-based）
- Shapley Value 归因
- Markov Chain 归因
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from collections import defaultdict, Counter
from itertools import chain, combinations
import math


class RuleBasedAttribution:
    """规则归因模型"""

    @staticmethod
    def last_touch(path: List[str]) -> Dict[str, float]:
        """Last-touch 归因：100% 归给最后一个触点"""
        attribution = {}
        for channel in set(path):
            attribution[channel] = 1.0 if channel == path[-1] else 0.0
        return attribution

    @staticmethod
    def first_touch(path: List[str]) -> Dict[str, float]:
        """First-touch 归因：100% 归给第一个触点"""
        attribution = {}
        for channel in set(path):
            attribution[channel] = 1.0 if channel == path[0] else 0.0
        return attribution

    @staticmethod
    def linear(path: List[str]) -> Dict[str, float]:
        """Linear 归因：平均分配权重"""
        attribution = defaultdict(float)
        weight_per_touch = 1.0 / len(path)
        for channel in path:
            attribution[channel] += weight_per_touch
        return dict(attribution)

    @staticmethod
    def time_decay(path: List[str], decay_rate: float = 0.7) -> Dict[str, float]:
        """Time-decay 归因：越接近转化，权重越高

        Args:
            path: 转化路径
            decay_rate: 衰减率，越大则最后触点权重越大
        """
        n = len(path)
        attribution = defaultdict(float)

        # 计算权重
        weights = []
        for i in range(n):
            weight = np.exp(-decay_rate * (n - 1 - i))
            weights.append(weight)

        # 归一化
        total = sum(weights)
        weights = [w / total for w in weights]

        # 累加同一渠道的权重
        for i, channel in enumerate(path):
            attribution[channel] += weights[i]

        return dict(attribution)

    @staticmethod
    def position_based(
        path: List[str],
        first_weight: float = 0.4,
        last_weight: float = 0.4
    ) -> Dict[str, float]:
        """Position-based 归因：首末各占固定比例，中间平分剩余

        Args:
            path: 转化路径
            first_weight: 第一个触点的权重（默认 0.4）
            last_weight: 最后一个触点的权重（默认 0.4）
        """
        n = len(path)
        attribution = defaultdict(float)

        if n == 1:
            attribution[path[0]] = 1.0
        elif n == 2:
            attribution[path[0]] = 0.5
            attribution[path[1]] = 0.5
        else:
            # 第一个触点
            attribution[path[0]] += first_weight
            # 最后一个触点
            attribution[path[-1]] += last_weight
            # 中间触点平分剩余权重
            middle_weight = (1.0 - first_weight - last_weight) / (n - 2)
            for i in range(1, n - 1):
                attribution[path[i]] += middle_weight

        return dict(attribution)


class ShapleyAttribution:
    """Shapley Value 归因

    基于合作博弈论的归因方法，公平地分配每个渠道的贡献。
    """

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: 包含 'path_list' 和 'revenue' 列的 DataFrame
        """
        self.df = df
        self.all_channels = set()
        for path in df['path_list']:
            self.all_channels.update(path)

        # 构建价值函数
        self.value_function = self._build_value_function()

    def _build_value_function(self) -> Dict[frozenset, float]:
        """构建价值函数 v(S)

        v(S) = 包含子集 S 中所有渠道的路径的平均收入
        """
        value_func = {}

        def powerset(iterable):
            """生成幂集"""
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

        for subset in powerset(self.all_channels):
            subset_set = frozenset(subset)

            if len(subset_set) == 0:
                value_func[subset_set] = 0.0
                continue

            # 找到包含这个子集所有渠道的路径
            matching_revenues = []
            for idx, row in self.df.iterrows():
                path_channels = set(row['path_list'])
                if subset_set.issubset(path_channels):
                    matching_revenues.append(row['revenue'])

            # 计算平均收入
            value_func[subset_set] = np.mean(matching_revenues) if matching_revenues else 0.0

        return value_func

    def compute_shapley(self, channel: str) -> float:
        """计算单个渠道的 Shapley Value

        公式: φ_i = Σ_{S ⊆ N\{i}} [|S|!(|N|-|S|-1)! / |N|!] * [v(S∪{i}) - v(S)]
        """
        n = len(self.all_channels)
        other_channels = self.all_channels - {channel}

        shapley_value = 0.0

        def powerset(iterable):
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

        # 枚举所有不包含当前渠道的子集
        for subset in powerset(other_channels):
            subset_set = frozenset(subset)
            s_size = len(subset_set)

            # 计算权重
            weight = math.factorial(s_size) * math.factorial(n - s_size - 1) / math.factorial(n)

            # 计算边际贡献
            v_with = self.value_function.get(subset_set | {channel}, 0.0)
            v_without = self.value_function.get(subset_set, 0.0)
            marginal_contribution = v_with - v_without

            shapley_value += weight * marginal_contribution

        return shapley_value

    def compute_all_shapley(self) -> Dict[str, float]:
        """计算所有渠道的 Shapley Value"""
        shapley_values = {}
        for channel in self.all_channels:
            shapley_values[channel] = self.compute_shapley(channel)
        return shapley_values


class MarkovAttribution:
    """Markov Chain 归因模型

    基于马尔可夫链的移除效应（Removal Effect）方法。
    """

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: 包含 'path_list' 列的 DataFrame
        """
        self.df = df
        self.channels = set()
        for path in df['path_list']:
            self.channels.update(path)

        # 构建转移矩阵
        self.transition_matrix = self._build_transition_matrix()

    def _build_transition_matrix(
        self,
        excluded_channel: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """构建转移概率矩阵

        Args:
            excluded_channel: 要排除的渠道（用于计算移除效应）

        Returns:
            转移概率字典 {from_state: {to_state: probability}}
        """
        # 统计转移次数
        transitions = defaultdict(lambda: defaultdict(int))

        for idx, row in self.df.iterrows():
            path = row['path_list']

            # 过滤掉被排除的渠道
            if excluded_channel:
                path = [ch for ch in path if ch != excluded_channel]

            if len(path) == 0:
                continue

            # Start -> 第一个渠道
            transitions['Start'][path[0]] += 1

            # 渠道之间的转移
            for i in range(len(path) - 1):
                transitions[path[i]][path[i + 1]] += 1

            # 最后一个渠道 -> Conversion
            transitions[path[-1]]['Conversion'] += 1

        # 转换为概率（归一化）
        transition_probs = {}
        for from_state, to_states in transitions.items():
            total = sum(to_states.values())
            transition_probs[from_state] = {
                to_state: count / total for to_state, count in to_states.items()
            }

        return transition_probs

    def _conversion_probability(
        self,
        transition_matrix: Dict[str, Dict[str, float]],
        max_steps: int = 20,
        n_simulations: int = 10000
    ) -> float:
        """计算从 Start 到 Conversion 的概率（使用蒙特卡洛模拟）

        Args:
            transition_matrix: 转移概率矩阵
            max_steps: 最大步数
            n_simulations: 模拟次数

        Returns:
            转化概率
        """
        conversions = 0

        for _ in range(n_simulations):
            state = 'Start'
            for step in range(max_steps):
                if state == 'Conversion':
                    conversions += 1
                    break

                if state not in transition_matrix or len(transition_matrix[state]) == 0:
                    break

                # 根据转移概率选择下一个状态
                next_states = list(transition_matrix[state].keys())
                probs = list(transition_matrix[state].values())
                state = np.random.choice(next_states, p=probs)

        return conversions / n_simulations

    def compute_removal_effects(self) -> Dict[str, float]:
        """计算每个渠道的移除效应

        移除效应 = (基准转化率 - 移除该渠道后的转化率) / 基准转化率

        Returns:
            每个渠道的移除效应
        """
        # 基准转化率（所有渠道都在）
        base_conversion_prob = self._conversion_probability(self.transition_matrix)

        removal_effects = {}

        for channel in self.channels:
            # 移除该渠道后重建转移矩阵
            transition_without = self._build_transition_matrix(excluded_channel=channel)

            # 计算移除后的转化率
            conversion_without = self._conversion_probability(transition_without)

            # 移除效应 = 转化率的下降比例
            if base_conversion_prob > 0:
                removal_effect = (base_conversion_prob - conversion_without) / base_conversion_prob
            else:
                removal_effect = 0.0

            removal_effects[channel] = max(removal_effect, 0.0)  # 避免负值

        return removal_effects


class MarketingAttribution:
    """营销归因统一接口

    整合多种归因方法，提供统一的调用接口。
    """

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: 包含 'path_list' 和 'revenue' 列的 DataFrame
        """
        self.df = df
        self.shapley_model = ShapleyAttribution(df)
        self.markov_model = MarkovAttribution(df)

    def apply_attribution(
        self,
        method: str = 'last_touch',
        **kwargs
    ) -> pd.DataFrame:
        """应用归因方法

        Args:
            method: 归因方法名称
                - 'last_touch': Last-touch 归因
                - 'first_touch': First-touch 归因
                - 'linear': Linear 归因
                - 'time_decay': Time-decay 归因
                - 'position_based': Position-based 归因
                - 'shapley': Shapley Value 归因
                - 'markov': Markov Chain 归因
            **kwargs: 方法特定的参数

        Returns:
            DataFrame 包含渠道级别的归因结果
        """
        channel_attribution = defaultdict(float)
        channel_revenue = defaultdict(float)
        channel_conversions = defaultdict(int)

        if method == 'shapley':
            # Shapley 归因
            shapley_values = self.shapley_model.compute_all_shapley()
            total_revenue = self.df['revenue'].sum()

            for channel, value in shapley_values.items():
                channel_attribution[channel] = value
                # 按 Shapley 值比例分配收入
                total_shapley = sum(shapley_values.values())
                if total_shapley > 0:
                    channel_revenue[channel] = (value / total_shapley) * total_revenue
                else:
                    channel_revenue[channel] = 0

        elif method == 'markov':
            # Markov 归因
            removal_effects = self.markov_model.compute_removal_effects()
            total_revenue = self.df['revenue'].sum()
            total_removal = sum(removal_effects.values())

            for channel, effect in removal_effects.items():
                channel_attribution[channel] = effect
                if total_removal > 0:
                    channel_revenue[channel] = (effect / total_removal) * total_revenue
                else:
                    channel_revenue[channel] = 0

        else:
            # 规则归因
            for idx, row in self.df.iterrows():
                path = row['path_list']
                revenue = row['revenue']

                # 选择归因方法
                if method == 'last_touch':
                    weights = RuleBasedAttribution.last_touch(path)
                elif method == 'first_touch':
                    weights = RuleBasedAttribution.first_touch(path)
                elif method == 'linear':
                    weights = RuleBasedAttribution.linear(path)
                elif method == 'time_decay':
                    weights = RuleBasedAttribution.time_decay(path, **kwargs)
                elif method == 'position_based':
                    weights = RuleBasedAttribution.position_based(path, **kwargs)
                else:
                    raise ValueError(f"Unknown method: {method}")

                # 分配收入
                for channel, weight in weights.items():
                    channel_revenue[channel] += revenue * weight
                    channel_attribution[channel] += weight
                    if weight > 0:
                        channel_conversions[channel] += 1

        # 创建结果 DataFrame
        result_df = pd.DataFrame({
            'Channel': list(channel_revenue.keys()),
            'Attributed_Revenue': list(channel_revenue.values()),
            'Attributed_Conversions': [channel_attribution[ch] for ch in channel_revenue.keys()],
            'Touch_Count': [channel_conversions.get(ch, 0) for ch in channel_revenue.keys()]
        }).sort_values('Attributed_Revenue', ascending=False)

        total_revenue = result_df['Attributed_Revenue'].sum()
        if total_revenue > 0:
            result_df['Revenue_Share'] = result_df['Attributed_Revenue'] / total_revenue
        else:
            result_df['Revenue_Share'] = 0

        return result_df
