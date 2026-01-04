"""
Multi-Armed Bandits
多臂老虎机算法

核心思想:
--------
在探索（Exploration）和利用（Exploitation）之间权衡
动态分配流量，最大化累积收益

算法对比:
--------
1. Epsilon-Greedy: 简单但有效
2. Thompson Sampling: 贝叶斯最优
3. UCB: 理论保证
4. Contextual Bandits: 个性化推荐

vs A/B Testing:
--------------
A/B Testing: 固定分流，明确因果
Bandits: 动态优化，最大化收益

适用场景:
--------
1. 在线广告：实时优化点击率
2. 推荐系统：动态调整推荐策略
3. 定价优化：快速收敛到最优价格
4. 内容测试：多个版本同时测试

面试考点:
--------
- Explore vs Exploit权衡
- Regret的定义和计算
- Thompson Sampling的原理
- Contextual Bandits的应用
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Arm:
    """老虎机臂"""

    def __init__(self, true_mean: float, name: str = None):
        """
        Parameters:
        -----------
        true_mean: 真实期望收益
        name: 臂的名称
        """
        self.true_mean = true_mean
        self.name = name or f"Arm(μ={true_mean:.2f})"
        self.pulls = 0
        self.total_reward = 0
        self.estimated_mean = 0

    def pull(self) -> float:
        """拉动该臂，返回收益（加噪声）"""
        reward = np.random.normal(self.true_mean, 1.0)
        self.pulls += 1
        self.total_reward += reward
        self.estimated_mean = self.total_reward / self.pulls
        return reward


class EpsilonGreedy:
    """Epsilon-Greedy算法"""

    def __init__(self, epsilon: float = 0.1):
        """
        Parameters:
        -----------
        epsilon: 探索概率
        """
        self.epsilon = epsilon
        self.name = f"ε-Greedy(ε={epsilon})"

    def select_arm(self, arms: List[Arm]) -> int:
        """
        选择臂

        Parameters:
        -----------
        arms: 臂列表

        Returns:
        --------
        选中的臂索引
        """
        # ε概率探索，(1-ε)概率利用
        if np.random.random() < self.epsilon:
            # 探索：随机选择
            return np.random.randint(len(arms))
        else:
            # 利用：选择当前最优
            # 对于未拉过的臂，给予无穷大的估计值以确保被尝试
            estimated_means = [
                arm.estimated_mean if arm.pulls > 0 else float('inf')
                for arm in arms
            ]
            return np.argmax(estimated_means)


class ThompsonSampling:
    """Thompson Sampling算法（贝叶斯）"""

    def __init__(self, prior_mean: float = 0, prior_std: float = 1.0):
        """
        Parameters:
        -----------
        prior_mean: 先验均值
        prior_std: 先验标准差
        """
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.name = "Thompson Sampling"

        # 每个臂的后验分布参数（正态-正态共轭）
        self.posterior_means = None
        self.posterior_stds = None

    def initialize(self, n_arms: int):
        """初始化后验分布"""
        self.posterior_means = np.full(n_arms, self.prior_mean)
        self.posterior_stds = np.full(n_arms, self.prior_std)

    def select_arm(self, arms: List[Arm]) -> int:
        """
        选择臂

        从每个臂的后验分布中采样，选择最大值

        Parameters:
        -----------
        arms: 臂列表

        Returns:
        --------
        选中的臂索引
        """
        if self.posterior_means is None:
            self.initialize(len(arms))

        # 从每个臂的后验分布采样
        samples = [
            np.random.normal(self.posterior_means[i], self.posterior_stds[i])
            for i in range(len(arms))
        ]

        return np.argmax(samples)

    def update(self, arm_index: int, reward: float, reward_std: float = 1.0):
        """
        更新后验分布

        正态-正态共轭更新

        Parameters:
        -----------
        arm_index: 臂索引
        reward: 观测到的收益
        reward_std: 收益的标准差（假设已知）
        """
        # 先验
        prior_mean = self.posterior_means[arm_index]
        prior_var = self.posterior_stds[arm_index] ** 2

        # 似然
        likelihood_var = reward_std ** 2

        # 后验（正态-正态共轭）
        posterior_var = 1 / (1 / prior_var + 1 / likelihood_var)
        posterior_mean = posterior_var * (prior_mean / prior_var + reward / likelihood_var)

        self.posterior_means[arm_index] = posterior_mean
        self.posterior_stds[arm_index] = np.sqrt(posterior_var)


class UCB:
    """Upper Confidence Bound算法"""

    def __init__(self, confidence: float = 2.0):
        """
        Parameters:
        -----------
        confidence: 置信度参数（通常取2）
        """
        self.confidence = confidence
        self.name = f"UCB(c={confidence})"

    def select_arm(self, arms: List[Arm], total_pulls: int) -> int:
        """
        选择臂

        选择 UCB 最大的臂:
        UCB_i = μ̂_i + c * sqrt(ln(t) / n_i)

        Parameters:
        -----------
        arms: 臂列表
        total_pulls: 总拉动次数

        Returns:
        --------
        选中的臂索引
        """
        ucb_values = []

        for arm in arms:
            if arm.pulls == 0:
                # 未拉过的臂，给予无穷大的UCB
                ucb_values.append(float('inf'))
            else:
                # UCB公式
                exploration_bonus = self.confidence * np.sqrt(
                    np.log(total_pulls + 1) / arm.pulls
                )
                ucb = arm.estimated_mean + exploration_bonus
                ucb_values.append(ucb)

        return np.argmax(ucb_values)


def simulate_bandit(
    arms: List[Arm],
    algorithm,
    n_rounds: int = 1000,
    seed: int = 42
) -> Tuple[List[Dict], float]:
    """
    模拟Bandit算法

    Parameters:
    -----------
    arms: 臂列表
    algorithm: Bandit算法对象
    n_rounds: 模拟轮数
    seed: 随机种子

    Returns:
    --------
    (每轮记录列表, 总Regret)
    """
    np.random.seed(seed)

    # 重置臂
    for arm in arms:
        arm.pulls = 0
        arm.total_reward = 0
        arm.estimated_mean = 0

    # 最优臂
    best_arm = max(arms, key=lambda a: a.true_mean)
    optimal_mean = best_arm.true_mean

    history = []
    cumulative_regret = 0
    cumulative_reward = 0

    for t in range(n_rounds):
        # 选择臂
        if isinstance(algorithm, UCB):
            chosen_index = algorithm.select_arm(arms, t)
        else:
            chosen_index = algorithm.select_arm(arms)

        # 拉动
        chosen_arm = arms[chosen_index]
        reward = chosen_arm.pull()

        # Thompson Sampling需要更新后验
        if isinstance(algorithm, ThompsonSampling):
            algorithm.update(chosen_index, reward)

        # 计算Regret
        instant_regret = optimal_mean - chosen_arm.true_mean
        cumulative_regret += instant_regret
        cumulative_reward += reward

        # 记录
        history.append({
            'round': t,
            'chosen_arm': chosen_index,
            'arm_name': chosen_arm.name,
            'reward': reward,
            'instant_regret': instant_regret,
            'cumulative_regret': cumulative_regret,
            'cumulative_reward': cumulative_reward
        })

    return history, cumulative_regret


def compare_algorithms(
    arms: List[Arm],
    algorithms: List,
    n_rounds: int = 1000,
    n_simulations: int = 100,
    seed: int = 42
) -> pd.DataFrame:
    """
    比较多个算法

    Parameters:
    -----------
    arms: 臂列表
    algorithms: 算法列表
    n_rounds: 每次模拟的轮数
    n_simulations: 模拟次数
    seed: 随机种子

    Returns:
    --------
    比较结果DataFrame
    """
    results = []

    for algo in algorithms:
        total_regrets = []

        for sim in range(n_simulations):
            # 创建新的臂实例（避免状态污染）
            arms_copy = [
                Arm(arm.true_mean, arm.name)
                for arm in arms
            ]

            _, regret = simulate_bandit(
                arms_copy,
                algo,
                n_rounds=n_rounds,
                seed=seed + sim
            )

            total_regrets.append(regret)

        results.append({
            'algorithm': algo.name,
            'mean_regret': np.mean(total_regrets),
            'std_regret': np.std(total_regrets),
            'min_regret': np.min(total_regrets),
            'max_regret': np.max(total_regrets)
        })

    return pd.DataFrame(results)


def plot_bandit_simulation(
    history: List[Dict],
    arms: List[Arm]
) -> go.Figure:
    """
    可视化Bandit模拟过程

    Parameters:
    -----------
    history: 模拟历史记录
    arms: 臂列表

    Returns:
    --------
    Plotly图表
    """
    df = pd.DataFrame(history)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Cumulative Regret',
            'Cumulative Reward',
            'Arm Selection Over Time',
            'Final Arm Pulls'
        )
    )

    # 1. 累积Regret
    fig.add_trace(
        go.Scatter(
            x=df['round'],
            y=df['cumulative_regret'],
            mode='lines',
            line=dict(color='#EB5757', width=2),
            name='Cumulative Regret'
        ),
        row=1, col=1
    )

    # 2. 累积收益
    fig.add_trace(
        go.Scatter(
            x=df['round'],
            y=df['cumulative_reward'],
            mode='lines',
            line=dict(color='#27AE60', width=2),
            name='Cumulative Reward'
        ),
        row=1, col=2
    )

    # 3. 臂选择时间线
    for i, arm in enumerate(arms):
        arm_rounds = df[df['chosen_arm'] == i]['round']
        fig.add_trace(
            go.Scatter(
                x=arm_rounds,
                y=[i] * len(arm_rounds),
                mode='markers',
                marker=dict(size=3, opacity=0.5),
                name=arm.name,
                showlegend=False
            ),
            row=2, col=1
        )

    # 4. 最终拉动次数
    arm_pulls = [arm.pulls for arm in arms]
    arm_names = [arm.name for arm in arms]

    fig.add_trace(
        go.Bar(
            x=arm_names,
            y=arm_pulls,
            marker_color='#2D9CDB',
            text=arm_pulls,
            textposition='outside',
            showlegend=False
        ),
        row=2, col=2
    )

    fig.update_layout(
        title='Bandit Simulation Results',
        template='plotly_white',
        height=800,
        showlegend=True
    )

    fig.update_xaxes(title_text='Round', row=1, col=1)
    fig.update_xaxes(title_text='Round', row=1, col=2)
    fig.update_xaxes(title_text='Round', row=2, col=1)
    fig.update_xaxes(title_text='Arm', row=2, col=2)

    fig.update_yaxes(title_text='Regret', row=1, col=1)
    fig.update_yaxes(title_text='Reward', row=1, col=2)
    fig.update_yaxes(title_text='Arm Index', row=2, col=1)
    fig.update_yaxes(title_text='Pulls', row=2, col=2)

    return fig


def plot_algorithm_comparison(
    comparison_df: pd.DataFrame
) -> go.Figure:
    """
    可视化算法比较

    Parameters:
    -----------
    comparison_df: 算法比较结果

    Returns:
    --------
    Plotly图表
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=comparison_df['algorithm'],
        y=comparison_df['mean_regret'],
        error_y=dict(type='data', array=comparison_df['std_regret']),
        marker_color='#2D9CDB',
        text=[f'{r:.1f}' for r in comparison_df['mean_regret']],
        textposition='outside'
    ))

    fig.update_layout(
        title='Algorithm Comparison: Mean Cumulative Regret',
        xaxis_title='Algorithm',
        yaxis_title='Mean Cumulative Regret',
        template='plotly_white',
        height=400
    )

    return fig


if __name__ == "__main__":
    # 测试代码
    # 创建臂
    arms = [
        Arm(true_mean=0.5, name="Arm A (μ=0.5)"),
        Arm(true_mean=0.6, name="Arm B (μ=0.6)"),
        Arm(true_mean=0.55, name="Arm C (μ=0.55)"),
        Arm(true_mean=0.7, name="Arm D (μ=0.7)"),  # 最优
    ]

    # 测试Thompson Sampling
    algo = ThompsonSampling()
    history, regret = simulate_bandit(arms, algo, n_rounds=1000)

    print(f"Thompson Sampling:")
    print(f"  Cumulative Regret: {regret:.2f}")
    print(f"  Final Pulls: {[arm.pulls for arm in arms]}")

    # 比较算法
    algorithms = [
        EpsilonGreedy(epsilon=0.1),
        ThompsonSampling(),
        UCB(confidence=2.0)
    ]

    arms_reset = [
        Arm(true_mean=0.5, name="Arm A"),
        Arm(true_mean=0.6, name="Arm B"),
        Arm(true_mean=0.55, name="Arm C"),
        Arm(true_mean=0.7, name="Arm D"),
    ]

    comparison = compare_algorithms(arms_reset, algorithms, n_rounds=1000, n_simulations=50)
    print("\nAlgorithm Comparison:")
    print(comparison[['algorithm', 'mean_regret', 'std_regret']])
