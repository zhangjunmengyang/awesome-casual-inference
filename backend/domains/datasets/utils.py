"""
Datasets 工具函数

提供数据集处理、描述统计、可视化等通用功能
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def train_test_split_causal(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    true_ite: Optional[np.ndarray] = None,
    test_size: float = 0.3,
    stratify_treatment: bool = True,
    seed: Optional[int] = 42
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """
    因果推断专用数据划分

    确保训练集和测试集中处理组/对照组比例一致

    Parameters:
    -----------
    X: 协变量矩阵
    T: 处理状态
    Y: 结果变量
    true_ite: 真实 ITE (可选，用于评估)
    test_size: 测试集比例
    stratify_treatment: 是否按处理状态分层抽样
    seed: 随机种子

    Returns:
    --------
    如果 true_ite 为 None:
        (X_train, X_test, T_train, T_test, Y_train, Y_test)
    否则:
        (X_train, X_test, T_train, T_test, Y_train, Y_test, ite_train, ite_test)

    Examples:
    ---------
    >>> from datasets import generate_linear_dgp
    >>> X, T, Y, ite = generate_linear_dgp(n_samples=1000)
    >>> X_tr, X_te, T_tr, T_te, Y_tr, Y_te, ite_tr, ite_te = train_test_split_causal(
    ...     X, T, Y, ite, test_size=0.3
    ... )
    >>> print(f"Train treatment rate: {T_tr.mean():.2%}")
    >>> print(f"Test treatment rate: {T_te.mean():.2%}")
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = len(T)
    indices = np.arange(n_samples)

    if stratify_treatment:
        # 分层抽样 - 按处理状态
        treat_idx = indices[T == 1]
        control_idx = indices[T == 0]

        # 分别划分
        n_test_treat = int(len(treat_idx) * test_size)
        n_test_control = int(len(control_idx) * test_size)

        np.random.shuffle(treat_idx)
        np.random.shuffle(control_idx)

        test_idx = np.concatenate([
            treat_idx[:n_test_treat],
            control_idx[:n_test_control]
        ])
        train_idx = np.concatenate([
            treat_idx[n_test_treat:],
            control_idx[n_test_control:]
        ])

    else:
        # 简单随机划分
        np.random.shuffle(indices)
        n_test = int(n_samples * test_size)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    # 划分数据
    X_train, X_test = X[train_idx], X[test_idx]
    T_train, T_test = T[train_idx], T[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    if true_ite is not None:
        ite_train, ite_test = true_ite[train_idx], true_ite[test_idx]
        return X_train, X_test, T_train, T_test, Y_train, Y_test, ite_train, ite_test
    else:
        return X_train, X_test, T_train, T_test, Y_train, Y_test


def describe_dataset(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    true_ite: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    生成数据集描述统计

    Parameters:
    -----------
    X: 协变量矩阵
    T: 处理状态
    Y: 结果变量
    true_ite: 真实 ITE (可选)
    feature_names: 特征名称 (可选)

    Returns:
    --------
    DataFrame with statistics

    Examples:
    ---------
    >>> from datasets import generate_heterogeneous_dgp
    >>> X, T, Y, ite = generate_heterogeneous_dgp()
    >>> stats = describe_dataset(X, T, Y, ite)
    >>> print(stats)
    """
    n_samples, n_features = X.shape

    # 基本统计
    stats = {
        'Metric': [],
        'Overall': [],
        'Treatment': [],
        'Control': []
    }

    # 样本量
    stats['Metric'].append('Sample Size')
    stats['Overall'].append(n_samples)
    stats['Treatment'].append((T == 1).sum())
    stats['Control'].append((T == 0).sum())

    # 处理率
    stats['Metric'].append('Treatment Rate')
    stats['Overall'].append(f"{T.mean():.2%}")
    stats['Treatment'].append('-')
    stats['Control'].append('-')

    # 结果统计
    stats['Metric'].append('Outcome Mean')
    stats['Overall'].append(f"{Y.mean():.3f}")
    stats['Treatment'].append(f"{Y[T==1].mean():.3f}")
    stats['Control'].append(f"{Y[T==0].mean():.3f}")

    stats['Metric'].append('Outcome Std')
    stats['Overall'].append(f"{Y.std():.3f}")
    stats['Treatment'].append(f"{Y[T==1].std():.3f}")
    stats['Control'].append(f"{Y[T==0].std():.3f}")

    # Naive ATE
    naive_ate = Y[T == 1].mean() - Y[T == 0].mean()
    stats['Metric'].append('Naive ATE')
    stats['Overall'].append(f"{naive_ate:.3f}")
    stats['Treatment'].append('-')
    stats['Control'].append('-')

    # 真实 ITE 统计 (如果提供)
    if true_ite is not None:
        stats['Metric'].append('True ATE')
        stats['Overall'].append(f"{true_ite.mean():.3f}")
        stats['Treatment'].append('-')
        stats['Control'].append('-')

        stats['Metric'].append('ITE Std')
        stats['Overall'].append(f"{true_ite.std():.3f}")
        stats['Treatment'].append('-')
        stats['Control'].append('-')

        stats['Metric'].append('ATE Bias')
        stats['Overall'].append(f"{abs(naive_ate - true_ite.mean()):.3f}")
        stats['Treatment'].append('-')
        stats['Control'].append('-')

    # 特征统计 (前 5 个或全部)
    n_features_show = min(5, n_features)
    for i in range(n_features_show):
        feat_name = feature_names[i] if feature_names else f"X{i+1}"

        stats['Metric'].append(f'{feat_name} Mean')
        stats['Overall'].append(f"{X[:, i].mean():.3f}")
        stats['Treatment'].append(f"{X[T==1, i].mean():.3f}")
        stats['Control'].append(f"{X[T==0, i].mean():.3f}")

    if n_features > n_features_show:
        stats['Metric'].append(f'... ({n_features - n_features_show} more features)')
        stats['Overall'].append('-')
        stats['Treatment'].append('-')
        stats['Control'].append('-')

    return pd.DataFrame(stats)


def check_covariate_balance(
    X: np.ndarray,
    T: np.ndarray,
    feature_names: Optional[List[str]] = None,
    threshold: float = 0.1
) -> pd.DataFrame:
    """
    检查协变量平衡性 (Covariate Balance)

    使用标准化均值差 (Standardized Mean Difference, SMD)

    Parameters:
    -----------
    X: 协变量矩阵
    T: 处理状态
    feature_names: 特征名称
    threshold: SMD 阈值 (通常用 0.1)

    Returns:
    --------
    DataFrame with SMD for each feature

    Examples:
    ---------
    >>> from datasets import generate_linear_dgp
    >>> X, T, Y, _ = generate_linear_dgp(confounding=True)
    >>> balance = check_covariate_balance(X, T)
    >>> print(balance[balance['SMD'] > 0.1])  # 不平衡特征
    """
    n_features = X.shape[1]

    smd_list = []
    balanced_list = []

    for i in range(n_features):
        # 处理组和对照组均值
        mean_treat = X[T == 1, i].mean()
        mean_control = X[T == 0, i].mean()

        # 合并标准差
        std_treat = X[T == 1, i].std()
        std_control = X[T == 0, i].std()
        pooled_std = np.sqrt((std_treat**2 + std_control**2) / 2)

        # SMD
        smd = abs(mean_treat - mean_control) / (pooled_std + 1e-8)

        smd_list.append(smd)
        balanced_list.append('✓' if smd < threshold else '✗')

    feat_names = feature_names if feature_names else [f'X{i+1}' for i in range(n_features)]

    balance_df = pd.DataFrame({
        'Feature': feat_names,
        'SMD': smd_list,
        'Balanced': balanced_list
    })

    balance_df = balance_df.sort_values('SMD', ascending=False).reset_index(drop=True)

    return balance_df


def compute_propensity_score(
    X: np.ndarray,
    T: np.ndarray,
    method: str = 'logistic'
) -> np.ndarray:
    """
    计算倾向得分 P(T=1|X)

    Parameters:
    -----------
    X: 协变量矩阵
    T: 处理状态
    method: 方法
        - 'logistic': 逻辑回归
        - 'rf': 随机森林

    Returns:
    --------
    倾向得分数组

    Examples:
    ---------
    >>> from datasets import generate_linear_dgp
    >>> X, T, Y, _ = generate_linear_dgp(confounding=True)
    >>> ps = compute_propensity_score(X, T)
    >>> print(f"PS range: [{ps.min():.3f}, {ps.max():.3f}]")
    """
    if method == 'logistic':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)
        model.fit(X, T)
        propensity = model.predict_proba(X)[:, 1]

    elif method == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, T)
        propensity = model.predict_proba(X)[:, 1]

    else:
        raise ValueError(f"Unknown method: {method}")

    return propensity


# ==================== 可视化工具 ====================

def plot_dataset_overview(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    true_ite: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None
) -> go.Figure:
    """
    绘制数据集概览图

    包括:
    - 处理组/对照组分布
    - 结果分布
    - ITE 分布 (如果提供)
    - 协变量分布 (前 2 个特征)
    """
    if true_ite is not None:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Treatment Distribution',
                'Outcome Distribution',
                'ITE Distribution',
                'Covariates (X1 vs X2)'
            )
        )
    else:
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                'Treatment Distribution',
                'Outcome Distribution',
                'Covariates (X1 vs X2)'
            )
        )

    # 1. 处理分布
    treat_counts = [np.sum(T == 0), np.sum(T == 1)]
    fig.add_trace(go.Bar(
        x=['Control', 'Treatment'],
        y=treat_counts,
        marker_color=['#2D9CDB', '#27AE60'],
        text=treat_counts,
        textposition='auto'
    ), row=1, col=1)

    # 2. 结果分布 (按处理状态)
    fig.add_trace(go.Histogram(
        x=Y[T == 0],
        name='Control',
        marker_color='#2D9CDB',
        opacity=0.7,
        nbinsx=30
    ), row=1, col=2)

    fig.add_trace(go.Histogram(
        x=Y[T == 1],
        name='Treatment',
        marker_color='#27AE60',
        opacity=0.7,
        nbinsx=30
    ), row=1, col=2)

    if true_ite is not None:
        # 3. ITE 分布
        fig.add_trace(go.Histogram(
            x=true_ite,
            marker_color='#9B59B6',
            nbinsx=30,
            showlegend=False
        ), row=2, col=1)

        # 添加 ATE 线
        fig.add_vline(
            x=true_ite.mean(),
            line_dash="dash",
            line_color="red",
            row=2, col=1
        )

        # 4. 协变量散点图
        fig.add_trace(go.Scatter(
            x=X[T == 0, 0],
            y=X[T == 0, 1] if X.shape[1] > 1 else np.zeros(np.sum(T == 0)),
            mode='markers',
            name='Control',
            marker=dict(color='#2D9CDB', size=4, opacity=0.6),
            showlegend=False
        ), row=2, col=2)

        fig.add_trace(go.Scatter(
            x=X[T == 1, 0],
            y=X[T == 1, 1] if X.shape[1] > 1 else np.zeros(np.sum(T == 1)),
            mode='markers',
            name='Treatment',
            marker=dict(color='#27AE60', size=4, opacity=0.6),
            showlegend=False
        ), row=2, col=2)

    else:
        # 3. 协变量散点图
        fig.add_trace(go.Scatter(
            x=X[T == 0, 0],
            y=X[T == 0, 1] if X.shape[1] > 1 else np.zeros(np.sum(T == 0)),
            mode='markers',
            name='Control',
            marker=dict(color='#2D9CDB', size=4, opacity=0.6),
            showlegend=False
        ), row=1, col=3)

        fig.add_trace(go.Scatter(
            x=X[T == 1, 0],
            y=X[T == 1, 1] if X.shape[1] > 1 else np.zeros(np.sum(T == 1)),
            mode='markers',
            name='Treatment',
            marker=dict(color='#27AE60', size=4, opacity=0.6),
            showlegend=False
        ), row=1, col=3)

    # 更新布局
    fig.update_layout(
        title='Dataset Overview',
        height=600 if true_ite is not None else 400,
        template='plotly_white',
        showlegend=True,
        barmode='overlay'
    )

    # 更新轴标签
    x1_name = feature_names[0] if feature_names else 'X1'
    x2_name = feature_names[1] if feature_names and len(feature_names) > 1 else 'X2'

    if true_ite is not None:
        fig.update_xaxes(title_text=x1_name, row=2, col=2)
        fig.update_yaxes(title_text=x2_name, row=2, col=2)
    else:
        fig.update_xaxes(title_text=x1_name, row=1, col=3)
        fig.update_yaxes(title_text=x2_name, row=1, col=3)

    return fig


def plot_propensity_overlap(
    X: np.ndarray,
    T: np.ndarray
) -> go.Figure:
    """
    绘制倾向得分重叠图 (Propensity Score Overlap)

    用于检查共同支撑假设 (Common Support / Overlap)
    """
    # 计算倾向得分
    ps = compute_propensity_score(X, T)

    fig = go.Figure()

    # 处理组
    fig.add_trace(go.Histogram(
        x=ps[T == 1],
        name='Treatment',
        marker_color='#27AE60',
        opacity=0.6,
        nbinsx=30
    ))

    # 对照组
    fig.add_trace(go.Histogram(
        x=ps[T == 0],
        name='Control',
        marker_color='#2D9CDB',
        opacity=0.6,
        nbinsx=30
    ))

    fig.update_layout(
        title='Propensity Score Distribution',
        xaxis_title='Propensity Score P(T=1|X)',
        yaxis_title='Frequency',
        barmode='overlay',
        template='plotly_white'
    )

    return fig


if __name__ == "__main__":
    # 测试代码
    from .synthetic import generate_heterogeneous_dgp

    print("="*60)
    print("Testing Dataset Utils")
    print("="*60)

    # 生成测试数据
    X, T, Y, ite = generate_heterogeneous_dgp(
        n_samples=1000,
        heterogeneity_type='linear',
        seed=42
    )

    print("\n1. Train-Test Split")
    print("-"*60)
    X_tr, X_te, T_tr, T_te, Y_tr, Y_te, ite_tr, ite_te = train_test_split_causal(
        X, T, Y, ite, test_size=0.3, stratify_treatment=True
    )
    print(f"Train size: {len(T_tr)}")
    print(f"Test size: {len(T_te)}")
    print(f"Train treatment rate: {T_tr.mean():.2%}")
    print(f"Test treatment rate: {T_te.mean():.2%}")

    print("\n2. Dataset Description")
    print("-"*60)
    stats = describe_dataset(X, T, Y, ite)
    print(stats.to_string(index=False))

    print("\n3. Covariate Balance Check")
    print("-"*60)
    balance = check_covariate_balance(X, T, threshold=0.1)
    print(balance.head(10).to_string(index=False))

    imbalanced = balance[balance['SMD'] > 0.1]
    if len(imbalanced) > 0:
        print(f"\n⚠ Warning: {len(imbalanced)} features are imbalanced (SMD > 0.1)")
    else:
        print("\n✓ All features are balanced")

    print("\n4. Propensity Score")
    print("-"*60)
    ps = compute_propensity_score(X, T, method='logistic')
    print(f"Propensity score range: [{ps.min():.3f}, {ps.max():.3f}]")
    print(f"Mean PS (treated): {ps[T==1].mean():.3f}")
    print(f"Mean PS (control): {ps[T==0].mean():.3f}")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
