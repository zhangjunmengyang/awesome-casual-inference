"""
挑战系统演示
展示如何使用挑战系统
"""

import numpy as np
import pandas as pd
from challenges import (
    ATEEstimationChallenge,
    CATEPredictionChallenge,
    UpliftRankingChallenge,
    Leaderboard
)


def demo_ate_challenge():
    """演示 ATE 估计挑战"""
    print("\n" + "=" * 70)
    print("DEMO 1: ATE Estimation Challenge")
    print("=" * 70)

    # 创建挑战
    challenge = ATEEstimationChallenge()
    print("\n挑战信息:")
    print(f"  名称: {challenge.name}")
    print(f"  难度: {challenge.difficulty}")
    print(f"  描述: {challenge.description}")

    # 生成数据
    print("\n生成数据...")
    train_data, test_data = challenge.generate_data(seed=42)
    print(f"  训练集: {train_data.shape}")
    print(f"  测试集: {test_data.shape}")
    print(f"  真实 ATE: {challenge.true_ate:.2f}")

    # 数据预览
    print("\n训练数据预览:")
    print(train_data.head())

    # 尝试基线方法
    print("\n基线方法性能:")
    methods = ['naive', 'ipw', 'matching']
    for method in methods:
        pred = challenge.get_baseline_predictions(method)
        result = challenge.evaluate(pred, user_name=f"Baseline-{method}")
        print(f"  {method:12s}: ATE={pred:8.2f}, Score={result.score:6.2f}, Error={result.primary_metric:.4f}")

    # 提交一个解决方案
    print("\n提交你的方案:")
    print("  方法: 双重稳健 (Doubly Robust)")

    # 简单的双重稳健估计
    from sklearn.linear_model import LogisticRegression, Ridge

    X_cols = ['age', 'education', 're74', 're75', 'black', 'hispanic', 'married']
    X = train_data[X_cols].values
    T = train_data['T'].values
    Y = train_data['Y'].values

    # 估计倾向得分
    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(X, T)
    ps = ps_model.predict_proba(X)[:, 1]
    ps = np.clip(ps, 0.01, 0.99)

    # 估计结果模型
    outcome_model = Ridge(alpha=1.0)
    X_with_T = np.column_stack([X, T])
    outcome_model.fit(X_with_T, Y)

    # 双重稳健估计
    X_1 = np.column_stack([X, np.ones(len(X))])
    X_0 = np.column_stack([X, np.zeros(len(X))])
    mu_1 = outcome_model.predict(X_1)
    mu_0 = outcome_model.predict(X_0)

    dr_ate = np.mean(
        T * (Y - mu_1) / ps + mu_1 -
        (1 - T) * (Y - mu_0) / (1 - ps) - mu_0
    )

    result = challenge.evaluate(dr_ate, user_name="Demo-DR")
    print(f"  估计 ATE: {dr_ate:.2f}")
    print(f"  得分: {result.score:.2f}")
    print(f"  相对误差: {result.primary_metric:.4f}")


def demo_cate_challenge():
    """演示 CATE 预测挑战"""
    print("\n" + "=" * 70)
    print("DEMO 2: CATE Prediction Challenge")
    print("=" * 70)

    # 创建挑战
    challenge = CATEPredictionChallenge()
    print("\n挑战信息:")
    print(f"  名称: {challenge.name}")
    print(f"  难度: {challenge.difficulty}")

    # 生成数据
    print("\n生成数据...")
    train_data, test_data = challenge.generate_data(seed=42)
    print(f"  训练集: {train_data.shape}")
    print(f"  测试集: {test_data.shape}")
    print(f"  真实 ATE: {challenge.true_ate:.4f}")

    # 尝试 X-Learner
    print("\n使用 X-Learner...")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression

    X_cols = [f'X{i}' for i in range(1, 11)]
    X_train = train_data[X_cols].values
    T_train = train_data['T'].values
    Y_train = train_data['Y'].values
    X_test = test_data[X_cols].values

    # Stage 1: 结果模型
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

    # 预测
    tau_0_pred = tau_0_model.predict(X_test)
    tau_1_pred = tau_1_model.predict(X_test)
    cate_pred = g_test * tau_0_pred + (1 - g_test) * tau_1_pred

    # 评估
    result = challenge.evaluate(cate_pred, user_name="Demo-XLearner")
    print(f"  PEHE: {result.primary_metric:.4f}")
    print(f"  得分: {result.score:.2f}")
    print(f"  相关系数: {result.secondary_metrics['correlation']:.4f}")
    print(f"  R²: {result.secondary_metrics['r2']:.4f}")

    # CATE 分布
    print(f"\n  CATE 统计:")
    print(f"    均值: {cate_pred.mean():.4f}")
    print(f"    标准差: {cate_pred.std():.4f}")
    print(f"    最小值: {cate_pred.min():.4f}")
    print(f"    最大值: {cate_pred.max():.4f}")


def demo_uplift_challenge():
    """演示 Uplift 排序挑战"""
    print("\n" + "=" * 70)
    print("DEMO 3: Uplift Ranking Challenge")
    print("=" * 70)

    # 创建挑战
    challenge = UpliftRankingChallenge()
    print("\n挑战信息:")
    print(f"  名称: {challenge.name}")
    print(f"  难度: {challenge.difficulty}")

    # 生成数据
    print("\n生成数据...")
    train_data, test_data = challenge.generate_data(seed=42)
    print(f"  训练集: {train_data.shape}")
    print(f"  测试集: {test_data.shape}")

    # 使用 T-Learner
    print("\n使用 T-Learner (分类版本)...")
    from sklearn.ensemble import RandomForestClassifier

    X_cols = [f'feature_{i}' for i in range(1, 9)]
    X_train = train_data[X_cols].values
    T_train = train_data['T'].values
    Y_train = train_data['Y'].values
    X_test = test_data[X_cols].values

    model_0 = RandomForestClassifier(n_estimators=100, random_state=42)
    model_1 = RandomForestClassifier(n_estimators=100, random_state=43)

    model_0.fit(X_train[T_train == 0], Y_train[T_train == 0])
    model_1.fit(X_train[T_train == 1], Y_train[T_train == 1])

    p0 = model_0.predict_proba(X_test)[:, 1]
    p1 = model_1.predict_proba(X_test)[:, 1]

    uplift_pred = p1 - p0

    # 评估
    result = challenge.evaluate(uplift_pred, user_name="Demo-TLearner")
    print(f"  AUUC: {result.primary_metric:.4f}")
    print(f"  得分: {result.score:.2f}")
    print(f"  归一化 AUUC: {result.secondary_metrics['normalized_auuc']:.4f}")
    print(f"  Kendall Tau: {result.secondary_metrics['kendall_tau']:.4f}")

    # 业务分析
    print("\n  业务指标:")
    for key in ['roi_30%', 'top_10%', 'top_20%', 'top_30%']:
        if key in result.secondary_metrics:
            print(f"    {key}: {result.secondary_metrics[key]:.4f}")

    # 最优干预比例
    print("\n  最优干预策略分析:")
    opt_info = challenge.calculate_optimal_targeting_fraction()
    print(f"    最优干预比例: {opt_info['optimal_fraction']*100:.1f}%")
    print(f"    最优 ROI: {opt_info['optimal_roi']:.2f}")


def demo_leaderboard():
    """演示排行榜系统"""
    print("\n" + "=" * 70)
    print("DEMO 4: Leaderboard System")
    print("=" * 70)

    # 创建挑战
    challenge = CATEPredictionChallenge()
    train_data, test_data = challenge.generate_data(seed=42)

    # 创建排行榜
    lb = Leaderboard("Demo CATE Challenge", storage_dir="./demo_submissions")

    # 清空旧数据
    lb.clear_submissions()

    # 模拟多个用户提交
    print("\n模拟用户提交...")
    users = [
        ("Alice", 0.8, "T-Learner + Feature Engineering"),
        ("Bob", 1.2, "S-Learner"),
        ("Charlie", 0.6, "X-Learner + Ensemble"),
        ("David", 1.5, "Baseline T-Learner"),
        ("Eve", 0.7, "Causal Forest")
    ]

    for user, noise_level, method in users:
        # 生成预测 (真实值 + 噪声)
        pred = challenge.true_cate_test + np.random.randn(len(test_data)) * noise_level
        result = challenge.evaluate(pred, user_name=user)
        result.method_description = method

        lb.add_submission(result)
        print(f"  {user:12s}: PEHE={result.primary_metric:.4f}, Score={result.score:.2f}")

    # 显示排名
    print("\n排行榜 Top 5:")
    rankings = lb.get_rankings(top_n=5)
    print(rankings[['rank', 'user_name', 'score', 'primary_metric', 'method_description']])

    # 统计信息
    print("\n统计信息:")
    stats = lb.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 清理
    import shutil
    shutil.rmtree("./demo_submissions", ignore_errors=True)

    print("\n排行榜演示完成!")


if __name__ == "__main__":
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + " " * 20 + "挑战系统演示" + " " * 34 + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)

    demo_ate_challenge()
    demo_cate_challenge()
    demo_uplift_challenge()
    demo_leaderboard()

    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70)
    print("\n接下来:")
    print("  1. 运行 'python app.py' 启动完整应用")
    print("  2. 访问 http://localhost:7860")
    print("  3. 进入 'Challenges' 标签页开始挑战!")
    print("\n祝你好运!")
