"""
Datasets Module Demo - 数据集模块演示

展示 datasets 模块的主要功能
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 导入数据集模块
from datasets import (
    load_lalonde,
    generate_ihdp_semi_synthetic,
    generate_linear_dgp,
    generate_nonlinear_dgp,
    generate_heterogeneous_dgp,
    train_test_split_causal,
    describe_dataset
)

from datasets.utils import (
    check_covariate_balance,
    compute_propensity_score,
    plot_dataset_overview,
    plot_propensity_overlap
)

from datasets.synthetic import generate_marketing_dgp


def demo_lalonde():
    """演示 LaLonde 数据集"""
    print("="*80)
    print("DEMO 1: LaLonde Dataset - 就业培训数据")
    print("="*80)

    # 加载三个版本
    nsw_df = load_lalonde('nsw')
    psid_df = load_lalonde('psid')
    cps_df = load_lalonde('cps')

    print("\n1.1 数据集规模对比")
    print("-"*80)
    print(f"NSW (RCT):             n = {len(nsw_df):,}")
    print(f"PSID (Observational):  n = {len(psid_df):,}")
    print(f"CPS (Observational):   n = {len(cps_df):,}")

    print("\n1.2 朴素 ATE 估计")
    print("-"*80)
    for name, df in [('NSW', nsw_df), ('PSID', psid_df), ('CPS', cps_df)]:
        ate = df[df['treat']==1]['re78'].mean() - df[df['treat']==0]['re78'].mean()
        print(f"{name:5s}: ${ate:,.2f}")

    print("\n💡 解读:")
    print("   - NSW (RCT) 的估计约为 $1,900 (真实因果效应)")
    print("   - PSID/CPS 的估计为负值，存在严重选择偏差!")
    print("   - 这正是 LaLonde (1986) 的经典发现")

    # 协变量平衡检查
    print("\n1.3 协变量平衡性检查")
    print("-"*80)
    feature_cols = ['age', 'education', 're74', 're75']

    for name, df in [('NSW', nsw_df), ('PSID', psid_df)]:
        balance = check_covariate_balance(
            df[feature_cols].values,
            df['treat'].values,
            feature_names=feature_cols,
            threshold=0.1
        )
        imbalanced = balance[balance['SMD'] > 0.1]
        print(f"\n{name}: {len(imbalanced)}/{len(feature_cols)} features imbalanced")
        if len(imbalanced) > 0:
            print(balance.head(3).to_string(index=False))


def demo_ihdp():
    """演示 IHDP 数据集"""
    print("\n" + "="*80)
    print("DEMO 2: IHDP Dataset - 婴儿健康发展计划")
    print("="*80)

    print("\n2.1 生成 IHDP 半合成数据 (设置 A)")
    print("-"*80)
    X, T, Y, true_ite = generate_ihdp_semi_synthetic(n_samples=747, setting='A', seed=42)

    print(f"样本量: {len(T)}")
    print(f"特征数: {X.shape[1]}")
    print(f"处理率: {T.mean():.2%}")
    print(f"\n真实 ATE: {true_ite.mean():.3f}")
    print(f"ITE 标准差: {true_ite.std():.3f}")
    print(f"ITE 范围: [{true_ite.min():.3f}, {true_ite.max():.3f}]")

    # 对比朴素估计
    naive_ate = Y[T==1].mean() - Y[T==0].mean()
    bias = abs(naive_ate - true_ite.mean())
    print(f"\n朴素 ATE: {naive_ate:.3f}")
    print(f"偏差: {bias:.3f}")

    print("\n2.2 设置 A vs 设置 B 对比")
    print("-"*80)
    for setting in ['A', 'B']:
        X_s, T_s, Y_s, ite_s = generate_ihdp_semi_synthetic(
            n_samples=747,
            setting=setting,
            seed=42
        )
        print(f"\n设置 {setting}:")
        print(f"  ATE: {ite_s.mean():.3f} ± {ite_s.std():.3f}")
        print(f"  异质性系数 (ITE_std/ATE): {ite_s.std()/ite_s.mean():.2f}")

    print("\n💡 用途: IHDP 是评估 CATE 方法的黄金标准")


def demo_synthetic():
    """演示合成数据生成器"""
    print("\n" + "="*80)
    print("DEMO 3: Synthetic Data Generators - 合成数据生成器")
    print("="*80)

    print("\n3.1 线性 DGP - 混淆效应演示")
    print("-"*80)

    results = []
    for confounding in [False, True]:
        X, T, Y, true_ite = generate_linear_dgp(
            n_samples=1000,
            confounding=confounding,
            treatment_effect=2.0,
            seed=42
        )

        naive_ate = Y[T==1].mean() - Y[T==0].mean()
        bias = abs(naive_ate - true_ite.mean())

        results.append({
            'Confounding': 'Yes' if confounding else 'No',
            'True ATE': f"{true_ite.mean():.3f}",
            'Naive ATE': f"{naive_ate:.3f}",
            'Bias': f"{bias:.3f}"
        })

    print(pd.DataFrame(results).to_string(index=False))
    print("\n💡 混淆导致朴素估计严重偏差!")

    print("\n3.2 非线性 DGP - 复杂度对比")
    print("-"*80)

    for complexity in ['low', 'medium', 'high']:
        X, T, Y, true_ite = generate_nonlinear_dgp(
            n_samples=1000,
            complexity=complexity,
            seed=42
        )

        print(f"\n{complexity.upper()}:")
        print(f"  ATE: {true_ite.mean():.3f}")
        print(f"  ITE std: {true_ite.std():.3f}")
        print(f"  ITE range: [{true_ite.min():.3f}, {true_ite.max():.3f}]")

    print("\n3.3 异质性 DGP - 不同异质性模式")
    print("-"*80)

    for het_type in ['linear', 'interaction', 'threshold', 'complex']:
        X, T, Y, true_ite = generate_heterogeneous_dgp(
            n_samples=1000,
            heterogeneity_type=het_type,
            seed=42
        )

        print(f"\n{het_type.upper()}:")
        print(f"  ATE: {true_ite.mean():.3f}")
        print(f"  ITE std: {true_ite.std():.3f}")
        print(f"  异质性系数: {true_ite.std()/abs(true_ite.mean()):.2f}")


def demo_marketing():
    """演示营销场景数据"""
    print("\n" + "="*80)
    print("DEMO 4: Marketing Scenarios - 营销场景数据")
    print("="*80)

    scenarios = ['coupon', 'email', 'recommendation']

    for scenario in scenarios:
        df, true_uplift = generate_marketing_dgp(
            n_samples=5000,
            scenario=scenario,
            seed=42
        )

        outcome_col = {
            'coupon': 'conversion',
            'email': 'click',
            'recommendation': 'purchase'
        }[scenario]

        treated_rate = df[df['treatment']==1][outcome_col].mean()
        control_rate = df[df['treatment']==0][outcome_col].mean()
        observed_uplift = treated_rate - control_rate

        print(f"\n{scenario.upper()} 场景:")
        print(f"  样本量: {len(df):,}")
        print(f"  处理组{outcome_col}率: {treated_rate:.2%}")
        print(f"  对照组{outcome_col}率: {control_rate:.2%}")
        print(f"  观测 Uplift: {observed_uplift:.4f}")
        print(f"  真实平均 Uplift: {true_uplift.mean():.4f}")


def demo_utils():
    """演示工具函数"""
    print("\n" + "="*80)
    print("DEMO 5: Utility Functions - 工具函数")
    print("="*80)

    # 生成测试数据
    X, T, Y, true_ite = generate_heterogeneous_dgp(
        n_samples=1000,
        heterogeneity_type='linear',
        seed=42
    )

    print("\n5.1 数据集描述")
    print("-"*80)
    stats = describe_dataset(X, T, Y, true_ite)
    print(stats.to_string(index=False))

    print("\n5.2 因果数据划分")
    print("-"*80)
    X_tr, X_te, T_tr, T_te, Y_tr, Y_te, ite_tr, ite_te = train_test_split_causal(
        X, T, Y, true_ite,
        test_size=0.3,
        stratify_treatment=True
    )

    print(f"训练集: {len(T_tr)} (处理率: {T_tr.mean():.2%})")
    print(f"测试集: {len(T_te)} (处理率: {T_te.mean():.2%})")
    print(f"处理率差异: {abs(T_tr.mean() - T_te.mean()):.4f}")

    print("\n5.3 协变量平衡检查")
    print("-"*80)
    balance = check_covariate_balance(X, T, threshold=0.1)
    print(balance.head(5).to_string(index=False))

    imbalanced = balance[balance['SMD'] > 0.1]
    if len(imbalanced) > 0:
        print(f"\n⚠ {len(imbalanced)} features are imbalanced")
    else:
        print("\n✓ All features are balanced")

    print("\n5.4 倾向得分")
    print("-"*80)
    ps = compute_propensity_score(X, T, method='logistic')
    print(f"倾向得分范围: [{ps.min():.3f}, {ps.max():.3f}]")
    print(f"处理组平均 PS: {ps[T==1].mean():.3f}")
    print(f"对照组平均 PS: {ps[T==0].mean():.3f}")


def demo_cate_evaluation():
    """演示 CATE 方法评估"""
    print("\n" + "="*80)
    print("DEMO 6: CATE Method Evaluation - CATE 方法评估示例")
    print("="*80)

    # 生成数据
    X, T, Y, true_ite = generate_ihdp_semi_synthetic(setting='A', seed=42)

    # 划分数据
    X_tr, X_te, T_tr, T_te, Y_tr, Y_te, ite_tr, ite_te = train_test_split_causal(
        X, T, Y, true_ite, test_size=0.3
    )

    print("\n6.1 训练 T-Learner")
    print("-"*80)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score

    # T-Learner
    model_0 = RandomForestRegressor(n_estimators=100, random_state=42)
    model_1 = RandomForestRegressor(n_estimators=100, random_state=42)

    model_0.fit(X_tr[T_tr==0], Y_tr[T_tr==0])
    model_1.fit(X_tr[T_tr==1], Y_tr[T_tr==1])

    # 预测 CATE
    pred_ite = model_1.predict(X_te) - model_0.predict(X_te)

    # 评估
    mse = mean_squared_error(ite_te, pred_ite)
    rmse = np.sqrt(mse)
    r2 = r2_score(ite_te, pred_ite)

    print(f"测试集样本: {len(ite_te)}")
    print(f"真实 ATE: {ite_te.mean():.3f}")
    print(f"预测 ATE: {pred_ite.mean():.3f}")
    print(f"\nCATE RMSE: {rmse:.3f}")
    print(f"CATE R²: {r2:.3f}")

    # 误差分析
    abs_error = np.abs(pred_ite - ite_te)
    print(f"\n绝对误差统计:")
    print(f"  Mean: {abs_error.mean():.3f}")
    print(f"  Median: {np.median(abs_error):.3f}")
    print(f"  90th percentile: {np.percentile(abs_error, 90):.3f}")


def demo_visualizations():
    """演示可视化功能"""
    print("\n" + "="*80)
    print("DEMO 7: Visualizations - 可视化示例")
    print("="*80)

    # 生成数据
    X, T, Y, true_ite = generate_heterogeneous_dgp(
        n_samples=1000,
        heterogeneity_type='threshold',
        seed=42
    )

    print("\n7.1 数据集概览图")
    print("-"*80)
    fig1 = plot_dataset_overview(X, T, Y, true_ite)
    print("✓ 图表已生成 (包含: 处理分布, 结果分布, ITE分布, 协变量散点)")

    print("\n7.2 倾向得分重叠图")
    print("-"*80)
    fig2 = plot_propensity_overlap(X, T)
    print("✓ 图表已生成 (检查共同支撑假设)")

    print("\n💡 在 Jupyter 中使用 fig.show() 查看交互式图表")


def main():
    """运行所有演示"""
    print("\n" + "="*80)
    print("🎯 DATASETS MODULE COMPREHENSIVE DEMO")
    print("="*80)
    print("\n本演示展示 datasets 模块的完整功能")
    print("包括: 经典数据集、合成数据、工具函数、评估示例\n")

    # 运行各个演示
    demo_lalonde()
    demo_ihdp()
    demo_synthetic()
    demo_marketing()
    demo_utils()
    demo_cate_evaluation()
    demo_visualizations()

    # 总结
    print("\n" + "="*80)
    print("✅ DEMO COMPLETE - 演示完成")
    print("="*80)
    print("\n主要功能:")
    print("  ✓ LaLonde: 观测数据 vs RCT 对比")
    print("  ✓ IHDP: CATE 评估基准")
    print("  ✓ Synthetic: 多种因果模型生成")
    print("  ✓ Marketing: 实际场景数据")
    print("  ✓ Utils: 完整工具链")
    print("  ✓ Visualization: 交互式可视化")

    print("\n下一步:")
    print("  1. 查看 datasets/README.md 了解详细文档")
    print("  2. 在 Jupyter 中运行示例代码")
    print("  3. 将数据集集成到因果推断模型中")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
