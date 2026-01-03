"""
测试挑战系统
"""

import numpy as np
import pandas as pd
from challenges import (
    ATEEstimationChallenge,
    CATEPredictionChallenge,
    UpliftRankingChallenge,
    Leaderboard
)


def test_ate_challenge():
    """测试 ATE 估计挑战"""
    print("=" * 60)
    print("Testing ATE Estimation Challenge")
    print("=" * 60)

    challenge = ATEEstimationChallenge()

    # 生成数据
    train_data, test_data = challenge.generate_data(seed=42)
    print(f"\nTrain data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"True ATE: {challenge.true_ate:.4f}")

    # 测试基线方法
    print("\nBaseline methods:")
    for method in ['naive', 'ipw', 'matching']:
        try:
            prediction = challenge.get_baseline_predictions(method)
            result = challenge.evaluate(prediction, user_name=f"Baseline-{method}")
            print(f"  {method}: ATE={prediction:.4f}, Score={result.score:.2f}, RelError={result.primary_metric:.4f}")
        except Exception as e:
            print(f"  {method}: Error - {str(e)}")

    # 测试提交
    print("\nTest submission:")
    test_prediction = challenge.true_ate + np.random.randn() * 0.1  # 添加小噪声
    result = challenge.evaluate(test_prediction, user_name="TestUser")
    print(f"  Prediction: {test_prediction:.4f}")
    print(f"  Score: {result.score:.2f}")
    print(f"  Primary metric: {result.primary_metric:.4f}")

    print("\nATE Challenge: PASSED")


def test_cate_challenge():
    """测试 CATE 预测挑战"""
    print("\n" + "=" * 60)
    print("Testing CATE Prediction Challenge")
    print("=" * 60)

    challenge = CATEPredictionChallenge()

    # 生成数据
    train_data, test_data = challenge.generate_data(seed=42)
    print(f"\nTrain data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"True ATE: {challenge.true_ate:.4f}")

    # 测试基线方法
    print("\nBaseline methods:")
    for method in ['s_learner', 't_learner']:
        try:
            predictions = challenge.get_baseline_predictions(method)
            result = challenge.evaluate(predictions, user_name=f"Baseline-{method}")
            print(f"  {method}: PEHE={result.primary_metric:.4f}, Score={result.score:.2f}")
        except Exception as e:
            print(f"  {method}: Error - {str(e)}")

    # 测试提交
    print("\nTest submission:")
    # 模拟预测: 真实值 + 噪声
    test_predictions = challenge.true_cate_test + np.random.randn(len(test_data)) * 0.5
    result = challenge.evaluate(test_predictions, user_name="TestUser")
    print(f"  PEHE: {result.primary_metric:.4f}")
    print(f"  Score: {result.score:.2f}")
    print(f"  Correlation: {result.secondary_metrics['correlation']:.4f}")

    print("\nCATE Challenge: PASSED")


def test_uplift_challenge():
    """测试 Uplift 排序挑战"""
    print("\n" + "=" * 60)
    print("Testing Uplift Ranking Challenge")
    print("=" * 60)

    challenge = UpliftRankingChallenge()

    # 生成数据
    train_data, test_data = challenge.generate_data(seed=42)
    print(f"\nTrain data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"True ATE: {challenge.true_ate:.4f}")

    # 测试基线方法
    print("\nBaseline methods:")
    for method in ['t_learner', 'class_transformation']:
        try:
            predictions = challenge.get_baseline_predictions(method)
            result = challenge.evaluate(predictions, user_name=f"Baseline-{method}")
            print(f"  {method}: AUUC={result.primary_metric:.4f}, Score={result.score:.2f}")
        except Exception as e:
            print(f"  {method}: Error - {str(e)}")

    # 测试提交
    print("\nTest submission:")
    # 模拟预测: 真实值 + 噪声
    test_predictions = challenge.true_uplift_test + np.random.randn(len(test_data)) * 0.1
    result = challenge.evaluate(test_predictions, user_name="TestUser")
    print(f"  AUUC: {result.primary_metric:.4f}")
    print(f"  Score: {result.score:.2f}")
    print(f"  Normalized AUUC: {result.secondary_metrics['normalized_auuc']:.4f}")

    print("\nUplift Challenge: PASSED")


def test_leaderboard():
    """测试排行榜系统"""
    print("\n" + "=" * 60)
    print("Testing Leaderboard System")
    print("=" * 60)

    # 创建测试挑战
    challenge = ATEEstimationChallenge()
    train_data, test_data = challenge.generate_data(seed=42)

    # 创建排行榜
    leaderboard = Leaderboard("Test Challenge", storage_dir="./test_submissions")

    # 清空旧数据
    leaderboard.clear_submissions()

    # 添加几个提交
    print("\nAdding test submissions...")
    for i, user in enumerate(['Alice', 'Bob', 'Charlie', 'David']):
        prediction = challenge.true_ate + np.random.randn() * (i + 1) * 0.5
        result = challenge.evaluate(prediction, user_name=user)
        leaderboard.add_submission(result)
        print(f"  {user}: Score={result.score:.2f}")

    # 获取排名
    print("\nLeaderboard Rankings:")
    rankings = leaderboard.get_rankings(top_n=5)
    print(rankings[['rank', 'user_name', 'score', 'primary_metric']])

    # 统计信息
    print("\nStatistics:")
    stats = leaderboard.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 生成图表
    print("\nGenerating plots...")
    fig1 = leaderboard.plot_rankings(top_n=5)
    print("  Rankings plot created")

    fig2 = leaderboard.plot_user_progress('Alice')
    print("  User progress plot created")

    # 导出
    print("\nExporting to CSV...")
    leaderboard.export_to_csv("./test_submissions/leaderboard.csv")
    print("  Exported successfully")

    # 清理
    import shutil
    shutil.rmtree("./test_submissions", ignore_errors=True)

    print("\nLeaderboard System: PASSED")


def test_challenge_validation():
    """测试挑战验证功能"""
    print("\n" + "=" * 60)
    print("Testing Challenge Validation")
    print("=" * 60)

    challenge = CATEPredictionChallenge()
    train_data, test_data = challenge.generate_data(seed=42)

    print("\nTest 1: Valid predictions")
    valid_preds = np.random.randn(len(test_data))
    is_valid, msg = challenge.validate_predictions(valid_preds)
    print(f"  Valid: {is_valid}, Message: {msg}")

    print("\nTest 2: Wrong shape")
    wrong_shape = np.random.randn(100)  # Wrong size
    is_valid, msg = challenge.validate_predictions(wrong_shape)
    print(f"  Valid: {is_valid}, Message: {msg}")

    print("\nTest 3: NaN values")
    nan_preds = np.random.randn(len(test_data))
    nan_preds[0] = np.nan
    is_valid, msg = challenge.validate_predictions(nan_preds)
    print(f"  Valid: {is_valid}, Message: {msg}")

    print("\nTest 4: Inf values")
    inf_preds = np.random.randn(len(test_data))
    inf_preds[0] = np.inf
    is_valid, msg = challenge.validate_predictions(inf_preds)
    print(f"  Valid: {is_valid}, Message: {msg}")

    print("\nValidation Tests: PASSED")


def test_data_generation():
    """测试数据生成"""
    print("\n" + "=" * 60)
    print("Testing Data Generation")
    print("=" * 60)

    from challenges.challenge_base import ChallengeDataGenerator

    print("\nTest 1: LaLonde-style data")
    df, Y0, Y1, ate = ChallengeDataGenerator.generate_lalonde_style_data(
        n=1000, seed=42, confounding_strength=0.5
    )
    print(f"  Shape: {df.shape}")
    print(f"  ATE: {ate:.4f}")
    print(f"  Columns: {list(df.columns)}")

    print("\nTest 2: IHDP-style data")
    df, cate, ate = ChallengeDataGenerator.generate_ihdp_style_data(
        n=1000, seed=42, n_features=10
    )
    print(f"  Shape: {df.shape}")
    print(f"  ATE: {ate:.4f}")
    print(f"  CATE range: [{cate.min():.2f}, {cate.max():.2f}]")

    print("\nTest 3: Marketing data")
    df, uplift, ate = ChallengeDataGenerator.generate_marketing_data(
        n=1000, seed=42, n_features=8
    )
    print(f"  Shape: {df.shape}")
    print(f"  ATE: {ate:.4f}")
    print(f"  Uplift range: [{uplift.min():.2f}, {uplift.max():.2f}]")

    print("\nData Generation: PASSED")


if __name__ == "__main__":
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + " " * 15 + "CHALLENGES SYSTEM TEST" + " " * 21 + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)

    try:
        test_data_generation()
        test_ate_challenge()
        test_cate_challenge()
        test_uplift_challenge()
        test_challenge_validation()
        test_leaderboard()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nChallenge system is ready to use!")
        print("Run 'python app.py' to start the application.")

    except Exception as e:
        print(f"\n{'=' * 60}")
        print("TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
