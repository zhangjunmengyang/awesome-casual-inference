"""
挑战基类 - Challenge Base Class
定义所有挑战的通用接口和结构
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


@dataclass
class ChallengeResult:
    """挑战评估结果"""

    # 基本信息
    challenge_name: str
    user_name: str
    submission_time: str

    # 核心指标
    primary_metric: float  # 主要评估指标
    secondary_metrics: Dict[str, float]  # 次要指标

    # 排名相关
    score: float  # 综合得分 (0-100)
    rank: Optional[int] = None

    # 详细信息
    method_description: str = ""
    execution_time: float = 0.0  # 秒

    def __repr__(self):
        return f"ChallengeResult(user={self.user_name}, score={self.score:.2f}, rank={self.rank})"


class Challenge(ABC):
    """
    挑战基类

    所有挑战都应该继承此类并实现必要的方法
    """

    def __init__(self, name: str, description: str, difficulty: str):
        """
        Parameters
        ----------
        name : str
            挑战名称
        description : str
            挑战描述
        difficulty : str
            难度级别: 'beginner', 'intermediate', 'advanced'
        """
        self.name = name
        self.description = description
        self.difficulty = difficulty

        # 数据集
        self.train_data = None
        self.test_data = None
        self.true_targets = None  # 真实目标 (仅用于评估)

    @abstractmethod
    def generate_data(self, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        生成训练和测试数据

        Returns
        -------
        train_data : pd.DataFrame
            训练数据
        test_data : pd.DataFrame
            测试数据 (不包含目标变量的真实值)
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        predictions: np.ndarray,
        user_name: str = "Anonymous"
    ) -> ChallengeResult:
        """
        评估用户的预测结果

        Parameters
        ----------
        predictions : np.ndarray
            用户的预测结果
        user_name : str
            用户名称

        Returns
        -------
        result : ChallengeResult
            评估结果
        """
        pass

    @abstractmethod
    def get_baseline_predictions(self, method: str = 'naive') -> np.ndarray:
        """
        获取基线方法的预测

        Parameters
        ----------
        method : str
            基线方法名称

        Returns
        -------
        predictions : np.ndarray
            基线预测
        """
        pass

    @abstractmethod
    def get_starter_code(self) -> str:
        """
        返回 starter code 模板

        Returns
        -------
        code : str
            Python 代码模板
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        获取挑战信息

        Returns
        -------
        info : dict
            包含挑战的详细信息
        """
        return {
            'name': self.name,
            'description': self.description,
            'difficulty': self.difficulty,
            'train_size': len(self.train_data) if self.train_data is not None else 0,
            'test_size': len(self.test_data) if self.test_data is not None else 0,
        }

    def reset_data(self, seed: int = 42):
        """重置数据集"""
        self.train_data, self.test_data = self.generate_data(seed)

    def validate_predictions(self, predictions: np.ndarray) -> Tuple[bool, str]:
        """
        验证预测格式是否正确

        Returns
        -------
        is_valid : bool
            预测是否有效
        message : str
            错误信息 (如果无效)
        """
        if self.test_data is None:
            return False, "Test data not loaded"

        expected_shape = (len(self.test_data),)

        if not isinstance(predictions, np.ndarray):
            return False, "Predictions must be a numpy array"

        if predictions.shape != expected_shape:
            return False, f"Expected shape {expected_shape}, got {predictions.shape}"

        if np.any(np.isnan(predictions)):
            return False, "Predictions contain NaN values"

        if np.any(np.isinf(predictions)):
            return False, "Predictions contain infinite values"

        return True, "Valid"

    def get_metric_description(self) -> Dict[str, str]:
        """
        返回评估指标的说明

        Returns
        -------
        descriptions : dict
            指标名称到说明的映射
        """
        return {}


class ChallengeDataGenerator:
    """
    数据生成器工具类

    提供各种因果推断场景的数据生成方法
    """

    @staticmethod
    def generate_lalonde_style_data(
        n: int,
        seed: int = 42,
        confounding_strength: float = 0.5
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
        """
        生成 LaLonde 风格的职业培训数据

        模拟真实的观察性研究数据，具有混淆偏差

        Returns
        -------
        df : pd.DataFrame
            数据集
        Y0 : np.ndarray
            潜在结果 Y(0)
        Y1 : np.ndarray
            潜在结果 Y(1)
        true_ate : float
            真实 ATE
        """
        np.random.seed(seed)

        # 协变量 (类似 LaLonde 数据)
        age = np.random.uniform(18, 60, n)
        education = np.random.randint(6, 18, n)
        re74 = np.maximum(0, np.random.normal(10000, 8000, n))  # 1974年收入
        re75 = np.maximum(0, np.random.normal(10000, 8000, n))  # 1975年收入
        black = np.random.binomial(1, 0.3, n)
        hispanic = np.random.binomial(1, 0.2, n)
        married = np.random.binomial(1, 0.4, n)

        # 综合倾向得分 (混淆: 低收入、低教育更可能参加培训)
        propensity_score = 1 / (1 + np.exp(
            -confounding_strength * (
                -0.0002 * re74
                - 0.0002 * re75
                - 0.1 * education
                + 0.3 * black
                + 0.2 * hispanic
                - 0.2 * married
            )
        ))

        # 随机分配处理 (基于倾向得分)
        T = np.random.binomial(1, propensity_score)

        # 潜在结果
        # Y(0): 没有培训的收入
        Y0 = (
            5000
            + 200 * education
            + 50 * age
            - 0.5 * age ** 2
            + 0.3 * re74
            + 0.3 * re75
            - 2000 * black
            - 1000 * hispanic
            + 1000 * married
            + np.random.normal(0, 3000, n)
        )

        # Y(1): 培训后的收入
        # 培训效应: 对低收入、低教育人群效果更好 (异质性)
        treatment_effect = (
            3000  # 基准效应
            - 100 * education  # 高学历效应较小
            + 500 * black  # 对少数族裔效应更好
            - 0.02 * re74  # 原本收入高的效应较小
        )

        Y1 = Y0 + treatment_effect

        # 观测结果
        Y = np.where(T == 1, Y1, Y0)

        # 真实 ATE
        true_ate = treatment_effect.mean()

        df = pd.DataFrame({
            'age': age,
            'education': education,
            're74': re74,
            're75': re75,
            'black': black,
            'hispanic': hispanic,
            'married': married,
            'T': T,
            'Y': Y
        })

        return df, Y0, Y1, true_ate

    @staticmethod
    def generate_ihdp_style_data(
        n: int,
        seed: int = 42,
        n_features: int = 10
    ) -> Tuple[pd.DataFrame, np.ndarray, float]:
        """
        生成 IHDP 风格的数据 (用于 CATE 预测)

        模拟婴儿健康发展干预项目数据，具有复杂的异质性效应

        Returns
        -------
        df : pd.DataFrame
            数据集
        true_cate : np.ndarray
            真实 CATE
        true_ate : float
            真实 ATE
        """
        np.random.seed(seed)

        # 生成协变量
        X = np.random.randn(n, n_features)

        # 随机分配处理 (略微不平衡)
        T = np.random.binomial(1, 0.6, n)

        # 基准结果 (受前几个特征影响)
        baseline = (
            10
            + 2 * X[:, 0]
            + 1.5 * X[:, 1]
            + 1 * X[:, 2]
            + 0.5 * X[:, 3]
        )

        # 复杂的异质性处理效应
        # 效应取决于多个特征的交互
        treatment_effect = (
            4  # 基准效应
            + 3 * X[:, 0]  # 线性异质性
            + 2 * X[:, 0] * X[:, 1]  # 二阶交互
            + 1.5 * np.sin(X[:, 2])  # 非线性
            + 1 * (X[:, 3] > 0).astype(float)  # 阈值效应
        )

        # 噪声
        noise = np.random.normal(0, 1, n)

        # 潜在结果
        Y0 = baseline + noise
        Y1 = baseline + treatment_effect + noise

        # 观测结果
        Y = np.where(T == 1, Y1, Y0)

        # 构建数据框
        df = pd.DataFrame(
            X,
            columns=[f'X{i+1}' for i in range(n_features)]
        )
        df['T'] = T
        df['Y'] = Y

        true_ate = treatment_effect.mean()

        return df, treatment_effect, true_ate

    @staticmethod
    def generate_marketing_data(
        n: int,
        seed: int = 42,
        n_features: int = 8
    ) -> Tuple[pd.DataFrame, np.ndarray, float]:
        """
        生成营销场景数据 (用于 Uplift Ranking)

        模拟优惠券发放场景，包含正面、零效应、负效应三类用户

        Returns
        -------
        df : pd.DataFrame
            数据集
        true_uplift : np.ndarray
            真实 uplift
        true_ate : float
            真实 ATE
        """
        np.random.seed(seed)

        # 生成协变量 (用户特征)
        X = np.random.randn(n, n_features)

        # 用户类型得分
        persuadable_score = X[:, 0] + 0.5 * X[:, 1]  # 可说服用户
        sure_thing_score = X[:, 2] + 0.5 * X[:, 3]  # 肯定购买用户
        lost_cause_score = X[:, 4] + 0.5 * X[:, 5]  # 不会购买用户
        sleeping_dog_score = X[:, 6] + 0.5 * X[:, 7]  # 负效应用户

        # 处理分配 (随机或略有偏向)
        T = np.random.binomial(1, 0.5, n)

        # 基准转化率
        base_conversion = 1 / (1 + np.exp(
            -0.3 * persuadable_score
            - 0.5 * sure_thing_score
            + 0.3 * lost_cause_score
            + 0.2 * sleeping_dog_score
        ))

        # Uplift 效应
        uplift = np.zeros(n)

        # Persuadables: 正效应
        uplift += 0.3 * (persuadable_score > 0).astype(float)

        # Sure things: 零效应 (本来就会买)
        uplift += 0.0 * (sure_thing_score > 1).astype(float)

        # Lost causes: 零效应 (怎么都不买)
        uplift += 0.0 * (lost_cause_score > 1).astype(float)

        # Sleeping dogs: 负效应 (干预反而减少购买)
        uplift -= 0.2 * (sleeping_dog_score > 0).astype(float)

        # 潜在结果
        Y0_prob = base_conversion
        Y1_prob = np.clip(base_conversion + uplift, 0, 1)

        # 观测结果 (二分类)
        Y0 = np.random.binomial(1, Y0_prob)
        Y1 = np.random.binomial(1, Y1_prob)
        Y = np.where(T == 1, Y1, Y0)

        # 真实 uplift 是概率差
        true_uplift = Y1_prob - Y0_prob
        true_ate = true_uplift.mean()

        # 构建数据框
        df = pd.DataFrame(
            X,
            columns=[f'feature_{i+1}' for i in range(n_features)]
        )
        df['T'] = T
        df['Y'] = Y

        return df, true_uplift, true_ate
