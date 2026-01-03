"""
LaLonde 就业培训数据集

经典的因果推断数据集，来自 Robert LaLonde 的就业培训项目研究。
用于评估国家支持工作示范 (NSW) 项目的效果。

数据来源:
---------
LaLonde, R. (1986). "Evaluating the Econometric Evaluations of Training Programs
with Experimental Data". American Economic Review.

变量说明:
---------
- age: 年龄
- education: 受教育年限
- black: 是否黑人 (1=是, 0=否)
- hispanic: 是否西班牙裔 (1=是, 0=否)
- married: 是否已婚 (1=是, 0=否)
- nodegree: 是否没有学位 (1=是, 0=否)
- re74: 1974年实际收入 (美元)
- re75: 1975年实际收入 (美元)
- re78: 1978年实际收入 (美元) [结果变量]
- treat: 是否接受培训 (1=是, 0=否)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def load_lalonde(
    version: str = 'nsw',
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    加载 LaLonde 数据集

    Parameters:
    -----------
    version: 数据版本
        - 'nsw': NSW 实验数据 (RCT, n=722)
        - 'psid': PSID 观测数据 (n=2490)
        - 'cps': CPS 观测数据 (n=15992)
    seed: 随机种子 (用于模拟数据)

    Returns:
    --------
    DataFrame with LaLonde data

    Examples:
    ---------
    >>> df = load_lalonde()
    >>> print(df.columns)
    >>> ate = df[df['treat']==1]['re78'].mean() - df[df['treat']==0]['re78'].mean()
    """
    if version == 'nsw':
        return _generate_nsw_data(seed)
    elif version == 'psid':
        return _generate_psid_data(seed)
    elif version == 'cps':
        return _generate_cps_data(seed)
    else:
        raise ValueError(f"Unknown version: {version}. Choose from 'nsw', 'psid', 'cps'")


def _generate_nsw_data(seed: Optional[int] = 42) -> pd.DataFrame:
    """
    生成 NSW 实验数据 (基于真实数据统计特征模拟)

    NSW 是随机对照试验 (RCT)，处理组和对照组通过随机分配
    """
    if seed is not None:
        np.random.seed(seed)

    # 样本量 (接近真实数据)
    n_treat = 185
    n_control = 260
    n_total = n_treat + n_control

    # 生成协变量 (基于真实数据分布)
    def generate_covariates(n):
        age = np.random.gamma(2.5, 8, n) + 18
        age = np.clip(age, 18, 55)

        education = np.random.choice(
            [8, 9, 10, 11, 12, 13, 14, 16],
            size=n,
            p=[0.05, 0.10, 0.15, 0.20, 0.25, 0.10, 0.10, 0.05]
        )

        black = np.random.binomial(1, 0.84, n)
        hispanic = np.random.binomial(1, 0.06, n)
        married = np.random.binomial(1, 0.19, n)
        nodegree = np.random.binomial(1, 0.71, n)

        # 1974 年收入 (许多人为0)
        re74_nonzero = np.random.gamma(2, 1500, n)
        re74 = np.where(np.random.rand(n) < 0.25, 0, re74_nonzero)

        # 1975 年收入
        re75_nonzero = np.random.gamma(2, 1500, n)
        re75 = np.where(np.random.rand(n) < 0.25, 0, re75_nonzero)

        return {
            'age': age,
            'education': education,
            'black': black,
            'hispanic': hispanic,
            'married': married,
            'nodegree': nodegree,
            're74': re74,
            're75': re75
        }

    # 生成处理组
    treat_data = generate_covariates(n_treat)
    treat_data['treat'] = np.ones(n_treat)

    # 生成对照组
    control_data = generate_covariates(n_control)
    control_data['treat'] = np.zeros(n_control)

    # 合并
    data = {k: np.concatenate([treat_data[k], control_data[k]]) for k in treat_data.keys()}

    # 生成 1978 年收入 (结果变量)
    # 基线收入 (基于历史收入和协变量)
    baseline = (
        2000 +
        100 * data['age'] +
        200 * data['education'] +
        0.3 * data['re74'] +
        0.3 * data['re75'] +
        500 * data['married'] -
        500 * data['nodegree']
    )

    # 处理效应 (约 $1794，真实数据的估计)
    treatment_effect = 1794

    # 添加噪声
    noise = np.random.randn(n_total) * 3000

    data['re78'] = baseline + treatment_effect * data['treat'] + noise
    data['re78'] = np.maximum(data['re78'], 0)  # 收入非负

    # 转换为 DataFrame
    df = pd.DataFrame(data)

    # 调整列顺序
    columns = ['age', 'education', 'black', 'hispanic', 'married',
               'nodegree', 're74', 're75', 'treat', 're78']
    return df[columns].reset_index(drop=True)


def _generate_psid_data(seed: Optional[int] = 42) -> pd.DataFrame:
    """
    生成 PSID 观测数据

    PSID (Panel Study of Income Dynamics) 是观测数据，
    控制组来自 PSID 调查，与 NSW 处理组有显著差异，存在选择偏差
    """
    if seed is not None:
        np.random.seed(seed)

    # NSW 处理组
    nsw_df = _generate_nsw_data(seed)
    treat_df = nsw_df[nsw_df['treat'] == 1].copy()

    # PSID 控制组 (n=2490)
    n_control = 2490

    # PSID 人群特征更好 (更高收入、更高教育)
    age = np.random.gamma(3, 10, n_control) + 20
    age = np.clip(age, 20, 60)

    education = np.random.choice(
        [10, 11, 12, 13, 14, 16, 18],
        size=n_control,
        p=[0.05, 0.08, 0.20, 0.15, 0.20, 0.22, 0.10]
    )

    black = np.random.binomial(1, 0.25, n_control)  # 更低比例
    hispanic = np.random.binomial(1, 0.07, n_control)
    married = np.random.binomial(1, 0.87, n_control)  # 更高比例
    nodegree = np.random.binomial(1, 0.30, n_control)  # 更低比例

    # 更高的历史收入
    re74 = np.random.gamma(3, 5000, n_control)
    re75 = np.random.gamma(3, 5000, n_control)

    # 1978 年收入
    baseline = (
        3000 +
        150 * age +
        300 * education +
        0.4 * re74 +
        0.4 * re75 +
        1000 * married -
        1000 * nodegree
    )
    noise = np.random.randn(n_control) * 4000
    re78 = baseline + noise
    re78 = np.maximum(re78, 0)

    control_df = pd.DataFrame({
        'age': age,
        'education': education,
        'black': black,
        'hispanic': hispanic,
        'married': married,
        'nodegree': nodegree,
        're74': re74,
        're75': re75,
        'treat': np.zeros(n_control),
        're78': re78
    })

    # 合并
    df = pd.concat([treat_df, control_df], ignore_index=True)

    columns = ['age', 'education', 'black', 'hispanic', 'married',
               'nodegree', 're74', 're75', 'treat', 're78']
    return df[columns].reset_index(drop=True)


def _generate_cps_data(seed: Optional[int] = 42) -> pd.DataFrame:
    """
    生成 CPS 观测数据

    CPS (Current Population Survey) 是大规模观测数据，
    与 NSW 处理组差异更大
    """
    if seed is not None:
        np.random.seed(seed)

    # NSW 处理组
    nsw_df = _generate_nsw_data(seed)
    treat_df = nsw_df[nsw_df['treat'] == 1].copy()

    # CPS 控制组 (n=15992)
    n_control = 15992

    # 更加多样化的人群
    age = np.random.gamma(3.5, 10, n_control) + 18
    age = np.clip(age, 18, 65)

    education = np.random.choice(
        [8, 10, 12, 13, 14, 16, 18],
        size=n_control,
        p=[0.03, 0.07, 0.25, 0.15, 0.20, 0.22, 0.08]
    )

    black = np.random.binomial(1, 0.07, n_control)
    hispanic = np.random.binomial(1, 0.07, n_control)
    married = np.random.binomial(1, 0.85, n_control)
    nodegree = np.random.binomial(1, 0.25, n_control)

    # 历史收入
    re74 = np.random.gamma(3, 6000, n_control)
    re75 = np.random.gamma(3, 6000, n_control)

    # 1978 年收入
    baseline = (
        4000 +
        150 * age +
        400 * education +
        0.35 * re74 +
        0.35 * re75 +
        1500 * married -
        1500 * nodegree
    )
    noise = np.random.randn(n_control) * 5000
    re78 = baseline + noise
    re78 = np.maximum(re78, 0)

    control_df = pd.DataFrame({
        'age': age,
        'education': education,
        'black': black,
        'hispanic': hispanic,
        'married': married,
        'nodegree': nodegree,
        're74': re74,
        're75': re75,
        'treat': np.zeros(n_control),
        're78': re78
    })

    # 合并
    df = pd.concat([treat_df, control_df], ignore_index=True)

    columns = ['age', 'education', 'black', 'hispanic', 'married',
               'nodegree', 're74', 're75', 'treat', 're78']
    return df[columns].reset_index(drop=True)


def get_lalonde_statistics(df: pd.DataFrame) -> dict:
    """
    计算 LaLonde 数据集的统计摘要

    Parameters:
    -----------
    df: LaLonde DataFrame

    Returns:
    --------
    Dictionary with statistics
    """
    treat = df[df['treat'] == 1]
    control = df[df['treat'] == 0]

    stats = {
        'n_total': len(df),
        'n_treat': len(treat),
        'n_control': len(control),
        'ate_naive': treat['re78'].mean() - control['re78'].mean(),
        'treat_mean': {
            'age': treat['age'].mean(),
            'education': treat['education'].mean(),
            're74': treat['re74'].mean(),
            're75': treat['re75'].mean(),
            're78': treat['re78'].mean(),
        },
        'control_mean': {
            'age': control['age'].mean(),
            'education': control['education'].mean(),
            're74': control['re74'].mean(),
            're75': control['re75'].mean(),
            're78': control['re78'].mean(),
        }
    }

    return stats


if __name__ == "__main__":
    # 测试代码
    print("Loading LaLonde NSW data...")
    nsw_df = load_lalonde('nsw')
    print(f"\nNSW Data Shape: {nsw_df.shape}")
    print("\nFirst 5 rows:")
    print(nsw_df.head())

    print("\n" + "="*60)
    stats = get_lalonde_statistics(nsw_df)
    print(f"\nSample Size: {stats['n_total']}")
    print(f"  Treatment: {stats['n_treat']}")
    print(f"  Control: {stats['n_control']}")
    print(f"\nNaive ATE: ${stats['ate_naive']:.2f}")

    print("\n" + "="*60)
    print("\nLoading PSID comparison data...")
    psid_df = load_lalonde('psid')
    print(f"PSID Data Shape: {psid_df.shape}")

    stats_psid = get_lalonde_statistics(psid_df)
    print(f"\nNaive ATE (with PSID control): ${stats_psid['ate_naive']:.2f}")
    print("\nCovariate Balance (Treatment vs Control):")
    print(f"  Age: {stats_psid['treat_mean']['age']:.1f} vs {stats_psid['control_mean']['age']:.1f}")
    print(f"  Education: {stats_psid['treat_mean']['education']:.1f} vs {stats_psid['control_mean']['education']:.1f}")
    print(f"  RE74: ${stats_psid['treat_mean']['re74']:.0f} vs ${stats_psid['control_mean']['re74']:.0f}")
    print(f"  RE75: ${stats_psid['treat_mean']['re75']:.0f} vs ${stats_psid['control_mean']['re75']:.0f}")

    print("\n" + "="*60)
    print("\nLoading CPS comparison data...")
    cps_df = load_lalonde('cps')
    print(f"CPS Data Shape: {cps_df.shape}")

    stats_cps = get_lalonde_statistics(cps_df)
    print(f"\nNaive ATE (with CPS control): ${stats_cps['ate_naive']:.2f}")
