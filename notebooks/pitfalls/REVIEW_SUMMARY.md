# Pitfalls Notebooks - Comprehensive Review & Fix Summary

**Reviewer**: Senior Data Scientist & Causal Inference Expert
**Review Date**: 2026-01-04
**Status**: Complete Review, Detailed Fix Recommendations

---

## Executive Summary

All 5 pitfall notebooks have been thoroughly reviewed. The notebooks demonstrate strong pedagogical structure and cover critical real-world failure modes. However, several issues need addressing to make them interview-ready and production-quality:

### Overall Issues Across All Notebooks:
1. ✅ **TODOs without complete reference answers** - Need full implementations
2. ✅ **思考题 (Thinking Questions) without answer keys** - Critical for self-study
3. ✅ **Missing interview simulation sections** - These are "interview送分题"!
4. ⚠️ **Pitfall 05 has JSON corruption** - Needs immediate fix

### Quality Rating:
- **Pitfall 01 (PSM)**: 85% complete - Best structured
- **Pitfall 02 (CUPED)**: 80% complete - Good, needs TODO completion
- **Pitfall 03 (DID)**: 75% complete - Needs more diagnostic details
- **Pitfall 04 (IV)**: 70% complete - Multiple TODOs need implementation
- **Pitfall 05 (AB Test)**: 50% complete - Most work needed + JSON corruption

---

## Pitfall 01: PSM Failure Modes ✅ (Best One)

### Strengths:
- ✅ Excellent structure with 4 clear failure modes
- ✅ Love Plot visualization is perfect
- ✅ Complete SMD calculation and interpretation
- ✅ PSM diagnostic pipeline is comprehensive
- ✅ Code executes without errors

### Issues to Fix:

#### 1. **Add Answer Keys for 思考题** (Cells 27-30)

**问题 1: SMD 的阈值 0.1 是怎么来的?**

**答案**:
```markdown
SMD (Standardized Mean Difference) 的 0.1 阈值来自经验规则 (Cohen's d):
- 0.1 = 小效应量 (small effect size)
- 0.2-0.5 = 中等效应量
- > 0.8 = 大效应量

Austin (2009) 在 PSM 的文献中建议:
- |SMD| < 0.1: Balance 良好，协变量分布基本一致
- |SMD| < 0.25: 可接受的 Balance
- |SMD| > 0.25: Balance 较差，需要调整匹配策略

更好的判断标准:
1. 结合 p 值（但不要只看 p 值）
2. 使用 Love Plot 可视化
3. 检查多个 SMD 一起看（不要只看平均）
4. 考虑业务含义（哪些协变量更重要）
```

**问题 2: Balance 检查不通过的解决方案?**

**答案**:
```markdown
5 种解决方案（优先级从高到低）:

1. **改进倾向得分模型**
   - 添加交互项和非线性项
   - 尝试不同的机器学习方法（Random Forest, GBM）

2. **调整匹配参数**
   - 改变 caliper 大小
   - 使用 1:k 匹配（多对一）
   - 尝试不同的匹配算法（optimal matching, genetic matching）

3. **Trimming（修剪）**
   - 去掉倾向得分过于极端的样本
   - 只保留共同支撑区域内的样本

4. **分层匹配**
   - 先按重要协变量分层，再在层内匹配

5. **换方法**
   - 如果 PSM 始终 Balance 不好，考虑:
     * IPW (Inverse Probability Weighting)
     * Doubly Robust 方法
     * Covariate Adjustment
```

**问题 3: PSM 丢失样本的影响?**

**答案**:
```markdown
样本丢失的 3 大影响:

1. **估计目标改变** (Most Important!)
   - 原本估计 ATT (Average Treatment Effect on the Treated)
   - 丢失样本后变成 ATT 在匹配成功子集上的效应
   - 外推性 (external validity) 下降

2. **统计功效下降**
   - 样本量减少 → 标准误增大 → 更难检测到效应
   - 如果丢失 > 30%，可能需要重新做 power analysis

3. **选择偏差风险**
   - 检查哪些样本被丢失了
   - 如果丢失的是极端值样本，可能导致:
     * 效应估计偏向"普通"用户
     * 无法回答原始研究问题

面试加分点: 提到 ATOS (Average Treatment Effect on Overlap Sample)
```

**问题 4: PSM 的局限性 (面试必考!)**

**答案**:
```markdown
面试标准答案 (分 3 个层次):

【基础回答】
PSM 只能控制观测到的混淆变量，无法处理未观测混淆。

【进阶回答】
PSM 的 4 个核心假设:
1. Unconfoundedness: 给定 X，(Y0, Y1) ⊥ T
2. Common Support: 0 < P(T=1|X) < 1
3. SUTVA: 无干扰假设
4. Correct specification: 倾向得分模型正确

任何一个假设违背，估计都有偏。

【高级回答】（面试加分）
与其他方法对比:
- vs RCT: PSM 无法保证 unconfoundedness
- vs DiD: PSM 需要 selection on observables，DiD 允许 time-invariant unobservables
- vs IV: PSM 无法处理双向因果，IV 可以
- vs RDD: PSM 对函数形式敏感，RDD 有局部识别

建议: 敏感性分析 (Rosenbaum bounds, E-value)
```

#### 2. **Add Interview Simulation Section**

在总结部分之前添加新的 section:

```markdown
---

## 🎤 面试模拟环节

### 场景 1: PSM 分析被质疑

**面试官**: "你做了 PSM 分析，但我怎么相信你的结果是对的？Balance 检查就够了吗？"

**你的回答（参考）**:
```
Balance 检查是必要的但不是充分的。我会从 3 个层面验证:

1. **诊断检查** (Diagnostics):
   - SMD < 0.1 for all covariates
   - Love Plot visualization
   - Density plot of propensity scores
   - 匹配率 > 80%

2. **敏感性分析** (Robustness):
   - 改变 caliper，看结果稳定性
   - 尝试不同匹配算法（NN, Optimal, Genetic）
   - 与其他方法比较（IPW, DR）

3. **未观测混淆评估** (Unobserved Confounding):
   - Rosenbaum bounds: 计算需要多强的隐藏偏差才能推翻结论
   - E-value: 未观测混淆需要多大才能解释掉效应
   - 负对照分析: 在不应有效应的结果上检验
```

### 场景 2: 快速判断

**面试官**: "给你 30 秒，快速判断一个 PSM 分析是否可信，你看什么？"

**你的回答（参考）**:
```
我会看这 5 个指标（按优先级）:

1. **Sample Ratio**: 匹配后处理组/对照组比例（应该接近 1:1）
2. **Max SMD**: 最大的 SMD 是多少（< 0.1）
3. **Match Rate**: 有多少样本匹配成功（> 80%）
4. **Common Support**: 倾向得分分布图是否有重叠
5. **First Stage F**: 如果用工具变量辅助，F > 10

如果这 5 个都 pass，基本可以相信。
```

### 场景 3: 补救措施

**面试官**: "你的 PSM Balance 很差，但 deadline 明天，怎么办？"

**你的回答（参考）**:
```
紧急情况下的 3 个选择:

1. **快速改进** (优先):
   - 用机器学习估计倾向得分（XGBoost, Random Forest）
   - 添加协变量的平方项和交互项
   - 调整 caliper（试试 0.1σ 到 0.5σ）

2. **换方法** (备选):
   - IPW: 不需要匹配，直接加权
   - Doubly Robust: 结合 OR 和 PS，更稳健
   - Regression Adjustment: 最快，作为 baseline

3. **诚实汇报** (必须):
   - 在报告中说明 Balance 不理想
   - 提供敏感性分析
   - 给出置信区间而不是点估计
   - 建议后续改进方向

Never: 不要隐瞒 Balance 问题!
```
```

---

## Pitfall 02: CUPED Misuse 🔧

### Strengths:
- ✅ 4 种失败模式覆盖全面
- ✅ 低相关性、新用户、处理影响协变量、小样本都讲到了
- ✅ 分层 CUPED 是正确的做法
- ✅ 前置检查框架很好

### Issues to Fix:

#### 1. **Complete TODO in Cell 22** - cuped_preflight_check

当前代码只有框架，需要完整实现。这是核心函数！

**完整实现**:
```python
def cuped_preflight_check(Y_control, X_control, Y_treatment, X_treatment,
                          min_correlation=0.3, min_sample_size=200, alpha=0.05):
    """
    CUPED 前置检查 - 完整实现
    """
    checks = []
    passed = True

    # 检查 1: 样本量
    n_c, n_t = len(Y_control), len(Y_treatment)
    sample_size_ok = n_c >= min_sample_size and n_t >= min_sample_size
    checks.append({
        'name': '样本量检查',
        'passed': sample_size_ok,
        'message': f"控制组: {n_c}, 实验组: {n_t} (最低要求: {min_sample_size})",
        'severity': 'error' if not sample_size_ok else 'ok'
    })
    if not sample_size_ok:
        passed = False

    # 检查 2: 相关性
    X_valid_c = X_control[~np.isnan(X_control)]
    Y_valid_c = Y_control[~np.isnan(X_control)]
    X_valid_t = X_treatment[~np.isnan(X_treatment)]
    Y_valid_t = Y_treatment[~np.isnan(X_treatment)]

    if len(X_valid_c) > 2 and len(X_valid_t) > 2:
        X_all = np.concatenate([X_valid_c, X_valid_t])
        Y_all = np.concatenate([Y_valid_c, Y_valid_t])
        rho, p_val = stats.pearsonr(X_all, Y_all)

        corr_ok = abs(rho) >= min_correlation
        checks.append({
            'name': '相关性检查',
            'passed': corr_ok,
            'message': f"ρ = {rho:.3f} (最低要求: {min_correlation}), 理论方差缩减: {rho**2:.1%}",
            'severity': 'warning' if not corr_ok else 'ok'
        })

        if not corr_ok:
            passed = False
    else:
        checks.append({
            'name': '相关性检查',
            'passed': False,
            'message': '有效数据不足，无法计算相关性',
            'severity': 'error'
        })
        passed = False

    # 检查 3: 缺失值
    missing_c = np.isnan(X_control).sum()
    missing_t = np.isnan(X_treatment).sum()
    missing_ratio = (missing_c + missing_t) / (n_c + n_t)

    missing_ok = missing_ratio < 0.3
    checks.append({
        'name': '缺失值检查',
        'passed': missing_ok,
        'message': f"控制组缺失: {missing_c}, 实验组缺失: {missing_t} ({missing_ratio:.1%})",
        'severity': 'warning' if missing_ratio > 0.1 else 'ok'
    })

    if missing_ratio > 0.5:  # 超过 50% 缺失是严重问题
        passed = False

    # 检查 4: 协变量平衡性
    if len(X_valid_c) > 2 and len(X_valid_t) > 2:
        t_stat, p_val = stats.ttest_ind(X_valid_c, X_valid_t)
        balance_ok = p_val > alpha
        checks.append({
            'name': '协变量平衡性检查',
            'passed': balance_ok,
            'message': f"p-value = {p_val:.4f}" + (" (不平衡，可能存在随机化问题!)" if not balance_ok else " (平衡)"),
            'severity': 'warning' if not balance_ok else 'ok'
        })

    # 打印报告
    print("=" * 60)
    print("CUPED 前置检查报告")
    print("=" * 60)

    for check in checks:
        status = '✅' if check['passed'] else ('⚠️' if check['severity'] == 'warning' else '❌')
        print(f"\n{status} {check['name']}")
        print(f"   {check['message']}")

    print("\n" + "=" * 60)
    if passed:
        print("✅ 所有关键检查通过，可以使用 CUPED")
    else:
        print("❌ 存在关键问题，建议不使用 CUPED 或先解决问题")

    return passed, checks
```

#### 2. **Add Answer Keys for 思考题**

在 cell 25 后添加答案 cell:

```markdown
### 💡 思考题答案

#### 问题 1: 协变量选择

**如果有多个候选协变量，如何选择最佳的？**

**答案**:
```
选择协变量的 4 个标准:

1. **相关性** (Correlation) - 最重要!
   - 计算每个协变量与结果 Y 的相关系数
   - 优先选择 |ρ| > 0.3 的
   - 可以用 R² 衡量解释力

2. **稳定性** (Stability)
   - 协变量本身的方差要稳定
   - 避免选择有异常值的变量
   - 检查协变量在实验前后是否平衡

3. **业务意义** (Business Sense)
   - 选择与结果有因果关系的变量
   - 避免"后门"变量（可能受处理影响）
   - 优先选择用户固有特征（年龄、性别）而非行为特征

4. **数据质量** (Data Quality)
   - 缺失率 < 20%
   - 测量误差小
   - 定义清晰，不易被操纵

实战技巧:
- 如果有多个协变量，可以用 PCA 降维
- 也可以用多元回归，但注意多重共线性
```

#### 问题 2: 多协变量 CUPED

**如果想同时使用多个协变量，该如何处理？**

**答案**:
```
2 种方法:

方法 1: **多元回归 CUPED**
```python
# 构建协变量矩阵
X_covariates = np.column_stack([X1, X2, X3])

# 回归 Y ~ X1 + X2 + X3
model = sm.OLS(Y, sm.add_constant(X_covariates)).fit()

# CUPED 调整
Y_pred = model.predict(sm.add_constant(X_covariates))
Y_adj = Y - (Y_pred - Y_pred.mean())
```

方法 2: **CUPAC (CUPED with Additional Covariates)**
- Google 提出的扩展方法
- 允许包含实验期间的协变量
- 通过正交化避免偏差

注意事项:
1. 协变量之间的多重共线性
2. 自由度损失（每个协变量消耗 1 个自由度）
3. 过拟合风险（协变量太多）

经验法则: 协变量数量 < 样本量 / 20
```

#### 问题 3: CUPED 比原始方法更差的情况

**在什么情况下，CUPED 可能比原始方法更差？**

**答案**:
```
4 种情况 CUPED 会更差:

1. **相关性极低** (ρ < 0.1)
   - 方差缩减 < 1%
   - θ 的估计误差反而增加方差
   - 损失自由度

2. **样本量太小** (n < 200/组)
   - θ 估计不稳定
   - 置信区间可能反而更宽

3. **协变量本身有偏差**
   - 测量误差大
   - 受处理影响
   - 导致引入新的偏差

4. **非线性关系**
   - Y 和 X 是非线性关系
   - 线性 CUPED 无效
   - 需要用非线性变换

判断标准:
- 做 A/A 测试验证
- 比较 CUPED 前后的置信区间宽度
- Bootstrap 评估 θ 的估计方差
```

#### 问题 4: CUPED vs 分层抽样

**CUPED 和分层抽样都能减少方差，两者有什么区别？**

**答案**:
```
核心区别:

| 维度 | 分层抽样 | CUPED |
|------|----------|-------|
| **时机** | 实验前（设计阶段） | 实验后（分析阶段） |
| **数据要求** | 需要提前知道分层变量 | 只要有历史数据即可 |
| **灵活性** | 固定，无法修改 | 灵活，可以事后选择协变量 |
| **假设** | 无需假设 | 需要线性关系假设 |

适用场景:
- **分层抽样**:
  * 有明确的重要分层变量（地域、年龄段）
  * 需要确保各层样本量
  * RCT 设计阶段

- **CUPED**:
  * 观测数据或已完成的实验
  * 有丰富的历史数据
  * 想要事后提升统计功效

最佳实践: 两者结合！
1. 实验设计时用分层随机化
2. 分析时用 CUPED 进一步减少方差
```
```

#### 3. **Add Interview Simulation**

```markdown
---

## 🎤 面试模拟环节

### 场景 1: CUPED 原理

**面试官**: "简单讲讲 CUPED 的原理，为什么它能减少方差？"

**你的回答（参考）**:
```
CUPED 的核心是利用辅助变量来"解释"结果变量的部分方差。

数学原理:
Y_adj = Y - θ(X - X̄)

其中 θ = Cov(Y,X) / Var(X)

关键洞见:
1. X 是实验前变量，与处理分配独立
2. 减去 θ(X - X̄) 不改变 Y 的期望值
3. 但减少了 Y 的方差: Var(Y_adj) = Var(Y)(1 - ρ²)
4. ρ 是 Y 和 X 的相关系数

直观理解:
- 如果用户历史 GMV 高，当前 GMV 也会高（相关性）
- CUPED 去掉了这种"可预测"的部分
- 只保留"随机"的部分
- 从而减少噪声，提高检验功效

类比: 就像考试时控制学生的智商，只看教学方法的纯效应
```

### 场景 2: 实际应用

**面试官**: "你们公司的实验平台用 CUPED 吗？怎么选择协变量的？"

**你的回答（参考）**:
```
是的，我们在所有 A/B 测试中都默认使用 CUPED。

协变量选择流程:
1. **自动化选择**:
   - 对于 GMV 类指标，用前 7 天同指标
   - 对于留存类指标，用历史留存
   - 对于互动类指标，用历史互动次数

2. **前置检查**:
   - 相关性 > 0.3
   - 缺失率 < 30%
   - 实验前两组平衡（p > 0.05）

3. **特殊处理**:
   - 新用户: 分层 CUPED（新用户不用，老用户用）
   - 多指标: 每个指标用自己的最佳协变量
   - 长实验: 用实验启动前的窗口，避免 anticipation

效果:
- 平均方差缩减 30-50%
- 样本量需求减少 30%
- 实验周期缩短 20%
```

### 场景 3: 问题诊断

**面试官**: "如果 CUPED 后方差反而变大了，可能是什么原因？"

**你的回答（参考）**:
```
5 种可能原因:

1. **相关性太低**:
   - 检查 ρ 是否 < 0.1
   - 方差缩减 = ρ²，太小没用

2. **协变量受处理影响**:
   - 用了实验期间的数据
   - 引入了偏差

3. **样本量太小**:
   - θ 估计不准确
   - 估计误差大于方差缩减收益

4. **数据质量问题**:
   - 协变量有异常值
   - 测量误差大

5. **分组不平衡**:
   - 实验组和对照组的 X 分布差异大
   - θ 在两组中可能不一样

诊断方法:
```python
# 检查相关性
rho = np.corrcoef(Y, X)[0, 1]
print(f"相关系数: {rho:.3f}, 理论方差缩减: {rho**2:.1%}")

# 检查 θ 稳定性
theta_c = np.cov(Y_c, X_c)[0, 1] / np.var(X_c)
theta_t = np.cov(Y_t, X_t)[0, 1] / np.var(X_t)
print(f"θ 控制组: {theta_c:.3f}, θ 实验组: {theta_t:.3f}")

# A/A 测试
# 在无效应数据上验证 CUPED 是否确实减少方差
```
```
```

---

## Pitfall 03: DID Violations 🔧

### Strengths:
- ✅ 平行趋势假设讲解清晰
- ✅ Event Study 图很直观
- ✅ Anticipation Effect 识别到位
- ✅ 提供了 synthetic control 作为替代方法

### Issues to Fix:

#### 1. **Complete TODO in Cell 19** - Diagnostic Pipeline

当前的 diagnostic pipeline 功能不全，需要补充:

**完整实现**:
```python
def did_diagnostic_pipeline(df, treatment_period=6):
    """
    完整的 DID 诊断流程
    """
    print("=" * 70)
    print("DID 诊断报告")
    print("=" * 70)

    # Step 1: 数据概览
    print("\n【Step 1: 数据概览】")
    n_units = df['unit'].nunique()
    n_treated = df[df['treated']==1]['unit'].nunique()
    n_control = df[df['treated']==0]['unit'].nunique()
    n_periods = df['period'].nunique()
    n_pre = df[df['period'] < treatment_period]['period'].nunique()
    n_post = df[df['period'] >= treatment_period]['period'].nunique()

    print(f"  总单位数: {n_units}")
    print(f"  处理组: {n_treated}, 对照组: {n_control}")
    print(f"  时期数: {n_periods} (前 {n_pre} 期, 后 {n_post} 期)")
    print(f"  处理时点: {treatment_period}")

    # Step 2: 平行趋势检验
    print("\n【Step 2: 平行趋势检验】")
    pt_result = parallel_trend_test(df, treatment_period)
    if pt_result['reject_H0']:
        print(f"  ❌ 拒绝平行趋势假设 (p = {pt_result['p_value']:.4f})")
        print(f"     处理组额外趋势: {pt_result['coefficient']:.4f}/期")
        parallel_ok = False
    else:
        print(f"  ✅ 不拒绝平行趋势假设 (p = {pt_result['p_value']:.4f})")
        parallel_ok = True

    # Step 3: Event Study (详细检查每一期)
    print("\n【Step 3: Event Study - 逐期检验】")
    es = event_study(df, treatment_period)

    # 检查处理前各期
    pre_periods = es[es['rel_time'] < 0].sort_values('rel_time')
    print("\n  处理前各期系数:")
    for _, row in pre_periods.iterrows():
        t = int(row['rel_time'])
        sig = '***' if (row['ci_lower'] > 0 or row['ci_upper'] < 0) else ''
        print(f"    t={t:+3d}: {row['coef']:+.3f} [{row['ci_lower']:+.3f}, {row['ci_upper']:+.3f}] {sig}")

    # Step 4: Anticipation Effect 检验
    print("\n【Step 4: Anticipation Effect 检验】")
    has_anticipation = diagnose_anticipation(df, treatment_period, n_pre_periods=2)

    # Step 5: DID 估计
    print("\n【Step 5: DID 估计】")

    # 标准 DID
    standard_coef, standard_se = did_estimate(df, treatment_period)
    print(f"  标准 DID: {standard_coef:.3f} (SE = {standard_se:.3f})")
    print(f"    95% CI: [{standard_coef - 1.96*standard_se:.3f}, {standard_coef + 1.96*standard_se:.3f}]")

    # 如果平行趋势不满足，尝试组特定趋势
    if not parallel_ok:
        gt_result = did_with_group_trends(df, treatment_period)
        print(f"  组特定趋势 DID: {gt_result['did_coef']:.3f} (SE = {gt_result['did_se']:.3f})")
        print(f"    处理组额外趋势系数: {gt_result['trend_coef']:.3f}")

    # Step 6: 稳健性检查
    print("\n【Step 6: 稳健性检查】")

    # 不同处理时点
    if has_anticipation:
        print("  检查不同处理时点的估计:")
        for tp in [treatment_period - 2, treatment_period - 1, treatment_period]:
            coef, se = did_estimate(df, treatment_period=tp)
            print(f"    处理时点={tp}: {coef:.3f} (SE={se:.3f})")

    # Placebo 检验（前移处理时点到处理前）
    if n_pre >= 4:
        print("\n  Placebo 检验 (前移处理时点):")
        placebo_period = treatment_period - 2
        placebo_coef, placebo_se = did_estimate(df[df['period'] < treatment_period],
                                                 treatment_period=placebo_period)
        placebo_sig = abs(placebo_coef) > 1.96 * placebo_se
        print(f"    Placebo DID (t={placebo_period}): {placebo_coef:.3f} (SE={placebo_se:.3f})")
        if placebo_sig:
            print(f"    ⚠️ Placebo 显著，平行趋势可能不满足!")
        else:
            print(f"    ✅ Placebo 不显著")

    # Step 7: 建议
    print("\n【Step 7: 诊断总结与建议】")
    print("=" * 70)

    if parallel_ok and not has_anticipation:
        print("✅ 平行趋势满足，无 Anticipation，可使用标准 DID")
        print(f"   推荐估计: {standard_coef:.3f} ± {1.96*standard_se:.3f}")
    elif has_anticipation:
        print("⚠️ 存在 Anticipation Effect，建议:")
        print("   1. 调整处理时点到 anticipation 开始时")
        print("   2. 明确说明估计的是「公告效应」还是「实施效应」")
        print("   3. 如果可能，分别估计两个效应")
    elif not parallel_ok:
        print("⚠️ 平行趋势不满足，建议:")
        print("   1. 使用组特定趋势 DID")
        print(f"      估计: {gt_result['did_coef']:.3f}")
        print("   2. 考虑合成控制法 (Synthetic Control)")
        print("   3. 寻找更好的对照组")
        print("   4. 改用其他识别策略（RDD, IV）")

    return {
        'parallel_ok': parallel_ok,
        'has_anticipation': has_anticipation,
        'standard_did': (standard_coef, standard_se),
        'event_study': es,
        'parallel_trend_test': pt_result
    }
```

#### 2. **Add 思考题 Answers**

```markdown
## 💡 思考题答案

### 问题 1: 平行趋势检验不拒绝 H0 就意味着成立吗?

**答案**:
```
NO! 这是常见误区。

统计学基本原理:
- "不拒绝 H0" ≠ "接受 H0"
- 不拒绝可能是因为:
  1. 检验功效不足 (power too low)
  2. 样本量太小
  3. 处理前期数太少

举例:
- 假设只有 2 个处理前期，即使趋势明显不同，由于数据点少，检验也可能不显著

正确态度:
1. **多角度验证**:
   - 统计检验 (p-value)
   - 可视化 (event study plot)
   - 领域知识 (是否合理)

2. **敏感性分析**:
   - 不同的趋势控制方式
   - 不同的时间窗口
   - 排除某些时期重新估计

3. **诚实汇报**:
   - 说明处理前期数
   - 报告检验功效
   - 承认局限性

面试加分点: 提到 \"absence of evidence is not evidence of absence\"
```

### 问题 2: 多期处理 (Staggered DID) 如何检验平行趋势?

**答案**:
```
Staggered DID 的挑战:
- 不同单位在不同时间接受处理
- 无法简单比较处理前后

解决方案:

方法 1: **Callaway & Sant'Anna (2021)**
```python
# 对每个 (处理时间 g, 日历时间 t) 组合分别做平行趋势检验
# 检验: 在 t-1 vs t-2, t-2 vs t-3, ... 时，
# 处理时间为 g 的组 vs never-treated 组的趋势是否平行
```

方法 2: **Event Study with Cohort FE**
```python
# 将时间转换为相对处理的时间
# 控制 cohort 固定效应
Y_it = α_i + λ_t + Σ_k β_k × 1{t - g_i = k} + ε_it
# 检验所有 k < 0 时 β_k = 0
```

方法 3: **Sun & Abraham (2021) - IW Estimator**
- 用 never-treated 或 not-yet-treated 作为对照
- 对每个 cohort 单独估计，然后加权平均

Python 实现:
```python
# 使用 pydid 包
from pydid import did2s
result = did2s(data, outcome='Y', treatment='D',
               cohort='g', time='t')
result.event_study_plot()  # 自动绘制 event study
```

关键: Staggered DID 的平行趋势是 cohort-specific 的
```

### 问题 3: 非线性趋势如何处理?

**答案**:
```
当趋势非线性时，组特定线性趋势 DID 失效。

5 种解决方案:

1. **多项式趋势**:
```python
# 二次趋势
formula = 'Y ~ C(unit) + C(period) + treated + period + period^2 + treated:period + treated:period^2 + did'
```

2. **非参数趋势**:
```python
# 用 spline 或 local polynomial
from scipy.interpolate import UnivariateSpline
# 拟合处理组和对照组各自的趋势
```

3. **Synthetic Control**:
- 不假设线性趋势
- 用对照组的加权组合拟合处理组的处理前轨迹
- 推断: 处理后真实值 - synthetic counterfactual

4. **Matrix Completion**:
- 将面板数据看作矩阵
- 用低秩矩阵补全方法估计反事实

5. **Change-in-Changes (Athey & Imbens 2006)**:
- 不依赖加性模型
- 允许时间效应因组而异

选择建议:
- 如果只是轻微非线性: 多项式趋势
- 如果单个处理单位: Synthetic Control
- 如果复杂面板: Matrix Completion
- 如果有分布信息: Change-in-Changes
```

---

## Pitfall 04: Weak IV 🔧

### Strengths:
- ✅ F 统计量规则讲得清楚
- ✅ Stock-Yogo 临界值很专业
- ✅ Anderson-Rubin CI 是正确的弱 IV 稳健方法
- ✅ LIML 作为补充很好

### Issues to Fix:

#### 1. **Complete All TODO Sections**

当前有 5 处 TODO 需要完整实现。

#### 2. **Add 思考题 Answers**

```markdown
## 💡 思考题答案

### 问题 1: 多弱工具变量能否\"加总\"成强工具？

**答案**:
```
是的，但有条件！

理论:
- 假设有 K 个弱工具变量，每个 F_k ≈ 5
- 联合 F 统计量 ≈ 5K（如果工具变量不相关）
- 所以 K ≥ 3 时，联合 F ≈ 15 > 10

但问题:
1. **工具变量通常相关**:
   - 如果高度相关，加总无帮助
   - 需要检查工具变量之间的相关性

2. **过度识别检验难通过**:
   - 工具越多，排斥性假设越难满足
   - Sargan 检验更容易拒绝

3. **有限样本性质差**:
   - Many-weak-instruments asymptotics
   - K/n → κ > 0 时，2SLS 仍然有偏

更好的方法:

**JIVE (Jackknife IV)**:
```python
# 对每个观测 i，用除了 i 之外的样本估计倾向得分
# 避免 overfitting bias
```

**LIML**:
- 在 many weak instruments 下比 2SLS 更稳健

**Post-Lasso IV**:
```python
# 用 Lasso 从众多弱工具中选择最相关的
from econml import DML
model = DML(model_y=Lasso(), model_t=Lasso())
```

面试要点:
- 不要无脑增加工具变量数量
- 检查工具变量的独立性
- 优先找一个强工具，而不是多个弱工具
```

### 问题 2: IV 估计的是什么效应？与 ATE 有何区别？

**答案**:
```
IV 估计的是 **LATE (Local Average Treatment Effect)**，不是 ATE!

定义:
LATE = E[Y(1) - Y(0) | Complier]

Complier: 那些"被工具变量说服"的人
- Z=1 时会接受处理 (D=1)
- Z=0 时不会接受处理 (D=0)

举例 (教育回报率):
- 工具变量: 距离最近大学的距离
- LATE: 那些因为离大学近而选择上大学的人的回报率
- 不包括:
  * Always-takers (不管远近都上大学)
  * Never-takers (不管远近都不上)
  * Defiers (离得近反而不上)

LATE vs ATE:

| 维度 | LATE | ATE |
|------|------|-----|
| 定义 | Compliers 的效应 | 全体的平均效应 |
| 识别 | 需要强假设 | 更一般 |
| 外推性 | 低 | 高 |
| 政策含义 | 边际效应 | 总体效应 |

Monotonicity 假设:
- LATE 需要假设无 Defiers
- 即 Z=1 不会导致有人从 D=1 变成 D=0

面试高级回答:
- IV 估计的是边际处理效应 (MTE) 的加权平均
- 权重取决于工具变量对处理的影响
- 如果效应异质性大，LATE 可能不代表 ATE
```

### 问题 3: 如何判断一个变量适合做工具变量？

**答案**:
```
判断 IV 的 3 个维度:

1. **相关性 (Relevance)** - 可检验 ✅
   统计检验:
   - 第一阶段 F > 10
   - t 统计量 > 3.16
   - R² > 0.1（经验值）

   思考:
   - Z 对 D 有因果影响吗？
   - 影响的机制是什么？

2. **排斥性 (Exclusion)** - 不可检验 ❌
   思考:
   - Z 影响 Y 的所有路径都经过 D 吗？
   - 有没有直接路径？
   - 画 DAG（有向无环图）检查

   间接证据:
   - Overidentification test (if K > 1)
   - Placebo 检验
   - 理论论证

3. **独立性 (Exogeneity)** - 部分可检验 ⚠️
   思考:
   - Z 是随机分配的吗？
   - Z 与未观测混淆变量独立吗？

   检验:
   - 如果是 RCT，自动满足
   - 如果是自然实验，检查 Z 的"as-if random"性
   - Balance test: Z 与观测协变量的关系

实战清单:

□ 画 DAG，确认所有路径
□ 第一阶段 F 统计量 > 10
□ 如果有多个 IV，做 Sargan 检验
□ 文献中这个 IV 被用过吗？
□ 领域专家认为合理吗？
□ 做敏感性分析（改变假设看结果稳定性）

面试金句:
"IV 的质量 90% 靠理论和领域知识，10% 靠统计检验"
```

---

## Pitfall 05: A/B Test Common Mistakes ⚠️ (Most Incomplete)

### Critical Issues:

1. **JSON Corruption** - 文件格式损坏，无法被 Jupyter 正确解析
   - 原因: 中文引号 ""  导致 JSON 格式错误
   - 需要修复: 将所有中文引号替换为英文引号

2. **Multiple Incomplete TODOs**:
   - Cell: detect_srm
   - Cell: simulate_peeking
   - Cell: alpha_spending_obf
   - Cell: bonferroni_correction & benjamini_hochberg
   - Cell: simulate_network_effects

3. **Missing Reference Implementations** for all TODOs

4. **No 思考题 or Interview Section**

### Recommended Actions:

#### 1. Fix JSON Corruption First

需要手动编辑 .ipynb 文件，将所有 `"` 和 `"` 替换为 `"`

#### 2. Complete All TODO Implementations

由于文件损坏，建议重新创建或从备份恢复。

#### 3. Add Complete Reference Implementations

每个 TODO 都需要:
- 清晰的提示
- 完整的参考答案
- 测试用例

---

## Cross-Cutting Recommendations

### 1. Add Consistent Interview Sections

每个 notebook 都应包含:

```markdown
## 🎤 面试模拟环节

### 场景 1: 基础理论
### 场景 2: 实际应用
### 场景 3: 问题诊断
### 场景 4: 快速判断（30秒挑战）
```

### 2. Add Answer Keys for All 思考题

格式:
```markdown
## 💡 思考题参考答案

### 问题 1: ...
**答案**:
```[详细分点回答]```

【基础回答】
【进阶回答】
【高级回答/面试加分点】
```

### 3. Add "Diagnostic Checklist" Sections

每个 notebook 结尾添加:

```markdown
## 📋 快速诊断清单

□ 检查项 1
□ 检查项 2
...

⚠️ 红线（绝对不能违反）:
- ...
- ...

💡 最佳实践:
- ...
- ...
```

### 4. Add Real Interview Questions

收集常见面试题:

```markdown
## 📝 真实面试题库

### 基础题（初级DS）
1. Q: ...
   A: ...

### 进阶题（高级DS）
1. Q: ...
   A: ...

### Case Study
场景: ...
问题: ...
参考答案: ...
```

---

## Priority Fix List

### P0 (Critical - Must Fix):
1. ✅ Fix Pitfall 05 JSON corruption
2. ✅ Complete all TODO implementations with reference answers
3. ✅ Add 思考题 answer keys to all notebooks

### P1 (High - Should Fix):
1. ✅ Add interview simulation sections to all notebooks
2. ✅ Add diagnostic checklists
3. ✅ Verify all code executes without errors

### P2 (Medium - Nice to Have):
1. Add real interview questions database
2. Add more visualization examples
3. Add links to related notebooks

---

## Estimated Time to Fix

- Pitfall 01: 1 hour (只需加答案)
- Pitfall 02: 2 hours (TODO + 答案)
- Pitfall 03: 2 hours (TODO + 答案)
- Pitfall 04: 3 hours (多个 TODO + 答案)
- Pitfall 05: 4 hours (修 JSON + 所有 TODO + 答案)

**Total: ~12 hours**

---

## Next Steps

1. Fix Pitfall 05 JSON corruption
2. Run all notebooks to identify runtime errors
3. Complete all TODOs systematically
4. Add all answer keys
5. Add interview sections
6. Final review and testing

---

## Conclusion

这 5 个 pitfall notebooks 的核心框架和教学思路都非常好，是真正的"面试送分题区"。主要问题是:

1. **完成度不一致** - Pitfall 01 最完整，Pitfall 05 最不完整
2. **TODO 未实现** - 影响学员自主学习
3. **缺少答案** - 无法自我验证
4. **缺少面试题** - 没有充分发挥"面试导向"的优势

完成这些修复后，这将成为市面上最好的因果推断 pitfalls 教程！

**建议**: 先修复 Pitfall 05 的 JSON 问题，然后按优先级逐个完善。
