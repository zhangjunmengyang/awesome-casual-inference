# Causal Inference Workbench - 项目指南

## 项目概述

这是一个因果推断学习与可视化平台，专注于 Model-Based 方法，涵盖深度学习和机器学习因果模型。
项目除了展示核心原理，另外的目的是作为教程项目，教我相关知识，循序渐进，体系化学习，在真实项目实践中学习，避免只懂公式的纸上谈兵。

## 工作模式 (最重要)

**核心原则: 你是调度者，不是执行者。**

不要事必躬亲。你的角色是:
- 理解用户意图，拆解任务
- 分发任务给 SubAgent
- 审核结果，做出判断
- 统筹全局，确保一致性

```
用户需求 -> 拆解任务 -> 分发给 SubAgent -> 审核结果 -> 交付
```

## 快速启动

```bash
# 安装依赖
pip install -r requirements.txt

# 运行应用
python app.py
```

访问 http://localhost:7860

## 项目结构

```
.
├── app.py                    # Gradio 主应用
├── requirements.txt          # 依赖
│
├── foundation_lab/           # 基础概念 (已完成)
│   ├── potential_outcomes.py # 潜在结果框架
│   ├── causal_dag.py         # 因果图
│   ├── confounding_bias.py   # 混淆偏差
│   └── selection_bias.py     # 选择偏差
│
├── uplift_lab/               # Uplift 模型 (已完成)
│   ├── meta_learners.py      # S/T/X-Learner
│   ├── uplift_tree.py        # Uplift 决策树
│   ├── evaluation.py         # Qini/Uplift 曲线
│   └── cate_comparison.py    # CATE 对比
│
├── deep_causal_lab/          # 深度因果 (已完成)
│   ├── tarnet.py             # TARNet
│   ├── dragonnet.py          # DragonNet
│   └── cevae.py              # CEVAE (待完善)
│
├── treatment_effect_lab/     # 处理效应 (待开发)
├── hetero_effect_lab/        # 异质效应 (待开发)
├── application_lab/          # 应用场景 (待开发)
├── evaluation_lab/           # 评估诊断 (待开发)
│
└── exercises/                # 练习代码
    └── chapter1_foundation/  # 已完成
```

## 开发规范

### 模块结构

每个 Lab 模块应包含:
- `__init__.py`: 模块导出
- `utils.py`: 工具函数
- 功能模块 (如 `meta_learners.py`)

### UI 组件规范

每个功能模块应实现 `render()` 函数:

```python
def render():
    """渲染 Gradio 界面"""
    with gr.Blocks() as block:
        gr.Markdown("## 模块标题")
        # UI 组件
    return None  # 或返回 load 事件
```

### 可视化规范

- 使用 Plotly 进行交互式可视化
- 颜色方案: #2D9CDB (主蓝), #27AE60 (绿), #EB5757 (红)
- 统一使用 `template='plotly_white'`

### 练习代码规范

练习文件包含:
1. TODO 注释标记需完成部分
2. 函数文档说明
3. 测试代码 (`if __name__ == "__main__"`)

## 待开发功能

### TreatmentEffectLab
- [ ] 倾向得分匹配 (PSM)
- [ ] 逆概率加权 (IPW)
- [ ] 双重稳健估计

### HeteroEffectLab
- [ ] Causal Forest
- [ ] BART
- [ ] 敏感性分析

### ApplicationLab
- [ ] 智能发券优化
- [ ] A/B 测试增强
- [ ] 用户分层干预

### EvaluationLab
- [ ] 平衡性检查
- [ ] 重叠假设检验
- [ ] 模型对比评估

## 技术栈

- UI: Gradio
- 可视化: Plotly
- 因果推断: econml, causalml, dowhy
- 深度学习: PyTorch
- ML: scikit-learn, xgboost

## 相关资源

- EconML: https://github.com/microsoft/EconML
- CausalML: https://github.com/uber/causalml
- DoWhy: https://github.com/py-why/dowhy
