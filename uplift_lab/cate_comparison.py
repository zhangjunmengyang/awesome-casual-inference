"""
CATE 估计对比模块

比较不同 CATE 估计方法的效果
"""

import gradio as gr


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## CATE 估计对比

CATE (Conditional Average Treatment Effect) 是给定协变量 X 的条件处理效应。

---

> ⚠️ **模块状态: 开发中**
>
> 该模块的完整实现正在开发中。
>
> **现有资源:**
> - **Meta-Learners**: 请参考 [Meta-Learners](meta_learners) 标签页
> - **Causal Forest**: 请参考 [HeteroEffectLab](../hetero_effect_lab) 模块
> - **Deep Learning**: 请参考 [DeepCausalLab](../deep_causal_lab) 模块
>
> 本模块计划实现多种 CATE 估计方法的统一对比功能。

---

### 估计方法概览

| 方法 | 类型 | 特点 | 推荐场景 |
|------|------|------|----------|
| **S-Learner** | Meta-Learner | 单模型 | 大样本，弱异质性 |
| **T-Learner** | Meta-Learner | 双模型 | 中等异质性 |
| **X-Learner** | Meta-Learner | 利用两阶段估计 | 样本不平衡 |
| **R-Learner** | Meta-Learner | Robinson 分解 | 混淆控制 |
| **Causal Forest** | Tree-based | 诚实分裂 | 强异质性，理论保证 |
| **TARNet** | Deep Learning | 双头网络 | 大样本，复杂特征 |
| **DragonNet** | Deep Learning | 带倾向得分头 | 端到端学习 |

### 练习

完成 `exercises/chapter3_uplift/ex1_meta_learners.py` 中的练习来比较这些方法。
        """)

    return None
