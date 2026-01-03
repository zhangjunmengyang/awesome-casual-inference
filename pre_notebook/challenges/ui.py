"""
挑战系统 UI - Gradio 界面
Kaggle 风格的竞赛体验
"""

import gradio as gr
import numpy as np
import pandas as pd
import traceback
from typing import Optional

from .challenge_1_ate_estimation import ATEEstimationChallenge
from .challenge_2_cate_prediction import CATEPredictionChallenge
from .challenge_3_uplift_ranking import UpliftRankingChallenge
from .leaderboard import Leaderboard


# 全局变量存储当前挑战
CURRENT_CHALLENGES = {}


def initialize_challenge(challenge_name: str, seed: int = 42):
    """初始化挑战"""
    global CURRENT_CHALLENGES

    if challenge_name == "ATE Estimation":
        challenge = ATEEstimationChallenge()
    elif challenge_name == "CATE Prediction":
        challenge = CATEPredictionChallenge()
    elif challenge_name == "Uplift Ranking":
        challenge = UpliftRankingChallenge()
    else:
        return None, None, "Invalid challenge name"

    # 生成数据
    train_data, test_data = challenge.generate_data(seed=seed)

    # 存储挑战实例
    CURRENT_CHALLENGES[challenge_name] = challenge

    # 返回数据预览
    train_preview = f"""
### Training Data Preview

Shape: {train_data.shape}

```
{train_data.head(10).to_string()}
```

### Test Data Preview

Shape: {test_data.shape}

```
{test_data.head(10).to_string()}
```
"""

    info = challenge.get_detailed_info()

    return train_preview, info, "Challenge initialized successfully!"


def run_baseline(challenge_name: str, method: str):
    """运行基线方法"""
    global CURRENT_CHALLENGES

    if challenge_name not in CURRENT_CHALLENGES:
        return None, "Please initialize the challenge first!"

    challenge = CURRENT_CHALLENGES[challenge_name]

    try:
        if challenge_name == "ATE Estimation":
            predictions = challenge.get_baseline_predictions(method)
            result = challenge.evaluate(predictions, user_name=f"Baseline ({method})")
        else:
            predictions = challenge.get_baseline_predictions(method)
            result = challenge.evaluate(predictions, user_name=f"Baseline ({method})")

        # 格式化结果
        result_md = f"""
### Baseline Result: {method}

**Score**: {result.score:.2f} / 100

**Primary Metric**: {result.primary_metric:.4f}

**Secondary Metrics**:
"""
        for key, value in result.secondary_metrics.items():
            result_md += f"- {key}: {value:.4f}\n"

        return None, result_md

    except Exception as e:
        return None, f"Error: {str(e)}\n\n{traceback.format_exc()}"


def submit_predictions(challenge_name: str, user_name: str, code: str):
    """
    提交预测代码

    代码应该定义一个函数返回预测结果
    """
    global CURRENT_CHALLENGES

    if challenge_name not in CURRENT_CHALLENGES:
        return None, "Please initialize the challenge first!"

    if not user_name.strip():
        return None, "Please provide a user name!"

    challenge = CURRENT_CHALLENGES[challenge_name]

    try:
        # 安全警告：exec() 执行用户代码存在安全风险
        # 在生产环境中应使用沙箱隔离（如 RestrictedPython 或容器化执行）
        # 当前实现仅适用于本地学习环境

        # 代码长度限制
        if len(code) > 50000:
            return None, "Error: Code is too long (max 50000 characters)"

        # 安全检查：使用 AST 解析来检测危险操作
        import ast

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return None, f"Syntax Error: {str(e)}"

        # 禁止的 AST 节点类型
        forbidden_nodes = {
            ast.Import: "import statements are not allowed (use np and pd directly)",
            ast.ImportFrom: "from...import statements are not allowed",
        }

        # 禁止的函数调用
        forbidden_calls = {
            'eval', 'exec', 'compile', 'open', 'input',
            '__import__', 'globals', 'locals', 'vars', 'dir',
            'getattr', 'setattr', 'delattr', 'hasattr',
            'breakpoint', 'exit', 'quit',
        }

        # 禁止的属性访问模式
        forbidden_attrs = {
            '__class__', '__base__', '__bases__', '__mro__',
            '__subclasses__', '__dict__', '__globals__',
            '__code__', '__closure__', '__func__',
            '__self__', '__module__', '__builtins__',
        }

        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.errors = []

            def visit_Import(self, node):
                self.errors.append("import statements are not allowed (use np and pd directly)")
                self.generic_visit(node)

            def visit_ImportFrom(self, node):
                self.errors.append("from...import statements are not allowed")
                self.generic_visit(node)

            def visit_Call(self, node):
                # 检查函数调用
                if isinstance(node.func, ast.Name):
                    if node.func.id in forbidden_calls:
                        self.errors.append(f"Function '{node.func.id}' is not allowed")
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in forbidden_calls:
                        self.errors.append(f"Function '{node.func.attr}' is not allowed")
                self.generic_visit(node)

            def visit_Attribute(self, node):
                # 检查危险属性访问
                if node.attr in forbidden_attrs:
                    self.errors.append(f"Accessing '{node.attr}' is not allowed")
                self.generic_visit(node)

        visitor = SecurityVisitor()
        visitor.visit(tree)

        if visitor.errors:
            return None, f"Security Error: {'; '.join(visitor.errors)}"

        # 定义受限的内置函数白名单
        safe_builtins = {
            'len': len, 'range': range, 'enumerate': enumerate, 'zip': zip,
            'map': map, 'filter': filter, 'sorted': sorted, 'reversed': reversed,
            'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
            'int': int, 'float': float, 'str': str, 'bool': bool, 'list': list,
            'dict': dict, 'tuple': tuple, 'set': set, 'print': print,
            'True': True, 'False': False, 'None': None,
            'isinstance': isinstance, 'type': type,
            'slice': slice, 'iter': iter, 'next': next,
            'all': all, 'any': any,
            'pow': pow, 'divmod': divmod,
            'Exception': Exception, 'ValueError': ValueError,
            'TypeError': TypeError, 'KeyError': KeyError,
            'IndexError': IndexError, 'AttributeError': AttributeError,
        }

        namespace = {
            '__builtins__': safe_builtins,
            'np': np,
            'pd': pd,
            'train_data': challenge.train_data.copy(),  # 使用副本防止修改原数据
            'test_data': challenge.test_data.copy(),
        }

        exec(code, namespace)

        # 获取预测结果
        # 用户代码应该定义 'predictions' 变量或 'predict()' 函数
        if 'predictions' in namespace:
            predictions = namespace['predictions']
        elif 'predict' in namespace:
            predictions = namespace['predict']()
        else:
            return None, "Error: Code must define 'predictions' variable or 'predict()' function"

        # 转换为 numpy 数组
        if isinstance(predictions, pd.Series):
            predictions = predictions.values
        elif isinstance(predictions, list):
            predictions = np.array(predictions)
        elif isinstance(predictions, (int, float)):
            predictions = np.array([predictions])

        # 评估
        result = challenge.evaluate(predictions, user_name=user_name.strip())

        # 添加到排行榜
        leaderboard = Leaderboard(challenge_name)
        leaderboard.add_submission(result)

        # 格式化结果
        result_md = f"""
### Submission Result

**User**: {result.user_name}
**Time**: {result.submission_time}

---

**Score**: {result.score:.2f} / 100

**Primary Metric**: {result.primary_metric:.4f}

**Secondary Metrics**:
"""
        for key, value in result.secondary_metrics.items():
            result_md += f"- {key}: {value:.4f}\n"

        result_md += "\n---\n\nYour submission has been added to the leaderboard!"

        # 更新排行榜
        leaderboard_fig = leaderboard.plot_rankings(top_n=10)

        return leaderboard_fig, result_md

    except Exception as e:
        return None, f"Error executing code:\n\n{str(e)}\n\n{traceback.format_exc()}"


def view_leaderboard(challenge_name: str, top_n: int):
    """查看排行榜"""
    leaderboard = Leaderboard(challenge_name)

    # 图表
    fig = leaderboard.plot_rankings(top_n=top_n)

    # Markdown 表格
    md = leaderboard.get_leaderboard_markdown(top_n=top_n)

    return fig, md


def view_user_progress(challenge_name: str, user_name: str):
    """查看用户进步"""
    if not user_name.strip():
        return None, "Please enter a user name"

    leaderboard = Leaderboard(challenge_name)
    fig = leaderboard.plot_user_progress(user_name.strip())

    # 用户历史
    history = leaderboard.get_user_history(user_name.strip())

    if history.empty:
        md = f"No submissions from {user_name}"
    else:
        md = f"### {user_name}'s Submission History\n\n"
        md += f"Total submissions: {len(history)}\n\n"
        md += f"Best score: {history['score'].max():.2f}\n\n"
        md += f"Latest score: {history['score'].iloc[-1]:.2f}\n\n"

    return fig, md


def render():
    """渲染 Gradio 界面"""

    with gr.Blocks() as block:
        gr.Markdown("""
## Challenges - 因果推断竞赛挑战

Kaggle 风格的因果推断挑战，测试你的技能，与他人竞争!

### 可用挑战

| Challenge | Difficulty | Task | Metric |
|-----------|-----------|------|--------|
| **ATE Estimation** | Beginner | 从观察数据估计 ATE | Relative Error |
| **CATE Prediction** | Intermediate | 预测个体处理效应 | PEHE |
| **Uplift Ranking** | Advanced | 按 uplift 排序用户 | AUUC |

---
        """)

        with gr.Tabs() as challenge_tabs:

            # ==================== ATE Estimation Challenge ====================
            with gr.Tab("ATE Estimation", id="ate"):
                gr.Markdown("""
### Challenge 1: ATE Estimation

从观察性数据中估计平均处理效应。数据存在混淆偏差，朴素估计会有偏!

**Difficulty**: Beginner
**Metric**: Relative Error (越小越好)
**Goal**: 相对误差 < 10%
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        ate_seed = gr.Number(value=42, label="Random Seed", precision=0)
                        ate_init_btn = gr.Button("Initialize Challenge", variant="primary")

                with gr.Row():
                    with gr.Column():
                        ate_data_preview = gr.Markdown("Click 'Initialize Challenge' to start")
                    with gr.Column():
                        ate_challenge_info = gr.Markdown()

                gr.Markdown("### Try Baseline Methods")

                with gr.Row():
                    ate_baseline_method = gr.Radio(
                        choices=['naive', 'ipw', 'matching'],
                        value='naive',
                        label="Baseline Method"
                    )
                    ate_baseline_btn = gr.Button("Run Baseline")

                ate_baseline_result = gr.Markdown()

                gr.Markdown("---\n### Submit Your Solution")

                with gr.Row():
                    with gr.Column():
                        ate_user_name = gr.Textbox(label="Your Name", placeholder="Enter your name")
                        ate_code = gr.Code(
                            value=ATEEstimationChallenge().get_starter_code(),
                            language="python",
                            label="Your Code",
                            lines=20
                        )
                        ate_submit_btn = gr.Button("Submit", variant="primary")

                with gr.Row():
                    with gr.Column():
                        ate_result = gr.Markdown()
                    with gr.Column():
                        ate_leaderboard_plot = gr.Plot()

                # Event handlers
                ate_init_btn.click(
                    fn=lambda seed: initialize_challenge("ATE Estimation", int(seed)),
                    inputs=[ate_seed],
                    outputs=[ate_data_preview, ate_challenge_info, ate_result]
                )

                ate_baseline_btn.click(
                    fn=lambda method: run_baseline("ATE Estimation", method),
                    inputs=[ate_baseline_method],
                    outputs=[ate_leaderboard_plot, ate_baseline_result]
                )

                ate_submit_btn.click(
                    fn=lambda user, code: submit_predictions("ATE Estimation", user, code),
                    inputs=[ate_user_name, ate_code],
                    outputs=[ate_leaderboard_plot, ate_result]
                )

            # ==================== CATE Prediction Challenge ====================
            with gr.Tab("CATE Prediction", id="cate"):
                gr.Markdown("""
### Challenge 2: CATE Prediction

预测每个个体的条件平均处理效应 (CATE)。效应具有复杂的异质性!

**Difficulty**: Intermediate
**Metric**: PEHE (越小越好)
**Goal**: PEHE < 2.0
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        cate_seed = gr.Number(value=42, label="Random Seed", precision=0)
                        cate_init_btn = gr.Button("Initialize Challenge", variant="primary")

                with gr.Row():
                    with gr.Column():
                        cate_data_preview = gr.Markdown("Click 'Initialize Challenge' to start")
                    with gr.Column():
                        cate_challenge_info = gr.Markdown()

                gr.Markdown("### Try Baseline Methods")

                with gr.Row():
                    cate_baseline_method = gr.Radio(
                        choices=['s_learner', 't_learner', 'x_learner'],
                        value='t_learner',
                        label="Baseline Method"
                    )
                    cate_baseline_btn = gr.Button("Run Baseline")

                cate_baseline_result = gr.Markdown()

                gr.Markdown("---\n### Submit Your Solution")

                with gr.Row():
                    with gr.Column():
                        cate_user_name = gr.Textbox(label="Your Name", placeholder="Enter your name")
                        cate_code = gr.Code(
                            value=CATEPredictionChallenge().get_starter_code(),
                            language="python",
                            label="Your Code",
                            lines=20
                        )
                        cate_submit_btn = gr.Button("Submit", variant="primary")

                with gr.Row():
                    with gr.Column():
                        cate_result = gr.Markdown()
                    with gr.Column():
                        cate_leaderboard_plot = gr.Plot()

                # Event handlers
                cate_init_btn.click(
                    fn=lambda seed: initialize_challenge("CATE Prediction", int(seed)),
                    inputs=[cate_seed],
                    outputs=[cate_data_preview, cate_challenge_info, cate_result]
                )

                cate_baseline_btn.click(
                    fn=lambda method: run_baseline("CATE Prediction", method),
                    inputs=[cate_baseline_method],
                    outputs=[cate_leaderboard_plot, cate_baseline_result]
                )

                cate_submit_btn.click(
                    fn=lambda user, code: submit_predictions("CATE Prediction", user, code),
                    inputs=[cate_user_name, cate_code],
                    outputs=[cate_leaderboard_plot, cate_result]
                )

            # ==================== Uplift Ranking Challenge ====================
            with gr.Tab("Uplift Ranking", id="uplift"):
                gr.Markdown("""
### Challenge 3: Uplift Ranking

对用户按 uplift 排序，最大化营销 ROI。识别 Persuadables，避免 Sleeping Dogs!

**Difficulty**: Advanced
**Metric**: AUUC (越大越好)
**Goal**: Normalized AUUC > 0.7
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        uplift_seed = gr.Number(value=42, label="Random Seed", precision=0)
                        uplift_init_btn = gr.Button("Initialize Challenge", variant="primary")

                with gr.Row():
                    with gr.Column():
                        uplift_data_preview = gr.Markdown("Click 'Initialize Challenge' to start")
                    with gr.Column():
                        uplift_challenge_info = gr.Markdown()

                gr.Markdown("### Try Baseline Methods")

                with gr.Row():
                    uplift_baseline_method = gr.Radio(
                        choices=['t_learner', 'class_transformation'],
                        value='t_learner',
                        label="Baseline Method"
                    )
                    uplift_baseline_btn = gr.Button("Run Baseline")

                uplift_baseline_result = gr.Markdown()

                gr.Markdown("---\n### Submit Your Solution")

                with gr.Row():
                    with gr.Column():
                        uplift_user_name = gr.Textbox(label="Your Name", placeholder="Enter your name")
                        uplift_code = gr.Code(
                            value=UpliftRankingChallenge().get_starter_code(),
                            language="python",
                            label="Your Code",
                            lines=20
                        )
                        uplift_submit_btn = gr.Button("Submit", variant="primary")

                with gr.Row():
                    with gr.Column():
                        uplift_result = gr.Markdown()
                    with gr.Column():
                        uplift_leaderboard_plot = gr.Plot()

                # Event handlers
                uplift_init_btn.click(
                    fn=lambda seed: initialize_challenge("Uplift Ranking", int(seed)),
                    inputs=[uplift_seed],
                    outputs=[uplift_data_preview, uplift_challenge_info, uplift_result]
                )

                uplift_baseline_btn.click(
                    fn=lambda method: run_baseline("Uplift Ranking", method),
                    inputs=[uplift_baseline_method],
                    outputs=[uplift_leaderboard_plot, uplift_baseline_result]
                )

                uplift_submit_btn.click(
                    fn=lambda user, code: submit_predictions("Uplift Ranking", user, code),
                    inputs=[uplift_user_name, uplift_code],
                    outputs=[uplift_leaderboard_plot, uplift_result]
                )

            # ==================== Leaderboard Tab ====================
            with gr.Tab("Leaderboard", id="leaderboard"):
                gr.Markdown("""
## Leaderboard - 排行榜

查看所有挑战的排名和用户进步
                """)

                with gr.Row():
                    lb_challenge = gr.Radio(
                        choices=['ATE Estimation', 'CATE Prediction', 'Uplift Ranking'],
                        value='ATE Estimation',
                        label="Select Challenge"
                    )
                    lb_top_n = gr.Slider(5, 20, value=10, step=1, label="Show Top N")
                    lb_view_btn = gr.Button("View Leaderboard", variant="primary")

                with gr.Row():
                    with gr.Column():
                        lb_plot = gr.Plot()
                    with gr.Column():
                        lb_table = gr.Markdown()

                lb_view_btn.click(
                    fn=view_leaderboard,
                    inputs=[lb_challenge, lb_top_n],
                    outputs=[lb_plot, lb_table]
                )

                gr.Markdown("---\n### User Progress")

                with gr.Row():
                    progress_user = gr.Textbox(label="User Name", placeholder="Enter user name")
                    progress_challenge = gr.Radio(
                        choices=['ATE Estimation', 'CATE Prediction', 'Uplift Ranking'],
                        value='ATE Estimation',
                        label="Challenge"
                    )
                    progress_btn = gr.Button("View Progress")

                with gr.Row():
                    with gr.Column():
                        progress_plot = gr.Plot()
                    with gr.Column():
                        progress_info = gr.Markdown()

                progress_btn.click(
                    fn=view_user_progress,
                    inputs=[progress_challenge, progress_user],
                    outputs=[progress_plot, progress_info]
                )

        gr.Markdown("""
---

### How to Participate

1. **Initialize**: Click "Initialize Challenge" to generate data
2. **Explore**: Review the data and challenge description
3. **Baseline**: Try baseline methods to understand the task
4. **Code**: Write your solution in the code editor
5. **Submit**: Enter your name and submit your predictions
6. **Compete**: Check the leaderboard and improve your score!

### Tips

- Start with baseline methods to get a benchmark
- Read the challenge description carefully
- Understand the evaluation metric
- Try multiple approaches and iterate
- Check the leaderboard for inspiration

### Evaluation Metrics

- **ATE Estimation**: Relative Error = |estimate - true| / |true|
- **CATE Prediction**: PEHE = sqrt(MSE(CATE))
- **Uplift Ranking**: AUUC = Area Under Uplift Curve (Qini)

### Scoring

Scores range from 0 to 100, with bonuses for exceptional performance:
- 90-100: Excellent
- 70-90: Good
- 50-70: Average
- <50: Needs improvement

Good luck!
        """)

    return None


if __name__ == "__main__":
    # 测试界面
    demo = gr.Blocks()
    with demo:
        render()

    demo.launch()
