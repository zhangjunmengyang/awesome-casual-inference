"""
Causal Inference Workbench - Gradio 应用主入口
一站式因果推断学习与可视化平台
"""

import gradio as gr

# 自定义 CSS - 因果推断主题
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --color-primary: #2D9CDB;
    --color-primary-light: #EBF5FB;
    --color-primary-dark: #1A5F7A;
    --color-success: #27AE60;
    --color-warning: #F2994A;
    --color-error: #EB5757;
    --font-mono: 'IBM Plex Mono', monospace;
}

.gradio-container {
    font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    max-width: 1400px !important;
}

/* 主标题样式 */
.main-header {
    background: linear-gradient(135deg, #EBF5FB 0%, #D6EAF8 100%);
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    border: 1px solid #AED6F1;
}

/* 主要按钮 */
button.primary, button[variant="primary"] {
    background: linear-gradient(135deg, #2D9CDB 0%, #1A5F7A 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    border-radius: 8px !important;
}

button.primary:hover, button[variant="primary"]:hover {
    background: linear-gradient(135deg, #1A5F7A 0%, #154360 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(45, 156, 219, 0.3) !important;
}

/* Tab 样式 */
.tab-nav button {
    font-weight: 600 !important;
    color: #6B7280 !important;
    border-radius: 8px 8px 0 0 !important;
}

.tab-nav button:hover {
    color: #2D9CDB !important;
    background: #EBF5FB !important;
}

.tab-nav button.selected {
    border-bottom: 3px solid #2D9CDB !important;
    color: #2D9CDB !important;
    background: #EBF5FB !important;
}

/* 信息面板 */
.info-panel {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 16px;
}

/* Markdown 表格 */
.prose table {
    border: 1px solid #E5E7EB !important;
}

.prose th, .prose td {
    border: 1px solid #E5E7EB !important;
    padding: 8px 12px !important;
}

/* 滑块颜色 */
input[type="range"]::-webkit-slider-thumb {
    background: #2D9CDB !important;
}

/* 隐藏页脚 */
footer {
    display: none !important;
}

/* 代码块 */
.prose code {
    background: #F3F4F6;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: var(--font-mono);
}
"""

# 自定义主题
CUSTOM_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont("Source Sans Pro"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
    radius_size="md",
    spacing_size="md",
).set(
    button_primary_background_fill="linear-gradient(135deg, #2D9CDB 0%, #1A5F7A 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #1A5F7A 0%, #154360 100%)",
    button_primary_text_color="white",
    slider_color="#2D9CDB",
    checkbox_background_color_selected="#2D9CDB",
)


def create_app():
    """创建 Gradio 应用"""

    with gr.Blocks(
        title="Causal Inference Workbench",
        theme=CUSTOM_THEME,
        css=CUSTOM_CSS,
        analytics_enabled=False
    ) as app:

        # 标题头部
        gr.HTML("""
        <div style="
            background: linear-gradient(135deg, #EBF5FB 0%, #D6EAF8 100%);
            border-radius: 12px;
            padding: 24px 32px;
            margin-bottom: 24px;
            border: 1px solid #AED6F1;
            display: flex;
            align-items: center;
            gap: 16px;
        ">
            <div style="
                width: 56px;
                height: 56px;
                background: linear-gradient(135deg, #2D9CDB 0%, #1A5F7A 100%);
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 28px;
            ">
                <span style="filter: brightness(0) invert(1);">&#x1F4CA;</span>
            </div>
            <div>
                <h1 style="margin: 0; font-size: 1.75rem; font-weight: 700; color: #1A5F7A;">
                    Causal Inference Workbench
                </h1>
                <p style="margin: 4px 0 0 0; color: #5D6D7E; font-size: 1rem;">
                    Interactive platform for learning causal inference
                </p>
            </div>
            <div style="margin-left: auto; display: flex; gap: 8px;">
                <span style="
                    background: #27AE60;
                    color: white;
                    padding: 4px 12px;
                    border-radius: 16px;
                    font-size: 0.75rem;
                    font-weight: 600;
                ">Model-Based</span>
                <span style="
                    background: #9B59B6;
                    color: white;
                    padding: 4px 12px;
                    border-radius: 16px;
                    font-size: 0.75rem;
                    font-weight: 600;
                ">Deep Learning</span>
            </div>
        </div>
        """)

        # 主导航 Tabs
        with gr.Tabs() as main_tabs:

            # ==================== FoundationLab ====================
            with gr.Tab("FoundationLab", id="foundation"):
                gr.Markdown("""
                ## FoundationLab - 因果推断基础概念

                这是你学习因果推断的起点，涵盖最核心的概念和框架。
                """)

                from foundation_lab import (
                    potential_outcomes,
                    causal_dag,
                    confounding_bias,
                    selection_bias
                )

                with gr.Tabs() as foundation_tabs:
                    with gr.Tab("Potential Outcomes", id="po"):
                        potential_outcomes.render()

                    with gr.Tab("Causal DAG", id="dag"):
                        causal_dag.render()

                    with gr.Tab("Confounding Bias", id="confound"):
                        confounding_bias.render()

                    with gr.Tab("Selection Bias", id="selection"):
                        selection_bias.render()

            # ==================== TreatmentEffectLab ====================
            with gr.Tab("TreatmentEffectLab", id="treatment"):
                gr.Markdown("""
                ## TreatmentEffectLab - 处理效应估计

                学习如何从观测数据中估计因果效应，包括 PSM、IPW、双重稳健等方法。
                """)

                from treatment_effect_lab import (
                    propensity_score,
                    ipw,
                    doubly_robust
                )

                with gr.Tabs() as treatment_tabs:
                    with gr.Tab("Propensity Score Matching", id="psm"):
                        propensity_score.render()

                    with gr.Tab("Inverse Probability Weighting", id="ipw"):
                        ipw.render()

                    with gr.Tab("Doubly Robust", id="dr"):
                        doubly_robust.render()

            # ==================== UpliftLab ====================
            with gr.Tab("UpliftLab", id="uplift"):
                gr.Markdown("""
                ## UpliftLab - 增益模型实验室

                学习 Uplift Modeling，估计个体条件平均处理效应 (CATE)。
                """)

                from uplift_lab import (
                    meta_learners,
                    uplift_tree,
                    cate_comparison,
                    evaluation
                )

                with gr.Tabs() as uplift_tabs:
                    with gr.Tab("Meta-Learners", id="meta"):
                        meta_learners.render()

                    with gr.Tab("Uplift Tree", id="tree"):
                        uplift_tree.render()

                    with gr.Tab("CATE Comparison", id="cate"):
                        cate_comparison.render()

                    with gr.Tab("Evaluation", id="eval"):
                        evaluation.render()

            # ==================== DeepCausalLab ====================
            with gr.Tab("DeepCausalLab", id="deep"):
                gr.Markdown("""
                ## DeepCausalLab - 深度因果模型

                探索使用深度学习进行因果推断，包括 TARNet、DragonNet、CEVAE 等。
                """)

                from deep_causal_lab import (
                    tarnet,
                    dragonnet,
                    cevae
                )

                with gr.Tabs() as deep_tabs:
                    with gr.Tab("TARNet", id="tarnet"):
                        tarnet.render()

                    with gr.Tab("DragonNet", id="dragon"):
                        dragonnet.render()

                    with gr.Tab("CEVAE", id="cevae"):
                        cevae.render()

            # ==================== Challenges ====================
            with gr.Tab("Challenges", id="challenges"):
                gr.Markdown("""
                ## Challenges - 因果推断竞赛

                Kaggle 风格的挑战，测试你的因果推断技能!
                """)

                from challenges import ui as challenges_ui
                challenges_ui.render()

            # ==================== HeteroEffectLab ====================
            with gr.Tab("HeteroEffectLab", id="hetero"):
                gr.Markdown("""
                ## HeteroEffectLab - 异质性处理效应实验室

                分析处理效应的异质性，识别受益人群，提供精准干预决策支持。
                """)

                from hetero_effect_lab import (
                    causal_forest,
                    sensitivity,
                    cate_visualization
                )

                with gr.Tabs() as hetero_tabs:
                    with gr.Tab("Causal Forest", id="cf"):
                        causal_forest.render()

                    with gr.Tab("Sensitivity Analysis", id="sensitivity"):
                        sensitivity.render()

                    with gr.Tab("CATE Visualization", id="cate_viz"):
                        cate_visualization.render()

            # ==================== ApplicationLab ====================
            with gr.Tab("ApplicationLab", id="application"):
                gr.Markdown("""
                ## ApplicationLab - 行业应用案例

                将因果推断应用于真实业务场景，解决实际问题。
                """)

                from application_lab import (
                    coupon_optimization,
                    ab_enhancement,
                    user_targeting
                )

                with gr.Tabs() as application_tabs:
                    with gr.Tab("Coupon Optimization", id="coupon"):
                        coupon_optimization.render()

                    with gr.Tab("A/B Enhancement", id="ab"):
                        ab_enhancement.render()

                    with gr.Tab("User Targeting", id="targeting"):
                        user_targeting.render()

            # ==================== IndustryCases ====================
            with gr.Tab("IndustryCases", id="industry_cases"):
                gr.Markdown("""
                ## IndustryCases - 行业真实案例

                展示 DoorDash、Netflix、Uber 等科技公司的因果推断应用。
                """)

                from industry_cases import (
                    doordash_delivery,
                    netflix_recommendation,
                    uber_surge_pricing
                )

                with gr.Tabs() as industry_tabs:
                    with gr.Tab("DoorDash Delivery", id="doordash"):
                        doordash_delivery.render()

                    with gr.Tab("Netflix Recommendation", id="netflix"):
                        netflix_recommendation.render()

                    with gr.Tab("Uber Surge Pricing", id="uber"):
                        uber_surge_pricing.render()

            # ==================== EvaluationLab ====================
            with gr.Tab("EvaluationLab", id="evaluation"):
                gr.Markdown("""
                ## EvaluationLab - 评估与诊断

                因果推断模型的评估与诊断工具。
                """)

                from evaluation_lab import (
                    balance_check,
                    overlap_check,
                    model_comparison
                )

                with gr.Tabs() as evaluation_tabs:
                    with gr.Tab("Balance Check", id="balance"):
                        balance_check.render()

                    with gr.Tab("Overlap Check", id="overlap"):
                        overlap_check.render()

                    with gr.Tab("Model Comparison", id="comparison"):
                        model_comparison.render()

        # 底部信息
        gr.HTML("""
        <div style="
            margin-top: 24px;
            padding: 16px;
            background: #F8FAFC;
            border-radius: 8px;
            border: 1px solid #E2E8F0;
            text-align: center;
            color: #6B7280;
            font-size: 0.875rem;
        ">
            <strong>Causal Inference Workbench</strong> |
            A learning platform for causal inference |
            Built with Gradio
        </div>
        """)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
