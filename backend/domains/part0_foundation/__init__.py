"""
Part 0: 因果思维基础模块

本模块提供因果推断的核心概念和分析工具，包括:
1. 潜在结果框架 (Potential Outcomes Framework)
2. 因果图与DAG (Causal DAG)
3. 识别策略框架 (Identification Strategies)
4. 偏差类型分析 (Bias Types)
"""

from .potential_outcomes import (
    generate_potential_outcomes_data,
    visualize_potential_outcomes,
    demonstrate_fundamental_problem
)

from .causal_dag import (
    create_dag_visualization,
    analyze_dag_structure,
    identify_backdoor_paths,
    simulate_confounding_effect
)

from .identification_strategies import (
    get_identification_strategy,
    create_strategy_decision_tree,
    evaluate_identification_assumptions,
    recommend_methods
)

from .bias_types import (
    analyze_confounding_bias,
    analyze_selection_bias,
    analyze_measurement_bias,
    demonstrate_simpsons_paradox,
    demonstrate_berksons_paradox
)

from .api import (
    analyze_potential_outcomes,
    analyze_causal_dag,
    analyze_confounding_bias,
    analyze_selection_bias,
    analyze_identification_strategy,
    analyze_bias_comparison
)

__all__ = [
    # 潜在结果
    'generate_potential_outcomes_data',
    'visualize_potential_outcomes',
    'demonstrate_fundamental_problem',

    # 因果图
    'create_dag_visualization',
    'analyze_dag_structure',
    'identify_backdoor_paths',
    'simulate_confounding_effect',

    # 识别策略
    'get_identification_strategy',
    'create_strategy_decision_tree',
    'evaluate_identification_assumptions',
    'recommend_methods',

    # 偏差类型
    'analyze_confounding_bias',
    'analyze_selection_bias',
    'analyze_measurement_bias',
    'demonstrate_simpsons_paradox',
    'demonstrate_berksons_paradox',

    # API 接口
    'analyze_potential_outcomes',
    'analyze_causal_dag',
    'analyze_confounding_bias',
    'analyze_selection_bias',
    'analyze_identification_strategy',
    'analyze_bias_comparison',
]
