// Hetero Effect Lab 类型定义

export interface CausalForestParams {
  n_samples: number
  effect_heterogeneity: 'weak' | 'moderate' | 'strong'
  confounding_strength: number
  n_trees: number
}

export interface CATEVisualizationParams {
  n_samples: number
  effect_heterogeneity: 'weak' | 'moderate' | 'strong'
  n_bootstrap: number
  n_subgroups: number
}

export interface SensitivityParams {
  n_samples: number
  confounder_strength: number
  correlation_with_x: number
  max_gamma: number
}

// 复用通用分析结果类型
export interface AnalysisResult {
  charts: Record<string, unknown>[]
  tables: Record<string, unknown>[]
  summary: string
  metrics: Record<string, unknown>
}
