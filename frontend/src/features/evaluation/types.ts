// Evaluation Lab 类型定义

export interface BalanceCheckParams {
  n_samples: number
  confounding_strength: number
  method: 'psm' | 'ipw' | 'none'
}

export interface OverlapCheckParams {
  n_samples: number
  confounding_strength: number
  show_trimming: boolean
}

export interface ModelComparisonParams {
  n_samples: number
  confounding_strength: number
  show_confidence_intervals: boolean
}

// 复用通用分析结果类型
export interface AnalysisResult {
  charts: Record<string, unknown>[]
  tables: Record<string, unknown>[]
  summary: string
  metrics: Record<string, unknown>
}
