// Treatment Effect Lab 类型定义

export interface PSMParams {
  n_samples: number
  confounding_strength: number
  caliper: number
}

export interface IPWParams {
  n_samples: number
  confounding_strength: number
  stabilized: boolean
  trimming: number
}

export interface DoublyRobustParams {
  n_samples: number
  confounding_strength: number
  outcome_model: 'linear' | 'rf' | 'xgb'
  propensity_model: 'logistic' | 'rf' | 'xgb'
}

// 复用通用分析结果类型
export interface AnalysisResult {
  charts: Record<string, unknown>[]
  tables: Record<string, unknown>[]
  summary: string
  metrics: Record<string, unknown>
}
