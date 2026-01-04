// Uplift Lab 类型定义

export interface MetaLearnersParams {
  n_samples: number
  effect_type: 'constant' | 'heterogeneous' | 'complex'
  noise_level: number
}

export interface UpliftTreeParams {
  n_samples: number
  feature_effect: number
  criterion: 'KL' | 'ED' | 'Chi'
}

export interface UpliftEvaluationParams {
  n_samples: number
  model_quality: 'perfect' | 'good' | 'medium' | 'poor'
}

// 复用通用分析结果类型
export interface AnalysisResult {
  charts: Record<string, unknown>[]
  tables: Record<string, unknown>[]
  summary: string
  metrics: Record<string, unknown>
}
