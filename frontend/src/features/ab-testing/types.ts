// A/B Testing 类型定义

export interface ExperimentAnalysisParams {
  n_samples: number
  true_effect: number
  metric_type: 'proportion' | 'continuous'
  use_cuped: boolean
}

export interface SampleSizeParams {
  baseline: number
  mde: number
  alpha: number
  power: number
}

export interface PowerAnalysisParams {
  sample_size: number
  baseline: number
  effect_size: number
  alpha: number
}

export interface ReportGenerationParams {
  experiment_name: string
  hypothesis: string
  n_samples: number
  baseline: number
  true_effect: number
}

// 复用通用分析结果类型
export interface AnalysisResult {
  charts: Record<string, unknown>[]
  tables: Record<string, unknown>[]
  summary: string
  metrics: Record<string, unknown>
}
