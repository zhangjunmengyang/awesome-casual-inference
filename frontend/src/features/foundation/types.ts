// Foundation Lab 类型定义

export interface PotentialOutcomesParams {
  n_samples: number
  treatment_effect: number
  noise_std: number
  confounding_strength: number
}

export interface CausalDAGParams {
  scenario: 'confounding' | 'mediation' | 'collider'
}

export interface ConfoundingBiasParams {
  n_samples: number
  confounding_strength: number
  treatment_effect: number
}

export interface SelectionBiasParams {
  n_samples: number
  selection_strength: number
  treatment_effect: number
}

// 通用分析结果类型
export interface ChartData {
  type: string
  title: string
  x_label: string
  y_label: string
  series: Record<string, unknown>[]
}

export interface TableData {
  columns: string[]
  rows: Record<string, unknown>[]
}

export interface AnalysisResult {
  charts: Record<string, unknown>[]  // Plotly figure dict
  tables: TableData[]
  summary: string
  metrics: Record<string, number | null>
}
