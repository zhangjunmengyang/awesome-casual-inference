// Case Studies 类型定义

export interface DoorDashParams {
  n_orders: number
  method: 'naive' | 'psm' | 'ipw' | 'dr'
}

export interface NetflixParams {
  n_users: number
}

export interface GrowthAttributionParams {
  n_users: number
  channels: string[]
}

// 复用通用分析结果类型
export interface AnalysisResult {
  charts: Record<string, unknown>[]
  tables: Record<string, unknown>[]
  summary: string
  metrics: Record<string, unknown>
}
