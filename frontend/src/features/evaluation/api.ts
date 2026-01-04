import { apiClient, ApiResponse } from '@/lib/api/client'
import type { BalanceCheckParams, OverlapCheckParams, ModelComparisonParams, AnalysisResult } from './types'

export const evaluationApi = {
  checkBalance: async (params: BalanceCheckParams) => {
    const response = await apiClient.post<ApiResponse<AnalysisResult>>(
      '/evaluation/balance-check',
      params
    )
    return response.data.data!
  },

  checkOverlap: async (params: OverlapCheckParams) => {
    const response = await apiClient.post<ApiResponse<AnalysisResult>>(
      '/evaluation/overlap-check',
      params
    )
    return response.data.data!
  },

  compareModels: async (params: ModelComparisonParams) => {
    const response = await apiClient.post<ApiResponse<AnalysisResult>>(
      '/evaluation/model-comparison',
      params
    )
    return response.data.data!
  },
}
