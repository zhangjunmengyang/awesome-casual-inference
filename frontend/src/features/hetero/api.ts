import { apiClient, ApiResponse } from '@/lib/api/client'
import type { CausalForestParams, CATEVisualizationParams, SensitivityParams, AnalysisResult } from './types'

export const heteroApi = {
  analyzeCausalForest: async (params: CausalForestParams) => {
    const response = await apiClient.post<ApiResponse<AnalysisResult>>(
      '/hetero/causal-forest',
      params
    )
    return response.data.data!
  },

  visualizeCate: async (params: CATEVisualizationParams) => {
    const response = await apiClient.post<ApiResponse<AnalysisResult>>(
      '/hetero/cate-visualization',
      params
    )
    return response.data.data!
  },

  analyzeSensitivity: async (params: SensitivityParams) => {
    const response = await apiClient.post<ApiResponse<AnalysisResult>>(
      '/hetero/sensitivity',
      params
    )
    return response.data.data!
  },
}
