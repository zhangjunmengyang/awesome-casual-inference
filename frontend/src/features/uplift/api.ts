import { apiClient, ApiResponse } from '@/lib/api/client'
import type { MetaLearnersParams, UpliftTreeParams, UpliftEvaluationParams, AnalysisResult } from './types'

export const upliftApi = {
  analyzeMetaLearners: async (params: MetaLearnersParams) => {
    const response = await apiClient.post<ApiResponse<AnalysisResult>>(
      '/uplift/meta-learners',
      params
    )
    return response.data.data!
  },

  analyzeUpliftTree: async (params: UpliftTreeParams) => {
    const response = await apiClient.post<ApiResponse<AnalysisResult>>(
      '/uplift/uplift-tree',
      params
    )
    return response.data.data!
  },

  analyzeUpliftEvaluation: async (params: UpliftEvaluationParams) => {
    const response = await apiClient.post<ApiResponse<AnalysisResult>>(
      '/uplift/evaluation',
      params
    )
    return response.data.data!
  },
}
