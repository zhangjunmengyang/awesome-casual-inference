import { apiClient, ApiResponse } from '@/lib/api/client'
import type { DoorDashParams, NetflixParams, GrowthAttributionParams, AnalysisResult } from './types'

export const caseStudiesApi = {
  analyzeDoorDash: async (params: DoorDashParams) => {
    const response = await apiClient.post<ApiResponse<AnalysisResult>>(
      '/cases/doordash',
      params
    )
    return response.data.data!
  },

  analyzeNetflix: async (params: NetflixParams) => {
    const response = await apiClient.post<ApiResponse<AnalysisResult>>(
      '/cases/netflix',
      params
    )
    return response.data.data!
  },

  analyzeGrowthAttribution: async (params: GrowthAttributionParams) => {
    const response = await apiClient.post<ApiResponse<AnalysisResult>>(
      '/cases/growth-attribution',
      params
    )
    return response.data.data!
  },
}
