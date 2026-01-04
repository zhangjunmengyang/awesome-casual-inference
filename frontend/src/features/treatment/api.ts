import { apiClient, ApiResponse } from '@/lib/api/client'
import type { PSMParams, IPWParams, DoublyRobustParams, AnalysisResult } from './types'

export const treatmentApi = {
  analyzePSM: async (params: PSMParams) => {
    const response = await apiClient.post<ApiResponse<AnalysisResult>>(
      '/treatment/psm',
      params
    )
    return response.data.data!
  },

  analyzeIPW: async (params: IPWParams) => {
    const response = await apiClient.post<ApiResponse<AnalysisResult>>(
      '/treatment/ipw',
      params
    )
    return response.data.data!
  },

  analyzeDoublyRobust: async (params: DoublyRobustParams) => {
    const response = await apiClient.post<ApiResponse<AnalysisResult>>(
      '/treatment/doubly-robust',
      params
    )
    return response.data.data!
  },
}
