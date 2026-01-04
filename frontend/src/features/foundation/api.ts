import { apiClient, ApiResponse } from '@/lib/api/client'
import type {
  PotentialOutcomesParams,
  CausalDAGParams,
  ConfoundingBiasParams,
  SelectionBiasParams,
  AnalysisResult,
} from './types'

export const foundationApi = {
  analyzePotentialOutcomes: async (params: PotentialOutcomesParams) => {
    const response = await apiClient.post<ApiResponse<AnalysisResult>>(
      '/foundation/potential-outcomes',
      params
    )
    return response.data.data!
  },

  analyzeCausalDAG: async (params: CausalDAGParams) => {
    const response = await apiClient.post<ApiResponse<AnalysisResult>>(
      '/foundation/causal-dag',
      params
    )
    return response.data.data!
  },

  analyzeConfoundingBias: async (params: ConfoundingBiasParams) => {
    const response = await apiClient.post<ApiResponse<AnalysisResult>>(
      '/foundation/confounding-bias',
      params
    )
    return response.data.data!
  },

  analyzeSelectionBias: async (params: SelectionBiasParams) => {
    const response = await apiClient.post<ApiResponse<AnalysisResult>>(
      '/foundation/selection-bias',
      params
    )
    return response.data.data!
  },
}
