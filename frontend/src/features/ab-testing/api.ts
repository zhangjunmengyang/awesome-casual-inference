import { apiClient, ApiResponse } from '@/lib/api/client'
import type {
  ExperimentAnalysisParams,
  SampleSizeParams,
  PowerAnalysisParams,
  ReportGenerationParams,
  AnalysisResult
} from './types'

export const abTestingApi = {
  analyzeExperiment: async (params: ExperimentAnalysisParams) => {
    const response = await apiClient.post<ApiResponse<AnalysisResult>>(
      '/ab-testing/experiment-analysis',
      params
    )
    return response.data.data!
  },

  calculateSampleSize: async (params: SampleSizeParams) => {
    const response = await apiClient.post<ApiResponse<Record<string, unknown>>>(
      '/ab-testing/sample-size',
      params
    )
    return response.data.data!
  },

  analyzePower: async (params: PowerAnalysisParams) => {
    const response = await apiClient.post<ApiResponse<Record<string, unknown>>>(
      '/ab-testing/power-analysis',
      params
    )
    return response.data.data!
  },

  generateReport: async (params: ReportGenerationParams) => {
    const response = await apiClient.post<ApiResponse<Record<string, unknown>>>(
      '/ab-testing/generate-report',
      params
    )
    return response.data.data!
  },
}
