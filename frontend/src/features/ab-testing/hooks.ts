import { useMutation } from '@tanstack/react-query'
import { abTestingApi } from './api'
import type {
  ExperimentAnalysisParams,
  SampleSizeParams,
  PowerAnalysisParams,
  ReportGenerationParams
} from './types'

export function useExperimentAnalysis() {
  return useMutation({
    mutationFn: (params: ExperimentAnalysisParams) => abTestingApi.analyzeExperiment(params),
  })
}

export function useSampleSize() {
  return useMutation({
    mutationFn: (params: SampleSizeParams) => abTestingApi.calculateSampleSize(params),
  })
}

export function usePowerAnalysis() {
  return useMutation({
    mutationFn: (params: PowerAnalysisParams) => abTestingApi.analyzePower(params),
  })
}

export function useReportGeneration() {
  return useMutation({
    mutationFn: (params: ReportGenerationParams) => abTestingApi.generateReport(params),
  })
}
