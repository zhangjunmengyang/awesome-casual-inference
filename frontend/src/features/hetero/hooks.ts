import { useMutation } from '@tanstack/react-query'
import { heteroApi } from './api'
import type { CausalForestParams, CATEVisualizationParams, SensitivityParams } from './types'

export function useCausalForest() {
  return useMutation({
    mutationFn: (params: CausalForestParams) => heteroApi.analyzeCausalForest(params),
  })
}

export function useCATEVisualization() {
  return useMutation({
    mutationFn: (params: CATEVisualizationParams) => heteroApi.visualizeCate(params),
  })
}

export function useSensitivity() {
  return useMutation({
    mutationFn: (params: SensitivityParams) => heteroApi.analyzeSensitivity(params),
  })
}
