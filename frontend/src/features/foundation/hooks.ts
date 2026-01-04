import { useMutation } from '@tanstack/react-query'
import { foundationApi } from './api'
import type {
  PotentialOutcomesParams,
  CausalDAGParams,
  ConfoundingBiasParams,
  SelectionBiasParams,
} from './types'

export function usePotentialOutcomes() {
  return useMutation({
    mutationFn: (params: PotentialOutcomesParams) =>
      foundationApi.analyzePotentialOutcomes(params),
  })
}

export function useCausalDAG() {
  return useMutation({
    mutationFn: (params: CausalDAGParams) =>
      foundationApi.analyzeCausalDAG(params),
  })
}

export function useConfoundingBias() {
  return useMutation({
    mutationFn: (params: ConfoundingBiasParams) =>
      foundationApi.analyzeConfoundingBias(params),
  })
}

export function useSelectionBias() {
  return useMutation({
    mutationFn: (params: SelectionBiasParams) =>
      foundationApi.analyzeSelectionBias(params),
  })
}
