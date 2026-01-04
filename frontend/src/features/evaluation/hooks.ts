import { useMutation } from '@tanstack/react-query'
import { evaluationApi } from './api'
import type { BalanceCheckParams, OverlapCheckParams, ModelComparisonParams } from './types'

export function useBalanceCheck() {
  return useMutation({
    mutationFn: (params: BalanceCheckParams) => evaluationApi.checkBalance(params),
  })
}

export function useOverlapCheck() {
  return useMutation({
    mutationFn: (params: OverlapCheckParams) => evaluationApi.checkOverlap(params),
  })
}

export function useModelComparison() {
  return useMutation({
    mutationFn: (params: ModelComparisonParams) => evaluationApi.compareModels(params),
  })
}
