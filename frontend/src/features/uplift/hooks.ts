import { useMutation } from '@tanstack/react-query'
import { upliftApi } from './api'
import type { MetaLearnersParams, UpliftTreeParams, UpliftEvaluationParams } from './types'

export function useMetaLearners() {
  return useMutation({
    mutationFn: (params: MetaLearnersParams) => upliftApi.analyzeMetaLearners(params),
  })
}

export function useUpliftTree() {
  return useMutation({
    mutationFn: (params: UpliftTreeParams) => upliftApi.analyzeUpliftTree(params),
  })
}

export function useUpliftEvaluation() {
  return useMutation({
    mutationFn: (params: UpliftEvaluationParams) => upliftApi.analyzeUpliftEvaluation(params),
  })
}
