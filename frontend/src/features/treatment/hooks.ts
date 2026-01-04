import { useMutation } from '@tanstack/react-query'
import { treatmentApi } from './api'
import type { PSMParams, IPWParams, DoublyRobustParams } from './types'

export function usePSM() {
  return useMutation({
    mutationFn: (params: PSMParams) => treatmentApi.analyzePSM(params),
  })
}

export function useIPW() {
  return useMutation({
    mutationFn: (params: IPWParams) => treatmentApi.analyzeIPW(params),
  })
}

export function useDoublyRobust() {
  return useMutation({
    mutationFn: (params: DoublyRobustParams) => treatmentApi.analyzeDoublyRobust(params),
  })
}
