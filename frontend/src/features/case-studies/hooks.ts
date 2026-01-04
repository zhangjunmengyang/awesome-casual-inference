import { useMutation } from '@tanstack/react-query'
import { caseStudiesApi } from './api'
import type { DoorDashParams, NetflixParams, GrowthAttributionParams } from './types'

export function useDoorDash() {
  return useMutation({
    mutationFn: (params: DoorDashParams) => caseStudiesApi.analyzeDoorDash(params),
  })
}

export function useNetflix() {
  return useMutation({
    mutationFn: (params: NetflixParams) => caseStudiesApi.analyzeNetflix(params),
  })
}

export function useGrowthAttribution() {
  return useMutation({
    mutationFn: (params: GrowthAttributionParams) => caseStudiesApi.analyzeGrowthAttribution(params),
  })
}
