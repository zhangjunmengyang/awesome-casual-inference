import { createBrowserRouter } from 'react-router-dom'
import { MainLayout } from '@/components/layout/MainLayout'

export const router = createBrowserRouter([
  {
    path: '/',
    element: <MainLayout />,
    children: [
      {
        index: true,
        lazy: () => import('@/pages/Home'),
      },
      // Foundation Lab
      {
        path: 'foundation',
        children: [
          {
            path: 'potential-outcomes',
            lazy: () => import('@/features/foundation/pages/PotentialOutcomes'),
          },
          {
            path: 'causal-dag',
            lazy: () => import('@/features/foundation/pages/CausalDAG'),
          },
          {
            path: 'confounding-bias',
            lazy: () => import('@/features/foundation/pages/ConfoundingBias'),
          },
          {
            path: 'selection-bias',
            lazy: () => import('@/features/foundation/pages/SelectionBias'),
          },
        ],
      },
      // Treatment Effect Lab
      {
        path: 'treatment',
        children: [
          {
            path: 'psm',
            lazy: () => import('@/features/treatment/pages/PSM'),
          },
          {
            path: 'ipw',
            lazy: () => import('@/features/treatment/pages/IPW'),
          },
          {
            path: 'doubly-robust',
            lazy: () => import('@/features/treatment/pages/DoublyRobust'),
          },
        ],
      },
      // Uplift Lab
      {
        path: 'uplift',
        children: [
          {
            path: 'meta-learners',
            lazy: () => import('@/features/uplift/pages/MetaLearners'),
          },
          {
            path: 'uplift-tree',
            lazy: () => import('@/features/uplift/pages/UpliftTree'),
          },
          {
            path: 'evaluation',
            lazy: () => import('@/features/uplift/pages/Evaluation'),
          },
        ],
      },
      // Heterogeneous Effect Lab
      {
        path: 'hetero',
        children: [
          {
            path: 'causal-forest',
            lazy: () => import('@/features/hetero/pages/CausalForest'),
          },
          {
            path: 'cate-visualization',
            lazy: () => import('@/features/hetero/pages/CATEVisualization'),
          },
          {
            path: 'sensitivity',
            lazy: () => import('@/features/hetero/pages/Sensitivity'),
          },
        ],
      },
      // Evaluation Lab
      {
        path: 'evaluation',
        children: [
          {
            path: 'balance-check',
            lazy: () => import('@/features/evaluation/pages/BalanceCheck'),
          },
        ],
      },
      // Case Studies
      {
        path: 'cases',
        children: [
          {
            path: 'doordash',
            lazy: () => import('@/features/case-studies/pages/DoorDash'),
          },
          {
            path: 'netflix',
            lazy: () => import('@/features/case-studies/pages/Netflix'),
          },
          {
            path: 'growth-attribution',
            lazy: () => import('@/features/case-studies/pages/GrowthAttribution'),
          },
        ],
      },
      // A/B Testing
      {
        path: 'ab-testing',
        children: [
          {
            path: 'experiment-analysis',
            lazy: () => import('@/features/ab-testing/pages/ExperimentAnalysis'),
          },
          {
            path: 'sample-size',
            lazy: () => import('@/features/ab-testing/pages/SampleSize'),
          },
        ],
      },
    ],
  },
])
