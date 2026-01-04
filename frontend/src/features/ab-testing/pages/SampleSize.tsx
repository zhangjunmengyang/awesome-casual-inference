import { useState } from 'react'
import { useSampleSize } from '../hooks'
import { Slider } from '@/components/ui/Slider'
import { Button } from '@/components/ui/Button'
import { Loader2 } from 'lucide-react'

export function Component() {
  const [params, setParams] = useState({
    baseline: 0.1,
    mde: 0.01,
    alpha: 0.05,
    power: 0.8,
  })

  const { mutate, data, isPending, error } = useSampleSize()

  const handleAnalyze = () => {
    mutate(params)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">样本量计算</h1>
        <p className="text-muted-foreground mt-1">
          Sample Size Calculator - 计算实验所需的最小样本量
        </p>
      </div>

      {/* Introduction */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">核心概念</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <p><strong>MDE (最小可检测效应)</strong></p>
            <p className="text-sm text-muted-foreground">
              实验能够可靠检测到的最小效应大小，
              通常根据业务价值来确定
            </p>
          </div>
          <div className="space-y-2">
            <p><strong>统计功效 (Power)</strong></p>
            <p className="text-sm text-muted-foreground">
              在真实效应存在时正确拒绝零假设的概率，
              通常设置为 80% 或 90%
            </p>
          </div>
        </div>
        <div className="mt-4 p-4 bg-muted rounded-lg">
          <p className="text-sm text-muted-foreground">
            <strong>样本量公式</strong>：对于比率指标，所需样本量与 MDE 的平方成反比。
            如果想检测更小的效应，需要成倍增加样本量。
            这就是为什么正确设置 MDE 对实验规划至关重要。
          </p>
        </div>
      </div>

      {/* Parameters */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">参数设置</h2>
        <div className="grid gap-6 md:grid-cols-2">
          <Slider
            label={`基线转化率: ${(params.baseline * 100).toFixed(1)}%`}
            min={0.01}
            max={0.99}
            step={0.01}
            value={[params.baseline]}
            onValueChange={([v]) => setParams((p) => ({ ...p, baseline: v }))}
          />
          <Slider
            label={`最小可检测效应 (MDE): ${(params.mde * 100).toFixed(2)}%`}
            min={0.001}
            max={0.1}
            step={0.001}
            value={[params.mde]}
            onValueChange={([v]) => setParams((p) => ({ ...p, mde: v }))}
          />
          <Slider
            label={`显著性水平 (α): ${params.alpha}`}
            min={0.01}
            max={0.2}
            step={0.01}
            value={[params.alpha]}
            onValueChange={([v]) => setParams((p) => ({ ...p, alpha: v }))}
          />
          <Slider
            label={`统计功效: ${(params.power * 100).toFixed(0)}%`}
            min={0.5}
            max={0.99}
            step={0.01}
            value={[params.power]}
            onValueChange={([v]) => setParams((p) => ({ ...p, power: v }))}
          />
        </div>
        <div className="mt-6">
          <Button onClick={handleAnalyze} disabled={isPending}>
            {isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            计算样本量
          </Button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="rounded-lg border border-destructive bg-destructive/10 p-4 text-destructive">
          {error.message}
        </div>
      )}

      {/* Results */}
      {data && (
        <>
          {/* Sample Size Result */}
          <div className="rounded-lg border bg-card p-6">
            <h2 className="text-lg font-semibold mb-4">计算结果</h2>
            <div className="grid gap-4 md:grid-cols-3">
              {Object.entries(data).map(([key, value]) => (
                <div key={key} className="p-4 rounded-lg bg-muted">
                  <div className="text-sm text-muted-foreground">{key}</div>
                  <div className="text-2xl font-bold">
                    {typeof value === 'number'
                      ? (Number.isInteger(value) ? value.toLocaleString() : value.toFixed(4))
                      : String(value)}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Interpretation */}
          <div className="rounded-lg border bg-card p-6">
            <h2 className="text-lg font-semibold mb-4">结果解读</h2>
            <div className="prose prose-sm max-w-none dark:prose-invert">
              <p>
                基于您设置的参数：
              </p>
              <ul>
                <li>基线转化率: {(params.baseline * 100).toFixed(1)}%</li>
                <li>最小可检测效应: {(params.mde * 100).toFixed(2)}% (绝对值)</li>
                <li>显著性水平: {params.alpha}</li>
                <li>统计功效: {(params.power * 100).toFixed(0)}%</li>
              </ul>
              <p>
                您需要每组至少 <strong>{typeof data.sample_size_per_group === 'number'
                  ? data.sample_size_per_group.toLocaleString()
                  : 'N/A'}</strong> 个样本，
                总计 <strong>{typeof data.total_sample_size === 'number'
                  ? data.total_sample_size.toLocaleString()
                  : 'N/A'}</strong> 个样本。
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
