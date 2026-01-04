import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { usePotentialOutcomes } from '../hooks'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { Slider } from '@/components/ui/Slider'
import { Button } from '@/components/ui/Button'
import { Loader2 } from 'lucide-react'

export function Component() {
  const [params, setParams] = useState({
    n_samples: 500,
    treatment_effect: 2.0,
    noise_std: 1.0,
    confounding_strength: 0.0,
  })

  const { mutate, data, isPending, error } = usePotentialOutcomes()

  const handleAnalyze = () => {
    mutate(params)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">潜在结果框架</h1>
        <p className="text-muted-foreground mt-1">
          Potential Outcomes Framework - Rubin Causal Model
        </p>
      </div>

      {/* Introduction */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">核心概念</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <p><strong>Y(0)</strong>: 不接受处理时的潜在结果</p>
            <p><strong>Y(1)</strong>: 接受处理时的潜在结果</p>
          </div>
          <div className="space-y-2">
            <p><strong>ITE = Y(1) - Y(0)</strong>: 个体处理效应</p>
            <p><strong>ATE = E[Y(1) - Y(0)]</strong>: 平均处理效应</p>
          </div>
        </div>
        <div className="mt-4 p-4 bg-muted rounded-lg">
          <p className="text-sm text-muted-foreground">
            <strong>基本问题 (Fundamental Problem)</strong>: 每个个体在同一时刻只能处于一种状态，
            因此我们永远无法同时观测到同一个体的 Y(0) 和 Y(1)。
          </p>
        </div>
      </div>

      {/* Parameters */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">参数设置</h2>
        <div className="grid gap-6 md:grid-cols-2">
          <Slider
            label="样本量"
            min={100}
            max={2000}
            step={100}
            value={[params.n_samples]}
            onValueChange={([v]) => setParams((p) => ({ ...p, n_samples: v }))}
          />
          <Slider
            label="真实处理效应 (ATE)"
            min={-5}
            max={5}
            step={0.5}
            value={[params.treatment_effect]}
            onValueChange={([v]) => setParams((p) => ({ ...p, treatment_effect: v }))}
          />
          <Slider
            label="噪声标准差"
            min={0.1}
            max={3}
            step={0.1}
            value={[params.noise_std]}
            onValueChange={([v]) => setParams((p) => ({ ...p, noise_std: v }))}
          />
          <Slider
            label="混淆强度"
            min={0}
            max={2}
            step={0.1}
            value={[params.confounding_strength]}
            onValueChange={([v]) => setParams((p) => ({ ...p, confounding_strength: v }))}
          />
        </div>
        <div className="mt-6">
          <Button onClick={handleAnalyze} disabled={isPending}>
            {isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            运行分析
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
          {/* Chart */}
          <div className="rounded-lg border bg-card p-6">
            <h2 className="text-lg font-semibold mb-4">可视化结果</h2>
            {data.charts[0] && (
              <PlotlyChart data={data.charts[0]} className="h-[600px]" />
            )}
          </div>

          {/* Metrics */}
          <div className="rounded-lg border bg-card p-6">
            <h2 className="text-lg font-semibold mb-4">关键指标</h2>
            <div className="grid gap-4 md:grid-cols-4">
              {Object.entries(data.metrics).map(([key, value]) => (
                <div key={key} className="p-4 rounded-lg bg-muted">
                  <div className="text-sm text-muted-foreground">{key}</div>
                  <div className="text-2xl font-bold">
                    {typeof value === 'number' ? value.toFixed(4) : 'N/A'}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Summary */}
          <div className="rounded-lg border bg-card p-6">
            <div className="prose prose-sm max-w-none dark:prose-invert">
              <ReactMarkdown>{data.summary}</ReactMarkdown>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
