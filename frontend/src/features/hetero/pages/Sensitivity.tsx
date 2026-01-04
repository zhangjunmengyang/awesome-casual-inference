import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { useSensitivity } from '../hooks'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { Slider } from '@/components/ui/Slider'
import { Button } from '@/components/ui/Button'
import { Loader2 } from 'lucide-react'

export function Component() {
  const [params, setParams] = useState({
    n_samples: 1000,
    confounder_strength: 0.5,
    correlation_with_x: 0.3,
    max_gamma: 3.0,
  })

  const { mutate, data, isPending, error } = useSensitivity()

  const handleAnalyze = () => {
    mutate(params)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">敏感性分析</h1>
        <p className="text-muted-foreground mt-1">
          Sensitivity Analysis - 评估隐藏混淆对因果估计的影响
        </p>
      </div>

      {/* Introduction */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">核心概念</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <p><strong>Rosenbaum Bounds</strong></p>
            <p className="text-sm text-muted-foreground">
              评估需要多强的隐藏偏差才能推翻因果结论，
              Gamma 值越大，结论越稳健
            </p>
          </div>
          <div className="space-y-2">
            <p><strong>E-value</strong></p>
            <p className="text-sm text-muted-foreground">
              未观测混淆变量需要与处理和结果有多强的关联
              才能完全解释观测到的效应
            </p>
          </div>
        </div>
        <div className="mt-4 p-4 bg-muted rounded-lg">
          <p className="text-sm text-muted-foreground">
            <strong>敏感性分析</strong> 回答这个关键问题：
            "我们的因果结论对未观测混淆有多敏感？"
            即使我们无法直接检验因果假设，敏感性分析也能告诉我们
            需要多严重的违反才会改变结论。
          </p>
        </div>
      </div>

      {/* Parameters */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">参数设置</h2>
        <div className="grid gap-6 md:grid-cols-2">
          <Slider
            label="样本量"
            min={200}
            max={5000}
            step={100}
            value={[params.n_samples]}
            onValueChange={([v]) => setParams((p) => ({ ...p, n_samples: v }))}
          />
          <Slider
            label="未观测混淆强度"
            min={0}
            max={1}
            step={0.1}
            value={[params.confounder_strength]}
            onValueChange={([v]) => setParams((p) => ({ ...p, confounder_strength: v }))}
          />
          <Slider
            label="U 与 X 的相关性"
            min={0}
            max={1}
            step={0.1}
            value={[params.correlation_with_x]}
            onValueChange={([v]) => setParams((p) => ({ ...p, correlation_with_x: v }))}
          />
          <Slider
            label="最大 Gamma 值"
            min={1}
            max={10}
            step={0.5}
            value={[params.max_gamma]}
            onValueChange={([v]) => setParams((p) => ({ ...p, max_gamma: v }))}
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
          {/* Charts */}
          <div className="rounded-lg border bg-card p-6">
            <h2 className="text-lg font-semibold mb-4">可视化结果</h2>
            <div className="grid gap-4">
              {data.charts.map((chart, idx) => (
                <PlotlyChart key={idx} data={chart} className="h-[500px]" />
              ))}
            </div>
          </div>

          {/* Metrics */}
          <div className="rounded-lg border bg-card p-6">
            <h2 className="text-lg font-semibold mb-4">关键指标</h2>
            <div className="grid gap-4 md:grid-cols-4">
              {Object.entries(data.metrics).map(([key, value]) => (
                <div key={key} className="p-4 rounded-lg bg-muted">
                  <div className="text-sm text-muted-foreground">{key}</div>
                  <div className="text-2xl font-bold">
                    {typeof value === 'number' ? value.toFixed(4) : String(value)}
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
