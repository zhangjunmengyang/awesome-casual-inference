import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { useConfoundingBias } from '../hooks'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { Slider } from '@/components/ui/Slider'
import { Button } from '@/components/ui/Button'
import { Loader2 } from 'lucide-react'

export function Component() {
  const [params, setParams] = useState({
    n_samples: 1000,
    confounding_strength: 1.0,
    treatment_effect: 2.0,
  })

  const { mutate, data, isPending, error } = useConfoundingBias()

  const handleAnalyze = () => {
    mutate(params)
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground">混淆偏差</h1>
        <p className="text-muted-foreground mt-1">
          Confounding Bias - 当存在同时影响处理和结果的变量时
        </p>
      </div>

      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">参数设置</h2>
        <div className="grid gap-6 md:grid-cols-3">
          <Slider
            label="样本量"
            min={100}
            max={5000}
            step={100}
            value={[params.n_samples]}
            onValueChange={([v]) => setParams((p) => ({ ...p, n_samples: v }))}
          />
          <Slider
            label="混淆强度"
            min={0}
            max={3}
            step={0.1}
            value={[params.confounding_strength]}
            onValueChange={([v]) => setParams((p) => ({ ...p, confounding_strength: v }))}
          />
          <Slider
            label="真实处理效应"
            min={-5}
            max={5}
            step={0.5}
            value={[params.treatment_effect]}
            onValueChange={([v]) => setParams((p) => ({ ...p, treatment_effect: v }))}
          />
        </div>
        <div className="mt-6">
          <Button onClick={handleAnalyze} disabled={isPending}>
            {isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            运行分析
          </Button>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-destructive bg-destructive/10 p-4 text-destructive">
          {error.message}
        </div>
      )}

      {data && (
        <>
          <div className="rounded-lg border bg-card p-6">
            <h2 className="text-lg font-semibold mb-4">可视化结果</h2>
            {data.charts[0] && (
              <PlotlyChart data={data.charts[0]} className="h-[400px]" />
            )}
          </div>

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
