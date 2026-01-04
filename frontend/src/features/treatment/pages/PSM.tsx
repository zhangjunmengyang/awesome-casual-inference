import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { usePSM } from '../hooks'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { Slider } from '@/components/ui/Slider'
import { Button } from '@/components/ui/Button'
import { Loader2 } from 'lucide-react'

export function Component() {
  const [params, setParams] = useState({
    n_samples: 1000,
    confounding_strength: 1.0,
    caliper: 0.2,
  })

  const { mutate, data, isPending, error } = usePSM()

  const handleAnalyze = () => {
    mutate(params)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">倾向得分匹配 (PSM)</h1>
        <p className="text-muted-foreground mt-1">
          Propensity Score Matching - 通过匹配相似个体来估计处理效应
        </p>
      </div>

      {/* Introduction */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">核心概念</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <p><strong>倾向得分 e(X)</strong>: 给定协变量 X 时接受处理的概率</p>
            <p><strong>匹配</strong>: 为每个处理组个体找到倾向得分相似的对照组个体</p>
          </div>
          <div className="space-y-2">
            <p><strong>卡尺 (Caliper)</strong>: 允许匹配的最大倾向得分差异</p>
            <p><strong>ATT</strong>: 处理组的平均处理效应</p>
          </div>
        </div>
        <div className="mt-4 p-4 bg-muted rounded-lg">
          <p className="text-sm text-muted-foreground">
            <strong>核心思想</strong>: 倾向得分是一个平衡得分 (Balancing Score)，
            在给定倾向得分后，处理分配与协变量条件独立，从而可以消除混淆偏差。
          </p>
        </div>
      </div>

      {/* Parameters */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">参数设置</h2>
        <div className="grid gap-6 md:grid-cols-3">
          <Slider
            label="样本量"
            min={200}
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
            label="卡尺宽度"
            min={0.01}
            max={1}
            step={0.01}
            value={[params.caliper]}
            onValueChange={([v]) => setParams((p) => ({ ...p, caliper: v }))}
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
