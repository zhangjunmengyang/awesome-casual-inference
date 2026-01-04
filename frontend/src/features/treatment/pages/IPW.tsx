import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { useIPW } from '../hooks'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { Slider } from '@/components/ui/Slider'
import { Button } from '@/components/ui/Button'
import { Switch } from '@/components/ui/Switch'
import { Loader2 } from 'lucide-react'

export function Component() {
  const [params, setParams] = useState({
    n_samples: 1000,
    confounding_strength: 1.0,
    stabilized: true,
    trimming: 0.01,
  })

  const { mutate, data, isPending, error } = useIPW()

  const handleAnalyze = () => {
    mutate(params)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">逆概率加权 (IPW)</h1>
        <p className="text-muted-foreground mt-1">
          Inverse Probability Weighting - 通过加权创建伪随机实验
        </p>
      </div>

      {/* Introduction */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">核心概念</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <p><strong>权重 w(X)</strong>: 处理组 1/e(X)，对照组 1/(1-e(X))</p>
            <p><strong>稳定权重</strong>: 使用边际概率归一化，减少方差</p>
          </div>
          <div className="space-y-2">
            <p><strong>截断 (Trimming)</strong>: 限制极端权重，提高稳定性</p>
            <p><strong>ATE</strong>: 加权平均的处理效应估计</p>
          </div>
        </div>
        <div className="mt-4 p-4 bg-muted rounded-lg">
          <p className="text-sm text-muted-foreground">
            <strong>核心思想</strong>: 通过对每个观测值赋予逆概率权重，
            IPW 创建了一个伪总体，在其中处理分配与协变量独立，从而可以无偏估计因果效应。
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
            label="混淆强度"
            min={0}
            max={3}
            step={0.1}
            value={[params.confounding_strength]}
            onValueChange={([v]) => setParams((p) => ({ ...p, confounding_strength: v }))}
          />
          <Slider
            label="权重截断阈值"
            min={0}
            max={0.1}
            step={0.01}
            value={[params.trimming]}
            onValueChange={([v]) => setParams((p) => ({ ...p, trimming: v }))}
          />
          <div className="flex items-center space-x-3">
            <Switch
              checked={params.stabilized}
              onCheckedChange={(v) => setParams((p) => ({ ...p, stabilized: v }))}
            />
            <span className="text-sm">使用稳定权重</span>
          </div>
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
