import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { useUpliftEvaluation } from '../hooks'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { Slider } from '@/components/ui/Slider'
import { Button } from '@/components/ui/Button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import { Loader2 } from 'lucide-react'

export function Component() {
  const [params, setParams] = useState({
    n_samples: 10000,
    model_quality: 'good' as const,
  })

  const { mutate, data, isPending, error } = useUpliftEvaluation()

  const handleAnalyze = () => {
    mutate(params)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Uplift 模型评估</h1>
        <p className="text-muted-foreground mt-1">
          Qini 曲线、AUUC - 评估增益模型的性能
        </p>
      </div>

      {/* Introduction */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">核心概念</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2 p-4 bg-muted rounded-lg">
            <p className="font-semibold">Qini 曲线</p>
            <p className="text-sm text-muted-foreground">
              类似 ROC 曲线，展示按 uplift 排序后的累积增益。
              X 轴为目标人群比例，Y 轴为累积处理效应。
            </p>
          </div>
          <div className="space-y-2 p-4 bg-muted rounded-lg">
            <p className="font-semibold">AUUC</p>
            <p className="text-sm text-muted-foreground">
              Area Under Uplift Curve：Qini 曲线下面积，
              衡量模型区分高低 uplift 用户的能力。
            </p>
          </div>
        </div>
        <div className="mt-4 p-4 bg-muted rounded-lg">
          <p className="text-sm text-muted-foreground">
            <strong>为什么需要专门的评估指标？</strong>
            传统的 AUC、MSE 等指标无法直接评估 uplift 模型，
            因为我们永远无法同时观测到同一用户的处理和非处理结果。
            Qini 曲线通过累积增益的方式，巧妙地解决了这个问题。
          </p>
        </div>
      </div>

      {/* Parameters */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">参数设置</h2>
        <div className="grid gap-6 md:grid-cols-2">
          <Slider
            label="样本量"
            min={1000}
            max={50000}
            step={1000}
            value={[params.n_samples]}
            onValueChange={([v]) => setParams((p) => ({ ...p, n_samples: v }))}
          />
          <div className="space-y-2">
            <label className="text-sm font-medium">模型质量</label>
            <Select
              value={params.model_quality}
              onValueChange={(v) => setParams((p) => ({ ...p, model_quality: v as 'perfect' | 'good' | 'medium' | 'poor' }))}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="perfect">完美模型</SelectItem>
                <SelectItem value="good">优秀模型</SelectItem>
                <SelectItem value="medium">中等模型</SelectItem>
                <SelectItem value="poor">较差模型</SelectItem>
              </SelectContent>
            </Select>
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
