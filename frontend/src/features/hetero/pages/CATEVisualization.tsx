import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { useCATEVisualization } from '../hooks'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { Slider } from '@/components/ui/Slider'
import { Button } from '@/components/ui/Button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import { Loader2 } from 'lucide-react'

export function Component() {
  const [params, setParams] = useState({
    n_samples: 2000,
    effect_heterogeneity: 'moderate' as const,
    n_bootstrap: 100,
    n_subgroups: 4,
  })

  const { mutate, data, isPending, error } = useCATEVisualization()

  const handleAnalyze = () => {
    mutate(params)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">CATE 可视化</h1>
        <p className="text-muted-foreground mt-1">
          条件平均处理效应的可视化分析与子群体发现
        </p>
      </div>

      {/* Introduction */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">核心概念</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <p><strong>CATE 分布</strong></p>
            <p className="text-sm text-muted-foreground">
              展示处理效应在样本中的分布，识别高收益和低收益群体
            </p>
          </div>
          <div className="space-y-2">
            <p><strong>子群体分析</strong></p>
            <p className="text-sm text-muted-foreground">
              按 CATE 估计值将样本分成多个子群体，分析各组特征
            </p>
          </div>
        </div>
        <div className="mt-4 p-4 bg-muted rounded-lg">
          <p className="text-sm text-muted-foreground">
            <strong>可视化 CATE</strong> 有助于理解处理效应的异质性模式，
            发现哪些特征与高处理效应相关，为精准干预提供依据。
            Bootstrap 置信区间帮助评估估计的不确定性。
          </p>
        </div>
      </div>

      {/* Parameters */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">参数设置</h2>
        <div className="grid gap-6 md:grid-cols-2">
          <Slider
            label="样本量"
            min={500}
            max={10000}
            step={500}
            value={[params.n_samples]}
            onValueChange={([v]) => setParams((p) => ({ ...p, n_samples: v }))}
          />
          <div className="space-y-2">
            <label className="text-sm font-medium">效应异质性强度</label>
            <Select
              value={params.effect_heterogeneity}
              onValueChange={(v) => setParams((p) => ({ ...p, effect_heterogeneity: v as 'weak' | 'moderate' | 'strong' }))}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="weak">弱异质性</SelectItem>
                <SelectItem value="moderate">中等异质性</SelectItem>
                <SelectItem value="strong">强异质性</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <Slider
            label="Bootstrap 次数"
            min={10}
            max={500}
            step={10}
            value={[params.n_bootstrap]}
            onValueChange={([v]) => setParams((p) => ({ ...p, n_bootstrap: v }))}
          />
          <Slider
            label="子群体数量"
            min={2}
            max={10}
            step={1}
            value={[params.n_subgroups]}
            onValueChange={([v]) => setParams((p) => ({ ...p, n_subgroups: v }))}
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
