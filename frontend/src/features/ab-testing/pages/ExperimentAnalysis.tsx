import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { useExperimentAnalysis } from '../hooks'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { Slider } from '@/components/ui/Slider'
import { Button } from '@/components/ui/Button'
import { Switch } from '@/components/ui/Switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import { Loader2 } from 'lucide-react'

export function Component() {
  const [params, setParams] = useState({
    n_samples: 10000,
    true_effect: 0.02,
    metric_type: 'proportion' as const,
    use_cuped: false,
  })

  const { mutate, data, isPending, error } = useExperimentAnalysis()

  const handleAnalyze = () => {
    mutate(params)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">A/B 实验分析</h1>
        <p className="text-muted-foreground mt-1">
          分析实验结果，支持 CUPED 方差缩减
        </p>
      </div>

      {/* Introduction */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">核心概念</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <p><strong>统计显著性</strong></p>
            <p className="text-sm text-muted-foreground">
              使用 t 检验或 z 检验评估处理效应是否显著不同于零
            </p>
          </div>
          <div className="space-y-2">
            <p><strong>CUPED</strong></p>
            <p className="text-sm text-muted-foreground">
              Controlled-experiment Using Pre-Experiment Data，
              利用实验前数据缩减方差，提高检测灵敏度
            </p>
          </div>
        </div>
        <div className="mt-4 p-4 bg-muted rounded-lg">
          <p className="text-sm text-muted-foreground">
            <strong>CUPED 原理</strong>：通过回归调整，利用与结果相关的协变量
            (如实验前的用户行为) 来减少结果的方差，从而在相同样本量下获得更精确的估计。
            方差缩减可达 20-50%，等效于增加 25-100% 的样本量。
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
            max={100000}
            step={1000}
            value={[params.n_samples]}
            onValueChange={([v]) => setParams((p) => ({ ...p, n_samples: v }))}
          />
          <Slider
            label="真实效应"
            min={-0.2}
            max={0.2}
            step={0.01}
            value={[params.true_effect]}
            onValueChange={([v]) => setParams((p) => ({ ...p, true_effect: v }))}
          />
          <div className="space-y-2">
            <label className="text-sm font-medium">指标类型</label>
            <Select
              value={params.metric_type}
              onValueChange={(v) => setParams((p) => ({ ...p, metric_type: v as 'proportion' | 'continuous' }))}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="proportion">比率指标 (如转化率)</SelectItem>
                <SelectItem value="continuous">连续指标 (如收入)</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="flex items-center space-x-3">
            <Switch
              checked={params.use_cuped}
              onCheckedChange={(v) => setParams((p) => ({ ...p, use_cuped: v }))}
            />
            <span className="text-sm">使用 CUPED 方差缩减</span>
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
