import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { useUpliftTree } from '../hooks'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { Slider } from '@/components/ui/Slider'
import { Button } from '@/components/ui/Button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import { Loader2 } from 'lucide-react'

export function Component() {
  const [params, setParams] = useState({
    n_samples: 2000,
    feature_effect: 0.15,
    criterion: 'KL' as const,
  })

  const { mutate, data, isPending, error } = useUpliftTree()

  const handleAnalyze = () => {
    mutate(params)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Uplift Tree</h1>
        <p className="text-muted-foreground mt-1">
          基于因果效应分裂的决策树 - 用于用户分群和精准营销
        </p>
      </div>

      {/* Introduction */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">核心概念</h2>
        <div className="grid gap-4 md:grid-cols-3">
          <div className="space-y-2 p-4 bg-muted rounded-lg">
            <p className="font-semibold">KL 散度</p>
            <p className="text-sm text-muted-foreground">
              Kullback-Leibler 散度：衡量处理组和对照组分布差异
            </p>
          </div>
          <div className="space-y-2 p-4 bg-muted rounded-lg">
            <p className="font-semibold">ED (欧氏距离)</p>
            <p className="text-sm text-muted-foreground">
              Euclidean Distance：直接计算处理效应的差异
            </p>
          </div>
          <div className="space-y-2 p-4 bg-muted rounded-lg">
            <p className="font-semibold">Chi-squared</p>
            <p className="text-sm text-muted-foreground">
              卡方统计量：检验处理效应在子节点间的差异显著性
            </p>
          </div>
        </div>
        <div className="mt-4 p-4 bg-muted rounded-lg">
          <p className="text-sm text-muted-foreground">
            <strong>Uplift Tree</strong> 与传统决策树的关键区别：
            分裂准则不是预测精度，而是最大化子节点间的处理效应差异。
            这使得树能够识别出对营销活动反应不同的用户群体。
          </p>
        </div>
      </div>

      {/* Parameters */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">参数设置</h2>
        <div className="grid gap-6 md:grid-cols-3">
          <Slider
            label="样本量"
            min={500}
            max={10000}
            step={500}
            value={[params.n_samples]}
            onValueChange={([v]) => setParams((p) => ({ ...p, n_samples: v }))}
          />
          <Slider
            label="特征效应强度"
            min={0}
            max={1}
            step={0.05}
            value={[params.feature_effect]}
            onValueChange={([v]) => setParams((p) => ({ ...p, feature_effect: v }))}
          />
          <div className="space-y-2">
            <label className="text-sm font-medium">分裂准则</label>
            <Select
              value={params.criterion}
              onValueChange={(v) => setParams((p) => ({ ...p, criterion: v as 'KL' | 'ED' | 'Chi' }))}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="KL">KL 散度</SelectItem>
                <SelectItem value="ED">欧氏距离</SelectItem>
                <SelectItem value="Chi">卡方统计量</SelectItem>
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
