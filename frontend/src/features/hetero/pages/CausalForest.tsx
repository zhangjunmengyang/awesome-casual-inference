import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { useCausalForest } from '../hooks'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { Slider } from '@/components/ui/Slider'
import { Button } from '@/components/ui/Button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import { Loader2 } from 'lucide-react'

export function Component() {
  const [params, setParams] = useState({
    n_samples: 2000,
    effect_heterogeneity: 'moderate' as const,
    confounding_strength: 0.5,
    n_trees: 100,
  })

  const { mutate, data, isPending, error } = useCausalForest()

  const handleAnalyze = () => {
    mutate(params)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">因果森林 (Causal Forest)</h1>
        <p className="text-muted-foreground mt-1">
          基于随机森林的异质处理效应估计 - 来自 EconML/GRF
        </p>
      </div>

      {/* Introduction */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">核心概念</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <p><strong>Generalized Random Forest (GRF)</strong></p>
            <p className="text-sm text-muted-foreground">
              使用局部矩条件估计 CATE，通过森林自适应地学习特征空间的分区
            </p>
          </div>
          <div className="space-y-2">
            <p><strong>Honest Estimation</strong></p>
            <p className="text-sm text-muted-foreground">
              使用独立样本进行树分裂和效应估计，确保置信区间的有效性
            </p>
          </div>
        </div>
        <div className="mt-4 p-4 bg-muted rounded-lg">
          <p className="text-sm text-muted-foreground">
            <strong>因果森林</strong> 是估计异质处理效应 (HTE) 的强大非参数方法。
            它扩展了随机森林，使用因果分裂准则来最大化子节点间的处理效应差异，
            同时提供有效的置信区间估计。
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
            label="混淆强度"
            min={0}
            max={1}
            step={0.1}
            value={[params.confounding_strength]}
            onValueChange={([v]) => setParams((p) => ({ ...p, confounding_strength: v }))}
          />
          <Slider
            label="树的数量"
            min={10}
            max={500}
            step={10}
            value={[params.n_trees]}
            onValueChange={([v]) => setParams((p) => ({ ...p, n_trees: v }))}
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
