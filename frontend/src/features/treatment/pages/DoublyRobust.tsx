import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { useDoublyRobust } from '../hooks'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { Slider } from '@/components/ui/Slider'
import { Button } from '@/components/ui/Button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import { Loader2 } from 'lucide-react'

export function Component() {
  const [params, setParams] = useState({
    n_samples: 1000,
    confounding_strength: 1.0,
    outcome_model: 'linear' as const,
    propensity_model: 'logistic' as const,
  })

  const { mutate, data, isPending, error } = useDoublyRobust()

  const handleAnalyze = () => {
    mutate(params)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">双重稳健估计 (DR)</h1>
        <p className="text-muted-foreground mt-1">
          Doubly Robust Estimation - 结合 IPW 和结果回归的稳健方法
        </p>
      </div>

      {/* Introduction */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">核心概念</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <p><strong>双重稳健性</strong>: 只要倾向得分或结果模型之一正确，估计就是一致的</p>
            <p><strong>AIPW</strong>: 增广逆概率加权估计量</p>
          </div>
          <div className="space-y-2">
            <p><strong>结果模型</strong>: 预测潜在结果 E[Y|X,T]</p>
            <p><strong>倾向得分模型</strong>: 预测处理概率 P(T=1|X)</p>
          </div>
        </div>
        <div className="mt-4 p-4 bg-muted rounded-lg">
          <p className="text-sm text-muted-foreground">
            <strong>核心思想</strong>: DR 估计量结合了 IPW 和结果回归的优点，
            即使其中一个模型被错误指定，只要另一个正确，估计仍然是一致的。
            这提供了额外的保护层，使得估计更加稳健。
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
          <div className="space-y-2">
            <label className="text-sm font-medium">结果模型</label>
            <Select
              value={params.outcome_model}
              onValueChange={(v) => setParams((p) => ({ ...p, outcome_model: v as 'linear' | 'rf' | 'xgb' }))}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="linear">线性回归 (Linear)</SelectItem>
                <SelectItem value="rf">随机森林 (Random Forest)</SelectItem>
                <SelectItem value="xgb">XGBoost</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium">倾向得分模型</label>
            <Select
              value={params.propensity_model}
              onValueChange={(v) => setParams((p) => ({ ...p, propensity_model: v as 'logistic' | 'rf' | 'xgb' }))}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="logistic">逻辑回归 (Logistic)</SelectItem>
                <SelectItem value="rf">随机森林 (Random Forest)</SelectItem>
                <SelectItem value="xgb">XGBoost</SelectItem>
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
