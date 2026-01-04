import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { useDoorDash } from '../hooks'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { Slider } from '@/components/ui/Slider'
import { Button } from '@/components/ui/Button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import { Loader2 } from 'lucide-react'

export function Component() {
  const [params, setParams] = useState({
    n_orders: 5000,
    method: 'dr' as const,
  })

  const { mutate, data, isPending, error } = useDoorDash()

  const handleAnalyze = () => {
    mutate(params)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">DoorDash 配送优化</h1>
        <p className="text-muted-foreground mt-1">
          使用因果推断分析配送算法对配送时间的真实影响
        </p>
      </div>

      {/* Introduction */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">案例背景</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <p><strong>业务问题</strong></p>
            <p className="text-sm text-muted-foreground">
              DoorDash 部署了新的配送路径优化算法，
              需要评估该算法对配送时间的因果影响。
            </p>
          </div>
          <div className="space-y-2">
            <p><strong>因果挑战</strong></p>
            <p className="text-sm text-muted-foreground">
              算法分配不是随机的，可能与订单复杂度、
              骑手经验等因素相关，存在选择偏差。
            </p>
          </div>
        </div>
        <div className="mt-4 p-4 bg-muted rounded-lg">
          <p className="text-sm text-muted-foreground">
            <strong>为什么需要因果推断？</strong>
            简单对比使用新算法和旧算法的配送时间差异可能有偏，
            因为算法分配可能与订单特征相关。
            我们使用 PSM、IPW、DR 等方法来消除这种选择偏差。
          </p>
        </div>
      </div>

      {/* Parameters */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">参数设置</h2>
        <div className="grid gap-6 md:grid-cols-2">
          <Slider
            label="订单数量"
            min={1000}
            max={20000}
            step={1000}
            value={[params.n_orders]}
            onValueChange={([v]) => setParams((p) => ({ ...p, n_orders: v }))}
          />
          <div className="space-y-2">
            <label className="text-sm font-medium">估计方法</label>
            <Select
              value={params.method}
              onValueChange={(v) => setParams((p) => ({ ...p, method: v as 'naive' | 'psm' | 'ipw' | 'dr' }))}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="naive">朴素对比 (Naive)</SelectItem>
                <SelectItem value="psm">倾向得分匹配 (PSM)</SelectItem>
                <SelectItem value="ipw">逆概率加权 (IPW)</SelectItem>
                <SelectItem value="dr">双重稳健 (DR)</SelectItem>
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
