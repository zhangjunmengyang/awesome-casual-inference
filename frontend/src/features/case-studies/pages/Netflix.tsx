import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { useNetflix } from '../hooks'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { Slider } from '@/components/ui/Slider'
import { Button } from '@/components/ui/Button'
import { Loader2 } from 'lucide-react'

export function Component() {
  const [params, setParams] = useState({
    n_users: 5000,
  })

  const { mutate, data, isPending, error } = useNetflix()

  const handleAnalyze = () => {
    mutate(params)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Netflix 推荐系统</h1>
        <p className="text-muted-foreground mt-1">
          评估推荐系统对用户观看行为的因果影响
        </p>
      </div>

      {/* Introduction */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">案例背景</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <p><strong>业务问题</strong></p>
            <p className="text-sm text-muted-foreground">
              Netflix 需要评估个性化推荐系统对用户观看时长的因果影响，
              而不仅仅是相关性。
            </p>
          </div>
          <div className="space-y-2">
            <p><strong>因果挑战</strong></p>
            <p className="text-sm text-muted-foreground">
              推荐系统会根据用户历史行为进行推荐，
              这种自选择会导致观测数据存在偏差。
            </p>
          </div>
        </div>
        <div className="mt-4 p-4 bg-muted rounded-lg">
          <p className="text-sm text-muted-foreground">
            <strong>为什么需要因果推断？</strong>
            用户本身的偏好和观看习惯会影响他们看到的推荐内容，
            也会影响他们的观看时长。这种混淆效应使得简单的
            推荐点击率和观看时长的相关性分析存在偏差。
          </p>
        </div>
      </div>

      {/* Parameters */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">参数设置</h2>
        <div className="grid gap-6 md:grid-cols-2">
          <Slider
            label="用户数量"
            min={1000}
            max={20000}
            step={1000}
            value={[params.n_users]}
            onValueChange={([v]) => setParams((p) => ({ ...p, n_users: v }))}
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
