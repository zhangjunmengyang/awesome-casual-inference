import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { useGrowthAttribution } from '../hooks'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { Slider } from '@/components/ui/Slider'
import { Button } from '@/components/ui/Button'
import { Loader2 } from 'lucide-react'

export function Component() {
  const [params, setParams] = useState({
    n_users: 10000,
    channels: ['organic', 'paid', 'referral', 'email'],
  })

  const { mutate, data, isPending, error } = useGrowthAttribution()

  const handleAnalyze = () => {
    mutate(params)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">增长归因分析</h1>
        <p className="text-muted-foreground mt-1">
          Growth Attribution - 使用因果方法进行多渠道归因分析
        </p>
      </div>

      {/* Introduction */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">案例背景</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <p><strong>业务问题</strong></p>
            <p className="text-sm text-muted-foreground">
              用户可能通过多个渠道 (自然流量、付费广告、推荐、邮件) 接触产品，
              需要准确归因各渠道对转化的真实贡献。
            </p>
          </div>
          <div className="space-y-2">
            <p><strong>因果挑战</strong></p>
            <p className="text-sm text-muted-foreground">
              传统的末次点击归因会低估早期触点的价值，
              而简单的多点归因又无法区分相关性和因果性。
            </p>
          </div>
        </div>
        <div className="mt-4 p-4 bg-muted rounded-lg">
          <p className="text-sm text-muted-foreground">
            <strong>因果归因 vs 传统归因</strong>
            传统归因模型 (如末次点击、线性归因) 只是分配信用的规则，
            不能回答 "如果用户没有看到这个广告会怎样" 这样的反事实问题。
            因果归因使用倾向得分等方法，尝试估计各渠道的真实增量贡献。
          </p>
        </div>
      </div>

      {/* Channels Display */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">分析渠道</h2>
        <div className="flex flex-wrap gap-2">
          {params.channels.map((channel) => (
            <span
              key={channel}
              className="px-3 py-1 rounded-full bg-primary/10 text-primary text-sm"
            >
              {channel === 'organic' && '自然流量'}
              {channel === 'paid' && '付费广告'}
              {channel === 'referral' && '推荐'}
              {channel === 'email' && '邮件营销'}
            </span>
          ))}
        </div>
      </div>

      {/* Parameters */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">参数设置</h2>
        <div className="grid gap-6 md:grid-cols-2">
          <Slider
            label="用户数量"
            min={1000}
            max={50000}
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
