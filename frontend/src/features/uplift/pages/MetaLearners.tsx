import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { useMetaLearners } from '../hooks'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { Slider } from '@/components/ui/Slider'
import { Button } from '@/components/ui/Button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import { Loader2 } from 'lucide-react'

export function Component() {
  const [params, setParams] = useState({
    n_samples: 3000,
    effect_type: 'heterogeneous' as const,
    noise_level: 0.5,
  })

  const { mutate, data, isPending, error } = useMetaLearners()

  const handleAnalyze = () => {
    mutate(params)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Meta-Learners</h1>
        <p className="text-muted-foreground mt-1">
          S-Learner, T-Learner, X-Learner - 估计条件平均处理效应 (CATE)
        </p>
      </div>

      {/* Introduction */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">核心概念</h2>
        <div className="grid gap-4 md:grid-cols-3">
          <div className="space-y-2 p-4 bg-muted rounded-lg">
            <p className="font-semibold">S-Learner</p>
            <p className="text-sm text-muted-foreground">
              单模型方法：将处理变量作为特征，训练单一模型预测结果
            </p>
          </div>
          <div className="space-y-2 p-4 bg-muted rounded-lg">
            <p className="font-semibold">T-Learner</p>
            <p className="text-sm text-muted-foreground">
              双模型方法：分别为处理组和对照组训练独立模型
            </p>
          </div>
          <div className="space-y-2 p-4 bg-muted rounded-lg">
            <p className="font-semibold">X-Learner</p>
            <p className="text-sm text-muted-foreground">
              交叉学习：使用伪结果和倾向得分加权，适合不平衡数据
            </p>
          </div>
        </div>
        <div className="mt-4 p-4 bg-muted rounded-lg">
          <p className="text-sm text-muted-foreground">
            <strong>CATE (τ(x))</strong>: 条件平均处理效应，表示给定协变量 X=x 时的期望处理效应 E[Y(1)-Y(0)|X=x]。
            Meta-Learners 通过不同策略将 CATE 估计转化为标准监督学习问题。
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
          <div className="space-y-2">
            <label className="text-sm font-medium">效应类型</label>
            <Select
              value={params.effect_type}
              onValueChange={(v) => setParams((p) => ({ ...p, effect_type: v as 'constant' | 'heterogeneous' | 'complex' }))}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="constant">常数效应</SelectItem>
                <SelectItem value="heterogeneous">异质性效应</SelectItem>
                <SelectItem value="complex">复杂效应</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <Slider
            label="噪声水平"
            min={0}
            max={2}
            step={0.1}
            value={[params.noise_level]}
            onValueChange={([v]) => setParams((p) => ({ ...p, noise_level: v }))}
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
