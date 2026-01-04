import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { useBalanceCheck } from '../hooks'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { Slider } from '@/components/ui/Slider'
import { Button } from '@/components/ui/Button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import { Loader2 } from 'lucide-react'

export function Component() {
  const [params, setParams] = useState({
    n_samples: 1000,
    confounding_strength: 1.0,
    method: 'psm' as const,
  })

  const { mutate, data, isPending, error } = useBalanceCheck()

  const handleAnalyze = () => {
    mutate(params)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">平衡性检查</h1>
        <p className="text-muted-foreground mt-1">
          Balance Check - 检验处理组和控制组的协变量分布是否平衡
        </p>
      </div>

      {/* Introduction */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">核心概念</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <p><strong>标准化均值差 (SMD)</strong></p>
            <p className="text-sm text-muted-foreground">
              衡量处理组和对照组协变量均值差异的标准化指标，
              一般认为 SMD &lt; 0.1 表示良好平衡
            </p>
          </div>
          <div className="space-y-2">
            <p><strong>Love Plot</strong></p>
            <p className="text-sm text-muted-foreground">
              可视化各协变量在调整前后的 SMD 变化，
              直观展示平衡改善效果
            </p>
          </div>
        </div>
        <div className="mt-4 p-4 bg-muted rounded-lg">
          <p className="text-sm text-muted-foreground">
            <strong>为什么要检查平衡性？</strong>
            因果推断方法（如 PSM、IPW）的目标是使处理组和对照组在协变量分布上相似，
            从而模拟随机实验。平衡性检查用于验证这一目标是否达成，
            是因果分析的重要诊断步骤。
          </p>
        </div>
      </div>

      {/* Parameters */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">参数设置</h2>
        <div className="grid gap-6 md:grid-cols-3">
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
            <label className="text-sm font-medium">平衡方法</label>
            <Select
              value={params.method}
              onValueChange={(v) => setParams((p) => ({ ...p, method: v as 'psm' | 'ipw' | 'none' }))}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">无调整 (原始数据)</SelectItem>
                <SelectItem value="psm">倾向得分匹配 (PSM)</SelectItem>
                <SelectItem value="ipw">逆概率加权 (IPW)</SelectItem>
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
