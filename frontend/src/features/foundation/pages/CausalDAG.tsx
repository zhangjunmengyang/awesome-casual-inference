import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { useCausalDAG } from '../hooks'
import { PlotlyChart } from '@/components/charts/PlotlyChart'
import { Button } from '@/components/ui/Button'
import { Loader2 } from 'lucide-react'

const scenarios = [
  { value: 'confounding', label: '混淆 (Confounding)', desc: 'U 同时影响 T 和 Y' },
  { value: 'mediation', label: '中介 (Mediation)', desc: 'M 传递 T 对 Y 的效应' },
  { value: 'collider', label: '碰撞 (Collider)', desc: 'C 同时被 T 和 Y 影响' },
] as const

export function Component() {
  const [scenario, setScenario] = useState<'confounding' | 'mediation' | 'collider'>('confounding')
  const { mutate, data, isPending, error } = useCausalDAG()

  const handleAnalyze = () => {
    mutate({ scenario })
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">因果图</h1>
        <p className="text-muted-foreground mt-1">
          Causal Directed Acyclic Graph (DAG)
        </p>
      </div>

      {/* Scenario Selection */}
      <div className="rounded-lg border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">选择场景</h2>
        <div className="grid gap-4 md:grid-cols-3">
          {scenarios.map((s) => (
            <button
              key={s.value}
              onClick={() => setScenario(s.value)}
              className={`p-4 rounded-lg border text-left transition-colors ${
                scenario === s.value
                  ? 'border-primary bg-primary/5'
                  : 'hover:border-primary/50'
              }`}
            >
              <div className="font-semibold">{s.label}</div>
              <div className="text-sm text-muted-foreground mt-1">{s.desc}</div>
            </button>
          ))}
        </div>
        <div className="mt-6">
          <Button onClick={handleAnalyze} disabled={isPending}>
            {isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            生成因果图
          </Button>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-destructive bg-destructive/10 p-4 text-destructive">
          {error.message}
        </div>
      )}

      {data && (
        <>
          <div className="rounded-lg border bg-card p-6">
            <h2 className="text-lg font-semibold mb-4">因果图</h2>
            {data.charts[0] && (
              <PlotlyChart data={data.charts[0]} className="h-[400px]" />
            )}
          </div>

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
