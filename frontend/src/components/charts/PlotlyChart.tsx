import { useEffect, useRef } from 'react'

// 动态导入 Plotly（避免 SSR 问题）
let Plotly: typeof import('plotly.js-dist-min') | null = null

interface PlotlyChartProps {
  data: Record<string, unknown>
  className?: string
}

export function PlotlyChart({ data, className = '' }: PlotlyChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const loadAndRender = async () => {
      if (!containerRef.current) return

      // 动态加载 Plotly
      if (!Plotly) {
        Plotly = await import('plotly.js-dist-min')
      }

      const plotData = (data as { data?: unknown[] }).data || []
      const layout = (data as { layout?: Record<string, unknown> }).layout || {}

      // 调整布局以适应容器
      const responsiveLayout = {
        ...layout,
        autosize: true,
        margin: { t: 50, r: 30, b: 50, l: 60 },
      }

      Plotly.newPlot(
        containerRef.current,
        plotData as Plotly.Data[],
        responsiveLayout as Partial<Plotly.Layout>,
        { responsive: true, displayModeBar: true }
      )
    }

    loadAndRender()

    return () => {
      if (containerRef.current && Plotly) {
        Plotly.purge(containerRef.current)
      }
    }
  }, [data])

  return (
    <div
      ref={containerRef}
      className={`w-full min-h-[400px] ${className}`}
    />
  )
}
