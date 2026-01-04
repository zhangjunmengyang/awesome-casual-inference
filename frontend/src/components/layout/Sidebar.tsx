import { Link, useLocation } from 'react-router-dom'
import { cn } from '@/lib/utils'
import {
  BookOpen,
  GitBranch,
  TrendingUp,
  TreeDeciduous,
  Brain,
  CheckCircle,
  Building2,
  FlaskConical,
  ChevronDown,
  Home,
} from 'lucide-react'
import { useState } from 'react'

interface NavItem {
  label: string
  icon: React.ReactNode
  path?: string
  children?: { label: string; path: string }[]
}

const navItems: NavItem[] = [
  {
    label: '首页',
    icon: <Home className="h-4 w-4" />,
    path: '/',
  },
  {
    label: '基础概念',
    icon: <BookOpen className="h-4 w-4" />,
    children: [
      { label: '潜在结果框架', path: '/foundation/potential-outcomes' },
      { label: '因果图', path: '/foundation/causal-dag' },
      { label: '混淆偏差', path: '/foundation/confounding-bias' },
      { label: '选择偏差', path: '/foundation/selection-bias' },
    ],
  },
  {
    label: '处理效应估计',
    icon: <GitBranch className="h-4 w-4" />,
    children: [
      { label: '倾向得分匹配 (PSM)', path: '/treatment/psm' },
      { label: '逆概率加权 (IPW)', path: '/treatment/ipw' },
      { label: '双重稳健估计', path: '/treatment/doubly-robust' },
    ],
  },
  {
    label: 'Uplift 模型',
    icon: <TrendingUp className="h-4 w-4" />,
    children: [
      { label: 'Meta-Learners', path: '/uplift/meta-learners' },
      { label: 'Uplift Tree', path: '/uplift/uplift-tree' },
      { label: '模型评估', path: '/uplift/evaluation' },
    ],
  },
  {
    label: '异质效应',
    icon: <TreeDeciduous className="h-4 w-4" />,
    children: [
      { label: '因果森林', path: '/hetero/causal-forest' },
      { label: 'CATE 可视化', path: '/hetero/cate-visualization' },
      { label: '敏感性分析', path: '/hetero/sensitivity' },
    ],
  },
  {
    label: '评估诊断',
    icon: <CheckCircle className="h-4 w-4" />,
    children: [
      { label: '平衡性检查', path: '/evaluation/balance-check' },
    ],
  },
  {
    label: '业务案例',
    icon: <Building2 className="h-4 w-4" />,
    children: [
      { label: 'DoorDash 配送优化', path: '/cases/doordash' },
      { label: 'Netflix 推荐系统', path: '/cases/netflix' },
      { label: '增长归因', path: '/cases/growth-attribution' },
    ],
  },
  {
    label: 'A/B 测试',
    icon: <FlaskConical className="h-4 w-4" />,
    children: [
      { label: '实验分析', path: '/ab-testing/experiment-analysis' },
      { label: '样本量计算', path: '/ab-testing/sample-size' },
    ],
  },
]

function NavGroup({ item }: { item: NavItem }) {
  const location = useLocation()
  const [isOpen, setIsOpen] = useState(
    item.children?.some((child) => location.pathname === child.path) ?? false
  )

  if (item.path) {
    return (
      <Link
        to={item.path}
        className={cn(
          'flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors',
          location.pathname === item.path
            ? 'bg-primary text-primary-foreground'
            : 'text-muted-foreground hover:bg-muted hover:text-foreground'
        )}
      >
        {item.icon}
        <span>{item.label}</span>
      </Link>
    )
  }

  return (
    <div>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
      >
        <div className="flex items-center gap-3">
          {item.icon}
          <span>{item.label}</span>
        </div>
        <ChevronDown
          className={cn('h-4 w-4 transition-transform', isOpen && 'rotate-180')}
        />
      </button>
      {isOpen && item.children && (
        <div className="ml-7 mt-1 space-y-1 border-l pl-3">
          {item.children.map((child) => (
            <Link
              key={child.path}
              to={child.path}
              className={cn(
                'block rounded-lg px-3 py-1.5 text-sm transition-colors',
                location.pathname === child.path
                  ? 'bg-primary/10 text-primary font-medium'
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              {child.label}
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}

export function Sidebar() {
  return (
    <aside className="w-64 border-r bg-card p-4">
      {/* Logo */}
      <div className="mb-6 flex items-center gap-2 px-3">
        <Brain className="h-6 w-6 text-primary" />
        <div>
          <h1 className="font-semibold text-foreground">因果推断工坊</h1>
          <p className="text-xs text-muted-foreground">Causal Inference Workbench</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="space-y-1">
        {navItems.map((item) => (
          <NavGroup key={item.label} item={item} />
        ))}
      </nav>
    </aside>
  )
}
