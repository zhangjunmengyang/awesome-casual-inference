import { Link } from 'react-router-dom'
import {
  BookOpen,
  GitBranch,
  TrendingUp,
  TreeDeciduous,
  Building2,
  FlaskConical,
  ArrowRight,
} from 'lucide-react'

const modules = [
  {
    title: '基础概念',
    description: '潜在结果框架、因果图、混淆偏差、选择偏差',
    icon: BookOpen,
    path: '/foundation/potential-outcomes',
    color: 'bg-blue-500',
  },
  {
    title: '处理效应估计',
    description: '倾向得分匹配、逆概率加权、双重稳健估计',
    icon: GitBranch,
    path: '/treatment/psm',
    color: 'bg-green-500',
  },
  {
    title: 'Uplift 模型',
    description: 'S/T/X-Learner、Uplift Tree、模型评估',
    icon: TrendingUp,
    path: '/uplift/meta-learners',
    color: 'bg-purple-500',
  },
  {
    title: '异质效应',
    description: '因果森林、CATE 可视化、敏感性分析',
    icon: TreeDeciduous,
    path: '/hetero/causal-forest',
    color: 'bg-orange-500',
  },
  {
    title: '业务案例',
    description: 'DoorDash、Netflix、增长归因',
    icon: Building2,
    path: '/cases/doordash',
    color: 'bg-red-500',
  },
  {
    title: 'A/B 测试',
    description: '实验分析、样本量计算、CUPED',
    icon: FlaskConical,
    path: '/ab-testing/experiment-analysis',
    color: 'bg-teal-500',
  },
]

export function Component() {
  return (
    <div className="max-w-5xl mx-auto">
      {/* Header */}
      <div className="mb-12 text-center">
        <h1 className="text-4xl font-bold text-foreground mb-4">
          因果推断学习工坊
        </h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          一个交互式的因果推断学习平台，涵盖从基础概念到高级应用的完整体系。
          通过可视化和实际案例，深入理解因果推断的核心思想。
        </p>
      </div>

      {/* Module Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {modules.map((module) => (
          <Link
            key={module.title}
            to={module.path}
            className="group block rounded-xl border bg-card p-6 transition-all hover:shadow-lg hover:border-primary/50"
          >
            <div className={`inline-flex p-3 rounded-lg ${module.color} mb-4`}>
              <module.icon className="h-6 w-6 text-white" />
            </div>
            <h2 className="text-xl font-semibold text-foreground mb-2 group-hover:text-primary transition-colors">
              {module.title}
            </h2>
            <p className="text-sm text-muted-foreground mb-4">
              {module.description}
            </p>
            <div className="flex items-center text-sm text-primary font-medium">
              开始学习
              <ArrowRight className="ml-1 h-4 w-4 transition-transform group-hover:translate-x-1" />
            </div>
          </Link>
        ))}
      </div>

      {/* Learning Path */}
      <div className="mt-16 rounded-xl border bg-card p-8">
        <h2 className="text-2xl font-bold text-foreground mb-6">学习路径</h2>
        <div className="relative">
          <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-border" />
          <div className="space-y-8">
            {[
              { step: 1, title: '基础概念', desc: '理解潜在结果框架和因果图' },
              { step: 2, title: '处理效应估计', desc: '学习 PSM、IPW 等经典方法' },
              { step: 3, title: 'Uplift 建模', desc: '掌握 Meta-Learners 和 Uplift Tree' },
              { step: 4, title: '异质效应', desc: '使用因果森林估计个体效应' },
              { step: 5, title: '实战应用', desc: '通过真实案例巩固所学知识' },
            ].map((item) => (
              <div key={item.step} className="relative flex gap-4 pl-10">
                <div className="absolute left-0 flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground text-sm font-bold">
                  {item.step}
                </div>
                <div>
                  <h3 className="font-semibold text-foreground">{item.title}</h3>
                  <p className="text-sm text-muted-foreground">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
