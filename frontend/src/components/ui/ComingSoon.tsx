import { Construction } from 'lucide-react'

interface ComingSoonProps {
  title: string
  description?: string
}

export function ComingSoon({ title, description }: ComingSoonProps) {
  return (
    <div className="flex flex-col items-center justify-center py-20">
      <Construction className="h-16 w-16 text-muted-foreground mb-4" />
      <h1 className="text-2xl font-bold text-foreground mb-2">{title}</h1>
      <p className="text-muted-foreground text-center max-w-md">
        {description || '该模块正在开发中，敬请期待...'}
      </p>
    </div>
  )
}
