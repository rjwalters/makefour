import { useTheme } from '../contexts/ThemeContext'
import { Button } from './ui/button'

export default function ThemeToggle() {
  const { theme, setTheme } = useTheme()

  const cycleTheme = () => {
    let newTheme: 'light' | 'dark' | 'system'
    if (theme === 'light') newTheme = 'dark'
    else if (theme === 'dark') newTheme = 'system'
    else newTheme = 'light'

    console.log('🔄 ThemeToggle clicked! Cycling from', theme, 'to', newTheme)
    setTheme(newTheme)
  }

  const getIcon = () => {
    if (theme === 'light') return '☀️'
    if (theme === 'dark') return '🌙'
    return '💻'
  }

  const getLabel = () => {
    if (theme === 'light') return 'Light'
    if (theme === 'dark') return 'Dark'
    return 'System'
  }

  return (
    <Button variant="outline" size="sm" onClick={cycleTheme} className="gap-2">
      <span>{getIcon()}</span>
      <span className="hidden sm:inline">{getLabel()}</span>
    </Button>
  )
}
