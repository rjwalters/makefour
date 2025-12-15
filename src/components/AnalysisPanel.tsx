import { useMemo } from 'react'
import { cn } from '@/lib/utils'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import type { Board, Player } from '../game/makefour'
import {
  analyzeThreats,
  getQuickEvaluation,
  type ThreatAnalysis,
} from '../ai/coach'

interface AnalysisPanelProps {
  board: Board
  currentPlayer: Player
  isGameOver: boolean
  disabled?: boolean
}

/**
 * Displays real-time position analysis during gameplay.
 * Shows evaluation bar, threat detection, and position assessment.
 */
export default function AnalysisPanel({
  board,
  currentPlayer,
  isGameOver,
  disabled = false,
}: AnalysisPanelProps) {
  // Compute analysis - memoized for performance
  const analysis = useMemo(() => {
    if (isGameOver || disabled) return null
    return getQuickEvaluation(board, currentPlayer)
  }, [board, currentPlayer, isGameOver, disabled])

  const threats = useMemo((): ThreatAnalysis | null => {
    if (isGameOver || disabled) return null
    return analyzeThreats(board, currentPlayer)
  }, [board, currentPlayer, isGameOver, disabled])

  if (disabled || isGameOver) {
    return null
  }

  // Convert score to percentage for evaluation bar (clamped -1000 to 1000)
  const scorePercent = analysis
    ? Math.min(100, Math.max(0, ((analysis.score + 1000) / 2000) * 100))
    : 50

  // Determine bar color based on result
  const getBarColor = () => {
    if (!analysis) return 'bg-gray-400'
    switch (analysis.result) {
      case 'win':
        return 'bg-green-500'
      case 'loss':
        return 'bg-red-500'
      case 'draw':
        return 'bg-gray-500'
      default:
        if (analysis.score > 100) return 'bg-green-400'
        if (analysis.score < -100) return 'bg-red-400'
        return 'bg-gray-400'
    }
  }

  // Get evaluation text color
  const getTextColor = () => {
    if (!analysis) return 'text-muted-foreground'
    switch (analysis.result) {
      case 'win':
        return 'text-green-600 dark:text-green-400'
      case 'loss':
        return 'text-red-600 dark:text-red-400'
      default:
        if (analysis.score > 100) return 'text-green-600 dark:text-green-400'
        if (analysis.score < -100) return 'text-red-600 dark:text-red-400'
        return 'text-muted-foreground'
    }
  }

  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          Position Analysis
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Evaluation Bar */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Losing</span>
            <span>Equal</span>
            <span>Winning</span>
          </div>
          <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden relative">
            {/* Center marker */}
            <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-gray-400 dark:bg-gray-500 z-10" />
            {/* Score indicator */}
            <div
              className={cn(
                'h-full transition-all duration-300 rounded-full',
                getBarColor()
              )}
              style={{ width: `${scorePercent}%` }}
            />
          </div>
        </div>

        {/* Evaluation Description */}
        {analysis && (
          <p className={cn('text-sm font-medium', getTextColor())}>
            {analysis.description}
          </p>
        )}

        {/* Threat Indicators */}
        {threats && (threats.winningMoves.length > 0 || threats.blockingMoves.length > 0) && (
          <div className="space-y-2">
            {threats.winningMoves.length > 0 && (
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-green-500" />
                <span className="text-xs text-green-600 dark:text-green-400">
                  Win in column{threats.winningMoves.length > 1 ? 's' : ''}{' '}
                  {threats.winningMoves.map((c) => c + 1).join(', ')}
                </span>
              </div>
            )}
            {threats.blockingMoves.length > 0 && (
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-red-500" />
                <span className="text-xs text-red-600 dark:text-red-400">
                  Block column{threats.blockingMoves.length > 1 ? 's' : ''}{' '}
                  {threats.blockingMoves.map((c) => c + 1).join(', ')}
                </span>
              </div>
            )}
          </div>
        )}

        {/* Move count indicator */}
        <div className="text-xs text-muted-foreground">
          Player {currentPlayer === 1 ? 'Red' : 'Yellow'} to move
        </div>
      </CardContent>
    </Card>
  )
}
