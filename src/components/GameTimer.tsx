/**
 * Chess-clock style game timer component
 *
 * Features:
 * - Smooth countdown (updates every 100ms locally)
 * - Syncs with server time on each poll
 * - Visual warnings for low time
 * - Shows tenths of seconds when under 10 seconds
 */

import { useState, useEffect, useRef } from 'react'
import { cn } from '../lib/utils'
import { formatTimeMs } from '../lib/timeFormatting'

interface GameTimerProps {
  /** Time remaining in milliseconds from server */
  timeMs: number | null
  /** When the current turn started (server timestamp) */
  turnStartedAt: number | null
  /** Whether this player's clock is currently running */
  isActive: boolean
  /** Label to display (e.g., "You" or "Opponent") */
  label: string
  /** Player color for visual indicator */
  playerColor: 'red' | 'yellow'
  /** Optional className for styling */
  className?: string
}

const LOW_TIME_THRESHOLD = 30000 // 30 seconds
const CRITICAL_TIME_THRESHOLD = 10000 // 10 seconds

export function GameTimer({
  timeMs,
  turnStartedAt,
  isActive,
  label,
  playerColor,
  className,
}: GameTimerProps) {
  const [displayTime, setDisplayTime] = useState(timeMs ?? 0)
  const lastServerTimeRef = useRef(timeMs)
  const lastTurnStartedAtRef = useRef(turnStartedAt)

  // Sync with server time and handle local countdown
  useEffect(() => {
    if (timeMs === null || turnStartedAt === null) {
      setDisplayTime(0)
      return
    }

    // Check if server time has been updated (new poll response)
    const serverTimeUpdated =
      timeMs !== lastServerTimeRef.current ||
      turnStartedAt !== lastTurnStartedAtRef.current

    if (serverTimeUpdated) {
      lastServerTimeRef.current = timeMs
      lastTurnStartedAtRef.current = turnStartedAt

      // Calculate initial display time based on server state
      if (isActive) {
        const now = Date.now()
        const elapsed = now - turnStartedAt
        setDisplayTime(Math.max(0, timeMs - elapsed))
      } else {
        setDisplayTime(timeMs)
      }
    }

    // If not active, just show the stored time
    if (!isActive) {
      setDisplayTime(timeMs)
      return
    }

    // Local countdown for smooth display
    const interval = setInterval(() => {
      const now = Date.now()
      const elapsed = now - turnStartedAt
      const remaining = Math.max(0, timeMs - elapsed)
      setDisplayTime(remaining)
    }, 100)

    return () => clearInterval(interval)
  }, [timeMs, turnStartedAt, isActive])

  // Format time for display - show tenths when critical
  const formatTime = (ms: number): string => {
    return formatTimeMs(ms, ms < CRITICAL_TIME_THRESHOLD)
  }

  const isLowTime = displayTime > 0 && displayTime < LOW_TIME_THRESHOLD
  const isCriticalTime = displayTime > 0 && displayTime < CRITICAL_TIME_THRESHOLD
  const isExpired = displayTime === 0 && timeMs !== null

  // Don't render if no time control
  if (timeMs === null) {
    return null
  }

  return (
    <div
      className={cn(
        'flex items-center gap-2 px-3 py-2 rounded-lg transition-all duration-200',
        isActive && 'ring-2 ring-offset-2',
        isActive && playerColor === 'red' && 'ring-red-500 bg-red-50 dark:bg-red-950',
        isActive && playerColor === 'yellow' && 'ring-yellow-500 bg-yellow-50 dark:bg-yellow-950',
        !isActive && 'bg-muted/50',
        isExpired && 'bg-red-100 dark:bg-red-900',
        className
      )}
    >
      {/* Player color indicator */}
      <div
        className={cn(
          'w-3 h-3 rounded-full',
          playerColor === 'red' ? 'bg-red-500' : 'bg-yellow-400'
        )}
      />

      {/* Label */}
      <span className="text-sm font-medium text-muted-foreground min-w-[60px]">
        {label}
      </span>

      {/* Time display */}
      <span
        className={cn(
          'font-mono text-lg font-bold tabular-nums min-w-[70px] text-right',
          isExpired && 'text-red-600 dark:text-red-400',
          isCriticalTime && !isExpired && 'text-red-600 dark:text-red-400 animate-pulse',
          isLowTime && !isCriticalTime && 'text-orange-600 dark:text-orange-400',
          !isLowTime && !isExpired && 'text-foreground'
        )}
      >
        {formatTime(displayTime)}
      </span>
    </div>
  )
}

/**
 * Container for both player timers
 */
interface GameTimersProps {
  player1TimeMs: number | null
  player2TimeMs: number | null
  turnStartedAt: number | null
  currentTurn: 1 | 2
  playerNumber: 1 | 2
  gameStatus: 'active' | 'completed' | 'abandoned'
  className?: string
}

export function GameTimers({
  player1TimeMs,
  player2TimeMs,
  turnStartedAt,
  currentTurn,
  playerNumber,
  gameStatus,
  className,
}: GameTimersProps) {
  // Don't show timers if no time control
  if (player1TimeMs === null && player2TimeMs === null) {
    return null
  }

  const isGameActive = gameStatus === 'active'

  // Determine which timer is "yours" and which is "opponent"
  const yourTimeMs = playerNumber === 1 ? player1TimeMs : player2TimeMs
  const opponentTimeMs = playerNumber === 1 ? player2TimeMs : player1TimeMs
  const yourColor = playerNumber === 1 ? 'red' : 'yellow'
  const opponentColor = playerNumber === 1 ? 'yellow' : 'red'
  const isYourTurn = currentTurn === playerNumber
  const isOpponentTurn = currentTurn !== playerNumber

  return (
    <div className={cn('flex flex-col gap-2', className)}>
      {/* Opponent timer (top) */}
      <GameTimer
        timeMs={opponentTimeMs}
        turnStartedAt={turnStartedAt}
        isActive={isGameActive && isOpponentTurn}
        label="Opponent"
        playerColor={opponentColor}
      />

      {/* Your timer (bottom) */}
      <GameTimer
        timeMs={yourTimeMs}
        turnStartedAt={turnStartedAt}
        isActive={isGameActive && isYourTurn}
        label="You"
        playerColor={yourColor}
      />
    </div>
  )
}

/**
 * Timer display for spectators - shows both player names
 */
interface SpectatorTimersProps {
  player1TimeMs: number | null
  player2TimeMs: number | null
  turnStartedAt: number | null
  currentTurn: 1 | 2
  player1Name: string
  player2Name: string
  gameStatus: 'active' | 'completed' | 'abandoned'
  className?: string
}

export function SpectatorTimers({
  player1TimeMs,
  player2TimeMs,
  turnStartedAt,
  currentTurn,
  player1Name,
  player2Name,
  gameStatus,
  className,
}: SpectatorTimersProps) {
  // Don't show timers if no time control
  if (player1TimeMs === null && player2TimeMs === null) {
    return null
  }

  const isGameActive = gameStatus === 'active'

  return (
    <div className={cn('flex flex-col gap-2', className)}>
      {/* Player 1 (Red) timer */}
      <GameTimer
        timeMs={player1TimeMs}
        turnStartedAt={turnStartedAt}
        isActive={isGameActive && currentTurn === 1}
        label={player1Name}
        playerColor="red"
      />

      {/* Player 2 (Yellow) timer */}
      <GameTimer
        timeMs={player2TimeMs}
        turnStartedAt={turnStartedAt}
        isActive={isGameActive && currentTurn === 2}
        label={player2Name}
        playerColor="yellow"
      />
    </div>
  )
}

export default GameTimer
