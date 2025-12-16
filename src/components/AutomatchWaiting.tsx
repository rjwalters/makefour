/**
 * Automatch Waiting component
 *
 * Shows while waiting in the automatch queue.
 * Includes progress toward bot match and "Play Bot Now" button.
 */

import { useState, useEffect } from 'react'
import { Button } from './ui/button'
import { Card, CardContent } from './ui/card'

interface AutomatchWaitingProps {
  waitTime: number // seconds since joining queue
  currentTolerance: number
  userRating: number
  onCancel: () => void
  onPlayBotNow: () => void
}

const BOT_READY_THRESHOLD = 60 // 60 seconds

export default function AutomatchWaiting({
  waitTime,
  currentTolerance,
  userRating,
  onCancel,
  onPlayBotNow,
}: AutomatchWaitingProps) {
  const [displayTime, setDisplayTime] = useState(waitTime)

  // Update display time every second
  useEffect(() => {
    const startTime = Date.now() - waitTime * 1000

    const interval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - startTime) / 1000)
      setDisplayTime(elapsed)
    }, 1000)

    return () => clearInterval(interval)
  }, [waitTime])

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const botReadyProgress = Math.min(100, (displayTime / BOT_READY_THRESHOLD) * 100)
  const isBotReady = displayTime >= BOT_READY_THRESHOLD

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardContent className="pt-8 pb-6 text-center">
        {/* Animated searching indicator */}
        <div className="relative w-24 h-24 mx-auto mb-6">
          {/* Outer pulsing ring */}
          <div className="absolute inset-0 border-4 border-primary/30 rounded-full animate-ping" />
          {/* Middle spinning ring */}
          <div className="absolute inset-2 border-4 border-primary border-t-transparent rounded-full animate-spin" />
          {/* Inner static icon */}
          <div className="absolute inset-0 flex items-center justify-center">
            <svg
              className="w-10 h-10 text-primary"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
          </div>
        </div>

        <h2 className="text-xl font-bold mb-2">Finding Opponent</h2>
        <p className="text-muted-foreground">
          Searching for a player near your rating...
        </p>

        {/* Wait time display */}
        <div className="mt-4 mb-4">
          <span className="text-3xl font-mono font-bold tabular-nums">
            {formatTime(displayTime)}
          </span>
        </div>

        {/* Rating range */}
        <div className="flex justify-center gap-4 text-sm mb-6">
          <div className="px-3 py-1 rounded-full bg-muted">
            <span className="text-muted-foreground">Your rating: </span>
            <span className="font-medium">{userRating}</span>
          </div>
          <div className="px-3 py-1 rounded-full bg-muted">
            <span className="text-muted-foreground">Range: </span>
            <span className="font-medium">
              {Math.max(0, userRating - currentTolerance)} - {userRating + currentTolerance}
            </span>
          </div>
        </div>

        {/* Bot ready progress bar */}
        <div className="mb-6">
          <div className="flex justify-between text-xs text-muted-foreground mb-1">
            <span>Searching for players</span>
            <span>{isBotReady ? 'Bot ready!' : `${Math.floor(botReadyProgress)}%`}</span>
          </div>
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-500 ${
                isBotReady ? 'bg-green-500' : 'bg-primary'
              }`}
              style={{ width: `${botReadyProgress}%` }}
            />
          </div>
          {!isBotReady && (
            <p className="text-xs text-muted-foreground mt-1">
              Bot match available in {BOT_READY_THRESHOLD - displayTime}s
            </p>
          )}
        </div>

        {/* Play Bot Now button */}
        <Button
          variant={isBotReady ? 'default' : 'outline'}
          onClick={onPlayBotNow}
          className={`w-full mb-3 transition-all ${
            isBotReady ? 'animate-pulse' : ''
          }`}
        >
          {isBotReady ? 'Play Bot Now (Ready!)' : 'Play Bot Now'}
        </Button>

        <Button variant="ghost" onClick={onCancel} className="w-full">
          Cancel Search
        </Button>

        {/* Info text */}
        <p className="text-xs text-muted-foreground mt-4">
          {isBotReady
            ? 'No players found yet. Click above to play a bot at your rating level.'
            : 'Looking for human players first. You can play a bot anytime.'}
        </p>
      </CardContent>
    </Card>
  )
}
