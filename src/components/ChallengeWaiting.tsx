/**
 * Challenge Waiting component
 *
 * Shows while waiting for the challenged player to accept.
 */

import { useState, useEffect } from 'react'
import { Button } from './ui/button'
import { Card, CardContent } from './ui/card'
import { formatTimeSeconds } from '../lib/timeFormatting'

interface ChallengeWaitingProps {
  targetUsername: string
  targetRating: number | null
  targetExists: boolean
  expiresAt: number
  onCancel: () => void
}

export default function ChallengeWaiting({
  targetUsername,
  targetRating,
  targetExists,
  expiresAt,
  onCancel,
}: ChallengeWaitingProps) {
  const [timeRemaining, setTimeRemaining] = useState(() => {
    return Math.max(0, Math.floor((expiresAt - Date.now()) / 1000))
  })

  useEffect(() => {
    const interval = setInterval(() => {
      const remaining = Math.max(0, Math.floor((expiresAt - Date.now()) / 1000))
      setTimeRemaining(remaining)

      if (remaining === 0) {
        clearInterval(interval)
      }
    }, 1000)

    return () => clearInterval(interval)
  }, [expiresAt])


  return (
    <Card className="w-full max-w-md mx-auto">
      <CardContent className="pt-8 pb-6 text-center">
        {/* Animated waiting indicator */}
        <div className="relative w-20 h-20 mx-auto mb-6">
          <div className="absolute inset-0 border-4 border-primary/20 rounded-full" />
          <div className="absolute inset-0 border-4 border-primary border-t-transparent rounded-full animate-spin" />
          <div className="absolute inset-0 flex items-center justify-center">
            <svg
              className="w-8 h-8 text-primary"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 10V3L4 14h7v7l9-11h-7z"
              />
            </svg>
          </div>
        </div>

        <h2 className="text-xl font-bold mb-2">Waiting for @{targetUsername}</h2>

        {targetExists ? (
          <p className="text-muted-foreground mb-1">
            Challenge sent! Waiting for them to accept.
          </p>
        ) : (
          <p className="text-amber-600 dark:text-amber-400 mb-1">
            User not found. Challenge will be waiting if they join.
          </p>
        )}

        {targetRating && (
          <p className="text-sm text-muted-foreground">
            Rating: {targetRating}
          </p>
        )}

        {/* Countdown */}
        <div className="mt-6 mb-6">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-muted">
            <svg
              className="w-4 h-4 text-muted-foreground"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <span className={`font-mono font-medium ${timeRemaining < 60 ? 'text-amber-600 dark:text-amber-400' : ''}`}>
              {formatTimeSeconds(timeRemaining)}
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            Challenge expires in
          </p>
        </div>

        {/* Info box */}
        <div className="p-4 rounded-lg bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-800 mb-6 text-left">
          <p className="text-sm text-blue-700 dark:text-blue-300">
            <strong>Tip:</strong> The other player needs to challenge you back for the game to start.
            They'll see a notification when they're online.
          </p>
        </div>

        <Button
          variant="outline"
          onClick={onCancel}
          className="w-full"
        >
          Cancel Challenge
        </Button>
      </CardContent>
    </Card>
  )
}
