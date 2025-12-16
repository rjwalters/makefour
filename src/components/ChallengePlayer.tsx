/**
 * Challenge Player component
 *
 * UI for entering a username to challenge a specific player.
 */

import { useState } from 'react'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'

interface ChallengePlayerProps {
  onSendChallenge: (username: string) => void
  onBack: () => void
  isLoading: boolean
  error: string | null
}

export default function ChallengePlayer({
  onSendChallenge,
  onBack,
  isLoading,
  error,
}: ChallengePlayerProps) {
  const [username, setUsername] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = username.trim()
    if (trimmed) {
      onSendChallenge(trimmed)
    }
  }

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardHeader className="text-center">
        <CardTitle className="text-xl">Challenge a Player</CardTitle>
        <p className="text-sm text-muted-foreground mt-1">
          Enter their username to send a challenge
        </p>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="username" className="block text-sm font-medium mb-2">
              Opponent's Username
            </label>
            <Input
              id="username"
              type="text"
              placeholder="Enter username..."
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              disabled={isLoading}
              autoComplete="off"
              autoFocus
            />
            <p className="text-xs text-muted-foreground mt-1">
              Their display name (or email prefix if they haven't set one)
            </p>
          </div>

          {error && (
            <div className="p-3 rounded-lg bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800">
              <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
            </div>
          )}

          <div className="flex gap-3">
            <Button
              type="button"
              variant="outline"
              onClick={onBack}
              disabled={isLoading}
              className="flex-1"
            >
              Back
            </Button>
            <Button
              type="submit"
              disabled={!username.trim() || isLoading}
              className="flex-1"
            >
              {isLoading ? 'Sending...' : 'Challenge'}
            </Button>
          </div>
        </form>

        <div className="mt-6 p-4 rounded-lg bg-muted/50">
          <h4 className="font-medium text-sm mb-2">How it works</h4>
          <ul className="text-xs text-muted-foreground space-y-1">
            <li>1. Enter the username of the player you want to challenge</li>
            <li>2. They'll receive a notification with your challenge</li>
            <li>3. When they accept, the game starts automatically</li>
            <li>4. Both players have 5 minutes each on the clock</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  )
}
