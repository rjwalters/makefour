/**
 * Bot Persona Selector Component
 *
 * Displays a grid of bot personas for the user to challenge
 */

import { Button } from './ui/button'
import { useBotPersonas, type BotPersona } from '../hooks/useBotPersonas'
import BotAvatar from './BotAvatar'
import { usePlayerBotStats, type BotStatsRecord } from '../hooks/usePlayerBotStats'

interface BotPersonaSelectorProps {
  selectedPersonaId: string | null
  onSelect: (persona: BotPersona) => void
}

// Play style badges
const PLAY_STYLE_COLORS: Record<string, string> = {
  aggressive: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300',
  defensive: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300',
  balanced: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300',
  tricky: 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300',
  adaptive: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300',
}

// Rating tier colors
function getRatingTierColor(rating: number): string {
  if (rating < 900) return 'text-gray-500'
  if (rating < 1200) return 'text-green-600 dark:text-green-400'
  if (rating < 1500) return 'text-blue-600 dark:text-blue-400'
  if (rating < 1800) return 'text-purple-600 dark:text-purple-400'
  return 'text-amber-600 dark:text-amber-400'
}

function getRatingTierName(rating: number): string {
  if (rating < 900) return 'Beginner'
  if (rating < 1200) return 'Intermediate'
  if (rating < 1500) return 'Advanced'
  if (rating < 1800) return 'Expert'
  return 'Master'
}

function BotPersonaCard({
  persona,
  isSelected,
  onSelect,
  playerStats,
}: {
  persona: BotPersona
  isSelected: boolean
  onSelect: () => void
  playerStats: BotStatsRecord | null
}) {
  return (
    <button
      onClick={onSelect}
      className={`
        relative w-full p-4 rounded-lg border-2 text-left transition-all
        hover:shadow-md hover:border-primary/50
        focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2
        ${isSelected
          ? 'border-primary bg-primary/5 shadow-sm'
          : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800'
        }
      `}
    >
      {/* Header with avatar, name and rating */}
      <div className="flex justify-between items-start mb-2">
        <div className="flex items-start gap-3">
          <BotAvatar
            avatarUrl={persona.avatarUrl}
            name={persona.name}
            size="md"
          />
          <div>
            <h3 className="font-semibold text-lg">{persona.name}</h3>
            <span
              className={`text-sm font-medium ${getRatingTierColor(persona.rating)}`}
            >
              {persona.rating} - {getRatingTierName(persona.rating)}
            </span>
          </div>
        </div>
        {/* Play style badge */}
        <span
          className={`text-xs px-2 py-1 rounded-full font-medium ${
            PLAY_STYLE_COLORS[persona.playStyle] || PLAY_STYLE_COLORS.balanced
          }`}
        >
          {persona.playStyle}
        </span>
      </div>

      {/* Description */}
      <p className="text-sm text-muted-foreground mb-3 line-clamp-2">
        {persona.description}
      </p>

      {/* Player's record against this bot */}
      {playerStats && playerStats.totalGames > 0 ? (
        <div className="flex items-center justify-between mb-2 px-2 py-1.5 bg-muted/50 rounded">
          <span className="text-xs font-medium">Your record:</span>
          <div className="flex items-center gap-2">
            <span className="text-sm font-bold">
              <span className="text-green-600 dark:text-green-400">{playerStats.wins}</span>
              <span className="text-muted-foreground">-</span>
              <span className="text-red-600 dark:text-red-400">{playerStats.losses}</span>
              {playerStats.draws > 0 && (
                <>
                  <span className="text-muted-foreground">-</span>
                  <span className="text-yellow-600 dark:text-yellow-400">{playerStats.draws}</span>
                </>
              )}
            </span>
            {playerStats.isMastered && (
              <span className="text-xs px-1.5 py-0.5 rounded bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300">
                Mastered
              </span>
            )}
            {!playerStats.isMastered && playerStats.isUndefeated && (
              <span className="text-xs px-1.5 py-0.5 rounded bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300">
                Undefeated
              </span>
            )}
            {playerStats.losses > 0 && playerStats.wins === 0 && (
              <span className="text-xs px-1.5 py-0.5 rounded bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300">
                Not beaten
              </span>
            )}
          </div>
        </div>
      ) : (
        <div className="flex items-center justify-between mb-2 px-2 py-1.5 bg-muted/30 rounded">
          <span className="text-xs text-muted-foreground italic">Not yet challenged</span>
        </div>
      )}

      {/* Bot Stats */}
      <div className="flex gap-4 text-xs text-muted-foreground">
        <span>
          {persona.gamesPlayed} games
        </span>
        {persona.gamesPlayed > 0 && (
          <span>
            {persona.winRate}% win rate
          </span>
        )}
      </div>

      {/* Selected indicator */}
      {isSelected && (
        <div className="absolute top-2 right-2">
          <div className="w-4 h-4 rounded-full bg-primary flex items-center justify-center">
            <svg
              className="w-3 h-3 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={3}
                d="M5 13l4 4L19 7"
              />
            </svg>
          </div>
        </div>
      )}
    </button>
  )
}

export default function BotPersonaSelector({
  selectedPersonaId,
  onSelect,
}: BotPersonaSelectorProps) {
  const { personas, isLoading, error } = useBotPersonas()
  const { data: playerBotStats } = usePlayerBotStats()

  // Create a map of bot ID to player stats for quick lookup
  const statsMap = new Map<string, BotStatsRecord>()
  if (playerBotStats) {
    for (const stat of playerBotStats.stats) {
      statsMap.set(stat.botId, stat)
    }
  }

  if (isLoading) {
    return (
      <div className="space-y-2">
        <label className="block text-sm font-medium mb-2">Choose Your Opponent</label>
        <div className="grid grid-cols-1 gap-3">
          {[1, 2, 3, 4].map((i) => (
            <div
              key={i}
              className="h-24 bg-gray-100 dark:bg-gray-800 rounded-lg animate-pulse"
            />
          ))}
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="space-y-2">
        <label className="block text-sm font-medium mb-2">Choose Your Opponent</label>
        <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg text-center">
          <p className="text-sm text-red-600 dark:text-red-400">
            Failed to load bot personas. Please try again.
          </p>
          <Button variant="outline" size="sm" className="mt-2" onClick={() => window.location.reload()}>
            Retry
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium mb-2">Choose Your Opponent</label>
      <div className="grid grid-cols-1 gap-3 max-h-[400px] overflow-y-auto pr-1">
        {personas.map((persona) => (
          <BotPersonaCard
            key={persona.id}
            persona={persona}
            isSelected={selectedPersonaId === persona.id}
            onSelect={() => onSelect(persona)}
            playerStats={statsMap.get(persona.id) || null}
          />
        ))}
      </div>
      {personas.length === 0 && (
        <p className="text-sm text-muted-foreground text-center py-4">
          No bot personas available.
        </p>
      )}
    </div>
  )
}
