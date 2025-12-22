import { useAuthenticatedApiCall } from './useApi'
import { API_USERS } from '../lib/apiEndpoints'

export interface BotStatsRecord {
  botId: string
  botName: string
  botPlayStyle: string
  botRating: number
  wins: number
  losses: number
  draws: number
  totalGames: number
  winRate: number
  currentStreak: number
  bestWinStreak: number
  firstWinAt: number | null
  lastPlayedAt: number | null
  hasPositiveRecord: boolean
  isUndefeated: boolean
  isMastered: boolean
}

export interface BotStatsSummary {
  totalGames: number
  totalWins: number
  totalLosses: number
  totalDraws: number
  overallWinRate: number
  botsPlayed: number
  botsDefeated: number
  botsMastered: number
  botsRemaining: number
}

export interface PlayerBotStatsResponse {
  stats: BotStatsRecord[]
  unplayed: BotStatsRecord[]
  summary: BotStatsSummary
}

export interface SingleBotStatsResponse extends BotStatsRecord {
  botDescription: string
  hasPlayed: boolean
}

/**
 * Hook for fetching player's stats against all bots
 */
export function usePlayerBotStats() {
  return useAuthenticatedApiCall<PlayerBotStatsResponse>(API_USERS.ME_BOT_STATS)
}

/**
 * Hook for fetching player's stats against a specific bot
 */
export function useSingleBotStats(botId: string | null) {
  return useAuthenticatedApiCall<SingleBotStatsResponse>(
    botId ? API_USERS.meBotStatsById(botId) : null,
    { skip: !botId }
  )
}

/**
 * Format a record as "W-L" or "W-L-D" string
 */
export function formatRecord(wins: number, losses: number, draws: number = 0): string {
  if (draws > 0) {
    return `${wins}-${losses}-${draws}`
  }
  return `${wins}-${losses}`
}

/**
 * Get a streak description string
 */
export function formatStreak(streak: number): string {
  if (streak === 0) return ''
  if (streak > 0) return `${streak}W streak`
  return `${Math.abs(streak)}L streak`
}

/**
 * Get record indicator icon
 */
export function getRecordIndicator(wins: number, losses: number): string {
  if (wins > losses) return '✅'
  if (losses > wins) return '❌'
  return '➖'
}
