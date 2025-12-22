/**
 * Shared API response types
 *
 * Consolidated type definitions for API responses used across the application.
 * This prevents duplication and ensures consistency.
 */

// =============================================================================
// Game Types
// =============================================================================

/**
 * Outcome of a completed game
 */
export type GameOutcome = 'win' | 'loss' | 'draw'

/**
 * Type of opponent in a game
 */
export type OpponentType = 'human' | 'ai'

/**
 * Recent game summary for display in lists
 */
export interface RecentGame {
  id: string
  outcome: GameOutcome
  moveCount: number
  ratingChange: number
  opponentType: OpponentType | string
  aiDifficulty?: string | null
  createdAt: number
}

/**
 * Rating history entry
 */
export interface RatingHistoryEntry {
  rating: number
  createdAt: number
}

// =============================================================================
// User Stats Types
// =============================================================================

/**
 * Core user statistics shared across responses
 */
export interface UserCoreStats {
  rating: number
  gamesPlayed: number
  wins: number
  losses: number
  draws: number
}

/**
 * Extended user info with account details
 */
export interface UserWithAccountInfo extends UserCoreStats {
  id: string
  email: string
  email_verified: boolean
  oauth_provider: string | null
  createdAt: number
  lastLogin: number
  updatedAt: number
}

/**
 * User performance statistics
 */
export interface UserPerformanceStats {
  peakRating: number
  lowestRating: number
  currentStreak: number
  longestWinStreak: number
  longestLossStreak: number
  ratingTrend: 'improving' | 'declining' | 'stable'
  recentRatingChange: number
}

/**
 * Extended performance stats with game breakdown
 */
export interface ExtendedPerformanceStats extends UserPerformanceStats {
  avgMoveCount: number
  gamesAsPlayer1: number
  gamesAsPlayer2: number
  aiGames: number
  humanGames: number
}

/**
 * Full user stats response from /api/users/me/stats
 */
export interface UserStatsResponse {
  user: UserWithAccountInfo
  stats: ExtendedPerformanceStats
  ratingHistory: RatingHistoryEntry[]
}

/**
 * Simplified user stats for StatsPage (subset of full response)
 */
export interface UserStatsSimple {
  user: UserCoreStats
  stats: UserPerformanceStats
  ratingHistory: RatingHistoryEntry[]
}

// =============================================================================
// Stats History Types
// =============================================================================

/**
 * Daily statistics entry
 */
export interface DailyStats {
  date: string
  games: number
  wins: number
  losses: number
  draws: number
  ratingChange: number
  avgMoveCount: number
}

/**
 * Weekly statistics entry
 */
export interface WeeklyStats {
  week: string
  games: number
  wins: number
  losses: number
  draws: number
}

/**
 * Win rate over time entry
 */
export interface WinRateEntry {
  date: string
  winRate: number
  games: number
}

/**
 * Opening column statistics
 */
export interface OpeningStats {
  column: number
  games: number
  wins: number
  losses: number
  draws: number
  winRate: number
}

/**
 * Recent game with extended details for stats page
 */
export interface RecentGameDetailed extends RecentGame {
  playerNumber: number
  firstMove: number | null
}

/**
 * Stats history response from /api/users/me/stats/history
 */
export interface StatsHistoryResponse {
  dateRange: {
    start: number
    end: number
  }
  summary: {
    totalGames: number
    wins: number
    losses: number
    draws: number
    avgMovesToWin: number
    avgMovesToLoss: number
    player1WinRate: number
    player2WinRate: number
  }
  dailyStats: DailyStats[]
  weeklyStats: WeeklyStats[]
  winRateOverTime: WinRateEntry[]
  openingStats: OpeningStats[]
  recentGames: RecentGameDetailed[]
}

// =============================================================================
// Leaderboard Types
// =============================================================================

/**
 * Single entry in the leaderboard
 */
export interface LeaderboardEntry {
  rank: number
  userId: string
  username: string
  rating: number
  gamesPlayed: number
  wins: number
  losses: number
  draws: number
  winRate: number
  isBot: boolean
  botPersonaId: string | null
  botDescription: string | null
  botAvatarUrl: string | null
}

/**
 * Current user's position in the leaderboard
 */
export interface CurrentUserPosition {
  rank: number
  entry: LeaderboardEntry
}

/**
 * Leaderboard response with pagination
 */
export interface LeaderboardResponse {
  leaderboard: LeaderboardEntry[]
  pagination: {
    total: number
    limit: number
    offset: number
    hasMore: boolean
  }
  currentUser: CurrentUserPosition | null
}

// =============================================================================
// Bot Types
// =============================================================================

/**
 * AI engine configuration
 */
export interface AIConfig {
  searchDepth?: number
  errorRate?: number
  timeMultiplier?: number
  temperature?: number
  useHybridSearch?: boolean
  hybridDepth?: number
  modelId?: string
}

/**
 * Bot matchup statistics
 */
export interface BotMatchup {
  opponentId: string
  opponentName: string
  opponentAvatarUrl: string | null
  totalGames: number
  wins: number
  losses: number
  draws: number
  winRate: number
  avgMoves: number
  lastGameAt: number
}

/**
 * Full bot profile response
 */
export interface BotProfile {
  id: string
  name: string
  description: string
  avatarUrl: string | null
  playStyle: string
  aiEngine?: string
  aiConfig?: AIConfig
  baseElo: number
  createdAt: number
  rating: number
  gamesPlayed: number
  wins: number
  losses: number
  draws: number
  winRate: number
  recentGames: RecentGame[]
  ratingHistory: RatingHistoryEntry[]
  matchups?: BotMatchup[]
}

// =============================================================================
// Username Types
// =============================================================================

/**
 * Username status response
 */
export interface UsernameStatus {
  username: string | null
  displayName: string
  canChange: boolean
  nextChangeAt: number | null
}

/**
 * Username change response
 */
export interface UsernameChangeResponse {
  success: boolean
  username: string
  nextChangeAt: number
}
