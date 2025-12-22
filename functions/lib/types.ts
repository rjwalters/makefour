/**
 * Shared type definitions for API endpoints
 *
 * These types represent database row structures used across multiple endpoints.
 * Using shared types ensures consistency and reduces duplication.
 */

/**
 * Active game row from the database.
 * This is a superset of all fields - individual endpoints may only use a subset.
 */
export interface ActiveGameRow {
  id: string
  player1_id: string
  player2_id: string
  moves: string
  current_turn: number
  status: string
  mode: string
  winner: string | null
  player1_rating: number
  player2_rating: number
  last_move_at: number
  // Timer fields (null = untimed game)
  time_control_ms: number | null
  player1_time_ms: number | null
  player2_time_ms: number | null
  turn_started_at: number | null
  // Spectator fields
  spectatable: number
  spectator_count: number
  // Bot game fields
  is_bot_game: number
  bot_difficulty: string | null
  bot_persona_id: string | null
  // Bot vs bot fields
  is_bot_vs_bot: number
  bot1_persona_id: string | null
  bot2_persona_id: string | null
  move_delay_ms: number | null
  next_move_at: number | null
  // Timestamps
  created_at: number
  updated_at: number
}

/**
 * User row from the database.
 * This is a superset of all fields - individual endpoints may only use a subset.
 */
export interface UserRow {
  id: string
  email: string
  email_verified: number
  username: string | null
  password_hash: string | null
  oauth_provider: string | null
  oauth_id: string | null
  rating: number
  games_played: number
  wins: number
  losses: number
  draws: number
  is_bot: number
  created_at: number
  updated_at: number
}

/**
 * Bot persona row from the database.
 * This is a superset of all fields - individual endpoints may only use a subset.
 */
export interface BotPersonaRow {
  id: string
  name: string
  description: string
  avatar_url: string | null
  play_style: string
  base_elo: number
  current_elo: number
  ai_engine: string
  ai_config: string
  chat_personality: string
  games_played: number
  wins: number
  losses: number
  draws: number
  is_active: number
  created_at: number
  updated_at: number
}

/**
 * Safely parse a moves JSON string into an array of column numbers.
 * Returns an empty array if parsing fails.
 *
 * @param movesJson - JSON string of moves (e.g., "[3, 4, 2]")
 * @returns Array of column numbers, or empty array on error
 */
export function safeParseMoves(movesJson: string | null | undefined): number[] {
  if (!movesJson) {
    return []
  }

  try {
    const parsed = JSON.parse(movesJson)
    if (!Array.isArray(parsed)) {
      console.warn('safeParseMoves: parsed value is not an array')
      return []
    }
    // Validate each element is a valid column number
    return parsed.filter(
      (move): move is number =>
        typeof move === 'number' && Number.isInteger(move) && move >= 0 && move <= 6
    )
  } catch (error) {
    console.warn('safeParseMoves: failed to parse moves JSON:', error)
    return []
  }
}

/**
 * Game constants - import these instead of hardcoding
 */
export const ROWS = 6
export const COLUMNS = 7
export const WIN_LENGTH = 4
