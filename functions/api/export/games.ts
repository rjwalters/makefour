/**
 * Export Games API endpoint
 *
 * POST /api/export/games - Export user's games in various formats
 */

import { validateSession, errorResponse, jsonResponse } from '../../lib/auth'
import { z } from 'zod'

interface Env {
  DB: D1Database
}

// Schema for export request
const exportRequestSchema = z.object({
  format: z.enum(['json', 'pgn']),
  filters: z
    .object({
      dateFrom: z.string().optional(),
      dateTo: z.string().optional(),
      minMoves: z.number().int().positive().optional(),
      maxMoves: z.number().int().positive().optional(),
      outcomes: z.array(z.enum(['win', 'loss', 'draw'])).optional(),
      opponentTypes: z.array(z.enum(['human', 'ai'])).optional(),
      aiDifficulties: z.array(z.enum(['beginner', 'intermediate', 'expert', 'perfect'])).optional(),
      limit: z.number().int().positive().max(10000).optional(),
    })
    .optional(),
})

// Schema for game from database
interface GameRow {
  id: string
  user_id: string
  outcome: string
  moves: string
  move_count: number
  rating_change: number | null
  opponent_type: string
  ai_difficulty: string | null
  player_number: number
  created_at: number
}

// Schema for user from database
interface UserRow {
  id: string
  email: string
  rating: number
}

/**
 * Convert column number to algebraic notation (a-g)
 */
function columnToLetter(col: number): string {
  return String.fromCharCode(97 + col) // 0 -> 'a', 1 -> 'b', etc.
}

/**
 * Format timestamp to PGN date format (YYYY.MM.DD)
 */
function formatPgnDate(timestamp: number): string {
  const date = new Date(timestamp)
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${year}.${month}.${day}`
}

/**
 * Format timestamp to ISO date format (YYYY-MM-DD)
 */
function formatIsoDate(timestamp: number): string {
  return new Date(timestamp).toISOString().split('T')[0]
}

/**
 * Convert outcome to PGN result format
 */
function outcomeToPgnResult(outcome: string, playerNumber: number): string {
  if (outcome === 'draw') return '1/2-1/2'
  if (outcome === 'win') return playerNumber === 1 ? '1-0' : '0-1'
  return playerNumber === 1 ? '0-1' : '1-0'
}

/**
 * Export games in JSON format
 */
function exportAsJson(
  games: GameRow[],
  userRating: number,
  filters: z.infer<typeof exportRequestSchema>['filters']
): object {
  const exportedGames = games.map((game) => ({
    id: game.id,
    moves: JSON.parse(game.moves) as number[],
    outcome: game.outcome,
    playerRating: userRating,
    ratingChange: game.rating_change ?? 0,
    opponentType: game.opponent_type,
    aiDifficulty: game.ai_difficulty,
    playerNumber: game.player_number,
    moveCount: game.move_count,
    createdAt: formatIsoDate(game.created_at),
  }))

  return {
    games: exportedGames,
    metadata: {
      exportDate: new Date().toISOString(),
      totalGames: games.length,
      filtersApplied: filters ?? {},
      format: 'makefour-v1',
    },
  }
}

/**
 * Export games in PGN-like format
 */
function exportAsPgn(games: GameRow[], userRating: number): string {
  const pgnGames = games.map((game) => {
    const moves = JSON.parse(game.moves) as number[]
    const result = outcomeToPgnResult(game.outcome, game.player_number)

    // Build PGN headers
    const headers = [
      `[Event "MakeFour Game"]`,
      `[Site "MakeFour Online"]`,
      `[Date "${formatPgnDate(game.created_at)}"]`,
      `[Round "-"]`,
      `[Player1 "${game.player_number === 1 ? 'User' : game.opponent_type === 'ai' ? `AI (${game.ai_difficulty})` : 'Opponent'}"]`,
      `[Player2 "${game.player_number === 2 ? 'User' : game.opponent_type === 'ai' ? `AI (${game.ai_difficulty})` : 'Opponent'}"]`,
      `[Player1Rating "${game.player_number === 1 ? userRating : getAiRating(game.ai_difficulty)}"]`,
      `[Player2Rating "${game.player_number === 2 ? userRating : getAiRating(game.ai_difficulty)}"]`,
      `[Result "${result}"]`,
      `[GameID "${game.id}"]`,
    ]

    // Build move list with move numbers
    const moveList = moves
      .map((col, idx) => {
        const moveNum = idx + 1
        return `${moveNum}. ${columnToLetter(col)}${getRowForMove(moves, idx)}`
      })
      .join(' ')

    return `${headers.join('\n')}\n\n${moveList} ${result}`
  })

  return pgnGames.join('\n\n\n')
}

/**
 * Get AI rating for PGN export
 */
function getAiRating(difficulty: string | null): number {
  const ratings: Record<string, number> = {
    beginner: 800,
    intermediate: 1200,
    expert: 1600,
    perfect: 2000,
  }
  return ratings[difficulty || 'intermediate'] || 1200
}

/**
 * Calculate the row number for a move in Connect Four
 * This simulates the game to determine where the piece landed
 */
function getRowForMove(moves: number[], moveIndex: number): number {
  // Build board state up to this move
  const board: number[][] = Array(7)
    .fill(null)
    .map(() => [])

  for (let i = 0; i <= moveIndex; i++) {
    const col = moves[i]
    board[col].push(i % 2 === 0 ? 1 : 2) // Alternate players
  }

  // Return the row (1-indexed from bottom)
  return board[moves[moveIndex]].length
}

/**
 * POST /api/export/games - Export games with optional filters
 */
export async function onRequestPost(context: EventContext<Env, unknown, unknown>) {
  const { DB } = context.env

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Parse and validate request body
    const body = await context.request.json()
    const parseResult = exportRequestSchema.safeParse(body)

    if (!parseResult.success) {
      const issues = parseResult.error.issues || []
      const message = issues[0]?.message || 'Invalid request data'
      return errorResponse(message, 400)
    }

    const { format, filters } = parseResult.data

    // Get user's current rating
    const user = await DB.prepare('SELECT id, email, rating FROM users WHERE id = ?')
      .bind(session.userId)
      .first<UserRow>()

    if (!user) {
      return errorResponse('User not found', 404)
    }

    // Build query with filters
    let query = `
      SELECT id, user_id, outcome, moves, move_count, rating_change, opponent_type, ai_difficulty, player_number, created_at
      FROM games
      WHERE user_id = ?
    `
    const params: (string | number)[] = [session.userId]

    if (filters) {
      // Date filters
      if (filters.dateFrom) {
        const fromTimestamp = new Date(filters.dateFrom).getTime()
        query += ' AND created_at >= ?'
        params.push(fromTimestamp)
      }

      if (filters.dateTo) {
        // Add 1 day to include the entire end date
        const toTimestamp = new Date(filters.dateTo).getTime() + 86400000
        query += ' AND created_at < ?'
        params.push(toTimestamp)
      }

      // Move count filters
      if (filters.minMoves) {
        query += ' AND move_count >= ?'
        params.push(filters.minMoves)
      }

      if (filters.maxMoves) {
        query += ' AND move_count <= ?'
        params.push(filters.maxMoves)
      }

      // Outcome filter
      if (filters.outcomes && filters.outcomes.length > 0) {
        const placeholders = filters.outcomes.map(() => '?').join(', ')
        query += ` AND outcome IN (${placeholders})`
        params.push(...filters.outcomes)
      }

      // Opponent type filter
      if (filters.opponentTypes && filters.opponentTypes.length > 0) {
        const placeholders = filters.opponentTypes.map(() => '?').join(', ')
        query += ` AND opponent_type IN (${placeholders})`
        params.push(...filters.opponentTypes)
      }

      // AI difficulty filter
      if (filters.aiDifficulties && filters.aiDifficulties.length > 0) {
        const placeholders = filters.aiDifficulties.map(() => '?').join(', ')
        query += ` AND ai_difficulty IN (${placeholders})`
        params.push(...filters.aiDifficulties)
      }
    }

    // Order by date and apply limit
    query += ' ORDER BY created_at DESC'
    const limit = filters?.limit || 1000
    query += ' LIMIT ?'
    params.push(limit)

    // Execute query
    const stmt = DB.prepare(query)
    const games = await stmt.bind(...params).all<GameRow>()

    // Export in requested format
    if (format === 'json') {
      const exportData = exportAsJson(games.results, user.rating, filters)
      return jsonResponse(exportData)
    }

    if (format === 'pgn') {
      const pgnData = exportAsPgn(games.results, user.rating)
      return jsonResponse({
        content: pgnData,
        format: 'pgn',
        filename: `makefour-games-${formatIsoDate(Date.now())}.pgn`,
      })
    }

    return errorResponse('Unsupported format', 400)
  } catch (error) {
    console.error('POST /api/export/games error:', error)
    return errorResponse('Internal server error', 500)
  }
}

/**
 * Handle OPTIONS for CORS preflight
 */
export async function onRequestOptions() {
  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  })
}
