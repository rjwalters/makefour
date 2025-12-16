/**
 * Export Games API endpoint
 *
 * POST /api/export/games - Export user's games in various formats
 */

import { validateSession, errorResponse, jsonResponse } from '../../lib/auth'
import { createDb } from '../../../shared/db/client'
import { games, users } from '../../../shared/db/schema'
import { eq, and, gte, lte, lt, inArray, desc } from 'drizzle-orm'
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

// Schema for game from database (Drizzle returns camelCase)
interface GameRow {
  id: string
  userId: string
  outcome: string
  moves: string
  moveCount: number
  ratingChange: number | null
  opponentType: string
  aiDifficulty: string | null
  playerNumber: number
  createdAt: number
}

// Schema for user from database (Drizzle returns camelCase)
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
    ratingChange: game.ratingChange ?? 0,
    opponentType: game.opponentType,
    aiDifficulty: game.aiDifficulty,
    playerNumber: game.playerNumber,
    moveCount: game.moveCount,
    createdAt: formatIsoDate(game.createdAt),
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
    const result = outcomeToPgnResult(game.outcome, game.playerNumber)

    // Build PGN headers
    const headers = [
      `[Event "MakeFour Game"]`,
      `[Site "MakeFour Online"]`,
      `[Date "${formatPgnDate(game.createdAt)}"]`,
      `[Round "-"]`,
      `[Player1 "${game.playerNumber === 1 ? 'User' : game.opponentType === 'ai' ? `AI (${game.aiDifficulty})` : 'Opponent'}"]`,
      `[Player2 "${game.playerNumber === 2 ? 'User' : game.opponentType === 'ai' ? `AI (${game.aiDifficulty})` : 'Opponent'}"]`,
      `[Player1Rating "${game.playerNumber === 1 ? userRating : getAiRating(game.aiDifficulty)}"]`,
      `[Player2Rating "${game.playerNumber === 2 ? userRating : getAiRating(game.aiDifficulty)}"]`,
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
  const db = createDb(DB)

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
    const user = await db.query.users.findFirst({
      where: eq(users.id, session.userId),
      columns: {
        id: true,
        email: true,
        rating: true,
      },
    })

    if (!user) {
      return errorResponse('User not found', 404)
    }

    // Build query conditions
    const conditions = [eq(games.userId, session.userId)]

    if (filters) {
      // Date filters
      if (filters.dateFrom) {
        const fromTimestamp = new Date(filters.dateFrom).getTime()
        conditions.push(gte(games.createdAt, fromTimestamp))
      }

      if (filters.dateTo) {
        // Add 1 day to include the entire end date
        const toTimestamp = new Date(filters.dateTo).getTime() + 86400000
        conditions.push(lt(games.createdAt, toTimestamp))
      }

      // Move count filters
      if (filters.minMoves) {
        conditions.push(gte(games.moveCount, filters.minMoves))
      }

      if (filters.maxMoves) {
        conditions.push(lte(games.moveCount, filters.maxMoves))
      }

      // Outcome filter
      if (filters.outcomes && filters.outcomes.length > 0) {
        conditions.push(inArray(games.outcome, filters.outcomes))
      }

      // Opponent type filter
      if (filters.opponentTypes && filters.opponentTypes.length > 0) {
        conditions.push(inArray(games.opponentType, filters.opponentTypes))
      }

      // AI difficulty filter
      if (filters.aiDifficulties && filters.aiDifficulties.length > 0) {
        conditions.push(inArray(games.aiDifficulty, filters.aiDifficulties))
      }
    }

    // Execute query
    const limit = filters?.limit || 1000
    const gameResults = await db
      .select()
      .from(games)
      .where(and(...conditions))
      .orderBy(desc(games.createdAt))
      .limit(limit)

    // Export in requested format
    if (format === 'json') {
      const exportData = exportAsJson(gameResults, user.rating, filters)
      return jsonResponse(exportData)
    }

    if (format === 'pgn') {
      const pgnData = exportAsPgn(gameResults, user.rating)
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
