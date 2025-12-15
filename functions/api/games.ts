/**
 * Games API endpoint
 *
 * GET /api/games - List user's games
 * POST /api/games - Save a completed game
 */

import { validateSession, errorResponse, jsonResponse } from '../lib/auth'
import { z } from 'zod'

interface Env {
  DB: D1Database
}

// Schema for creating a new game
const createGameSchema = z.object({
  outcome: z.enum(['win', 'loss', 'draw']),
  moves: z.array(z.number().int().min(0).max(6)),
  opponentType: z.enum(['human', 'ai']).default('ai'),
  aiDifficulty: z.enum(['beginner', 'intermediate', 'expert', 'perfect']).nullable().optional(),
  playerNumber: z.number().int().min(1).max(2).default(1),
})

// Schema for game from database
interface GameRow {
  id: string
  user_id: string
  outcome: string
  moves: string
  move_count: number
  opponent_type: string
  ai_difficulty: string | null
  player_number: number
  created_at: number
}

/**
 * GET /api/games - List user's games
 */
export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Get query params for pagination
    const url = new URL(context.request.url)
    const limit = Math.min(parseInt(url.searchParams.get('limit') || '20'), 100)
    const offset = parseInt(url.searchParams.get('offset') || '0')

    // Fetch games for user, ordered by most recent first
    const games = await DB.prepare(`
      SELECT id, outcome, moves, move_count, opponent_type, ai_difficulty, player_number, created_at
      FROM games
      WHERE user_id = ?
      ORDER BY created_at DESC
      LIMIT ? OFFSET ?
    `)
      .bind(session.userId, limit, offset)
      .all<GameRow>()

    // Get total count for pagination
    const countResult = await DB.prepare(`
      SELECT COUNT(*) as count FROM games WHERE user_id = ?
    `)
      .bind(session.userId)
      .first<{ count: number }>()

    const total = countResult?.count || 0

    // Parse moves JSON for each game
    const parsedGames = games.results.map((game) => ({
      id: game.id,
      outcome: game.outcome,
      moves: JSON.parse(game.moves),
      moveCount: game.move_count,
      opponentType: game.opponent_type,
      aiDifficulty: game.ai_difficulty,
      playerNumber: game.player_number,
      createdAt: game.created_at,
    }))

    return jsonResponse({
      games: parsedGames,
      pagination: {
        total,
        limit,
        offset,
        hasMore: offset + limit < total,
      },
    })
  } catch (error) {
    console.error('GET /api/games error:', error)
    return errorResponse('Internal server error', 500)
  }
}

/**
 * POST /api/games - Save a completed game
 */
export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Parse and validate request body
    const body = await context.request.json()
    const parseResult = createGameSchema.safeParse(body)

    if (!parseResult.success) {
      return errorResponse(parseResult.error.errors[0].message, 400)
    }

    const { outcome, moves, opponentType, aiDifficulty, playerNumber } = parseResult.data

    // Generate UUID for the game
    const gameId = crypto.randomUUID()
    const now = Date.now()

    // Insert the game
    await DB.prepare(`
      INSERT INTO games (id, user_id, outcome, moves, move_count, opponent_type, ai_difficulty, player_number, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `)
      .bind(
        gameId,
        session.userId,
        outcome,
        JSON.stringify(moves),
        moves.length,
        opponentType,
        aiDifficulty ?? null,
        playerNumber,
        now
      )
      .run()

    return jsonResponse(
      {
        id: gameId,
        outcome,
        moves,
        moveCount: moves.length,
        opponentType,
        aiDifficulty,
        playerNumber,
        createdAt: now,
      },
      201
    )
  } catch (error) {
    console.error('POST /api/games error:', error)
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
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  })
}
