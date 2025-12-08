/**
 * Single game API endpoint
 *
 * GET /api/games/:id - Get a specific game by ID
 */

import { validateSession, errorResponse, jsonResponse } from '../../lib/auth'

interface Env {
  DB: D1Database
}

interface GameRow {
  id: string
  user_id: string
  outcome: string
  moves: string
  move_count: number
  created_at: number
}

/**
 * GET /api/games/:id - Get a specific game
 */
export async function onRequestGet(context: EventContext<Env, any, { id: string }>) {
  const { DB } = context.env
  const { id: gameId } = context.params

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Fetch the game (only if it belongs to the user)
    const game = await DB.prepare(`
      SELECT id, outcome, moves, move_count, created_at
      FROM games
      WHERE id = ? AND user_id = ?
    `)
      .bind(gameId, session.userId)
      .first<GameRow>()

    if (!game) {
      return errorResponse('Game not found', 404)
    }

    return jsonResponse({
      id: game.id,
      outcome: game.outcome,
      moves: JSON.parse(game.moves),
      moveCount: game.move_count,
      createdAt: game.created_at,
    })
  } catch (error) {
    console.error('GET /api/games/:id error:', error)
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
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  })
}
