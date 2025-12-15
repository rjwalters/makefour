/**
 * Live games API endpoint
 *
 * GET /api/games/live - Get list of active spectatable games
 */

import { jsonResponse } from '../../lib/auth'

interface Env {
  DB: D1Database
}

interface LiveGameRow {
  id: string
  player1_id: string
  player2_id: string
  moves: string
  current_turn: number
  status: string
  mode: string
  player1_rating: number
  player2_rating: number
  spectator_count: number
  created_at: number
  updated_at: number
  player1_email: string
  player2_email: string
}

export interface LiveGame {
  id: string
  player1: {
    rating: number
    displayName: string
  }
  player2: {
    rating: number
    displayName: string
  }
  moveCount: number
  currentTurn: 1 | 2
  mode: 'ranked' | 'casual'
  spectatorCount: number
  createdAt: number
  updatedAt: number
}

/**
 * GET /api/games/live - Get list of active spectatable games
 *
 * Query parameters:
 * - limit: number (default 20, max 50)
 * - offset: number (default 0)
 * - minRating: number (filter by minimum average rating)
 * - maxRating: number (filter by maximum average rating)
 * - mode: 'ranked' | 'casual' (filter by game mode)
 */
export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB } = context.env
  const url = new URL(context.request.url)

  // Parse query parameters
  const limit = Math.min(Math.max(parseInt(url.searchParams.get('limit') || '20'), 1), 50)
  const offset = Math.max(parseInt(url.searchParams.get('offset') || '0'), 0)
  const minRating = url.searchParams.get('minRating')
  const maxRating = url.searchParams.get('maxRating')
  const mode = url.searchParams.get('mode')

  try {
    // Build query with optional filters
    let whereClause = 'ag.status = ? AND ag.spectatable = 1'
    const params: (string | number)[] = ['active']

    if (minRating) {
      whereClause += ' AND (ag.player1_rating + ag.player2_rating) / 2 >= ?'
      params.push(parseInt(minRating))
    }

    if (maxRating) {
      whereClause += ' AND (ag.player1_rating + ag.player2_rating) / 2 <= ?'
      params.push(parseInt(maxRating))
    }

    if (mode && (mode === 'ranked' || mode === 'casual')) {
      whereClause += ' AND ag.mode = ?'
      params.push(mode)
    }

    // Get total count for pagination
    const countResult = await DB.prepare(`
      SELECT COUNT(*) as total
      FROM active_games ag
      WHERE ${whereClause}
    `)
      .bind(...params)
      .first<{ total: number }>()

    const total = countResult?.total ?? 0

    // Get games with player info
    const games = await DB.prepare(`
      SELECT
        ag.id,
        ag.player1_id,
        ag.player2_id,
        ag.moves,
        ag.current_turn,
        ag.status,
        ag.mode,
        ag.player1_rating,
        ag.player2_rating,
        ag.spectator_count,
        ag.created_at,
        ag.updated_at,
        u1.email as player1_email,
        u2.email as player2_email
      FROM active_games ag
      JOIN users u1 ON ag.player1_id = u1.id
      JOIN users u2 ON ag.player2_id = u2.id
      WHERE ${whereClause}
      ORDER BY ag.spectator_count DESC, ag.updated_at DESC
      LIMIT ? OFFSET ?
    `)
      .bind(...params, limit, offset)
      .all<LiveGameRow>()

    // Transform to response format (hide full email for privacy)
    const liveGames: LiveGame[] = games.results.map((game) => {
      const moves = JSON.parse(game.moves) as number[]
      return {
        id: game.id,
        player1: {
          rating: game.player1_rating,
          displayName: maskEmail(game.player1_email),
        },
        player2: {
          rating: game.player2_rating,
          displayName: maskEmail(game.player2_email),
        },
        moveCount: moves.length,
        currentTurn: game.current_turn as 1 | 2,
        mode: game.mode as 'ranked' | 'casual',
        spectatorCount: game.spectator_count,
        createdAt: game.created_at,
        updatedAt: game.updated_at,
      }
    })

    return jsonResponse({
      games: liveGames,
      total,
      limit,
      offset,
    })
  } catch (error) {
    console.error('GET /api/games/live error:', error)
    return jsonResponse({ error: 'Internal server error' }, { status: 500 })
  }
}

/**
 * Mask email for display (e.g., "user@example.com" -> "us***@example.com")
 */
function maskEmail(email: string): string {
  const [local, domain] = email.split('@')
  if (!domain) return email
  if (local.length <= 2) return `${local}***@${domain}`
  return `${local.slice(0, 2)}***@${domain}`
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
