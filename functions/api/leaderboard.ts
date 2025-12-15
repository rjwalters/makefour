/**
 * Leaderboard API endpoint
 *
 * GET /api/leaderboard - Get top players by ELO rating
 */

import { jsonResponse, errorResponse } from '../lib/auth'

interface Env {
  DB: D1Database
}

// Schema for leaderboard entry from database
interface LeaderboardRow {
  id: string
  email: string
  rating: number
  games_played: number
  wins: number
  losses: number
  draws: number
}

/**
 * GET /api/leaderboard - Get top players by ELO rating
 *
 * Query params:
 * - limit: number of players to return (default 50, max 100)
 * - offset: pagination offset (default 0)
 */
export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    // Get query params for pagination
    const url = new URL(context.request.url)
    const limit = Math.min(parseInt(url.searchParams.get('limit') || '50'), 100)
    const offset = parseInt(url.searchParams.get('offset') || '0')

    // Fetch top players by rating (only verified users who have played at least 1 game)
    const players = await DB.prepare(`
      SELECT id, email, rating, games_played, wins, losses, draws
      FROM users
      WHERE games_played > 0 AND email_verified = 1
      ORDER BY rating DESC, wins DESC
      LIMIT ? OFFSET ?
    `)
      .bind(limit, offset)
      .all<LeaderboardRow>()

    // Get total count for pagination (only verified users)
    const countResult = await DB.prepare(`
      SELECT COUNT(*) as count FROM users WHERE games_played > 0 AND email_verified = 1
    `).first<{ count: number }>()

    const total = countResult?.count || 0

    // Map to public leaderboard entries (hide email, show username portion)
    const leaderboard = players.results.map((player, index) => ({
      rank: offset + index + 1,
      userId: player.id,
      username: player.email.split('@')[0], // Use email prefix as username
      rating: player.rating,
      gamesPlayed: player.games_played,
      wins: player.wins,
      losses: player.losses,
      draws: player.draws,
      winRate:
        player.games_played > 0
          ? Math.round((player.wins / player.games_played) * 100)
          : 0,
    }))

    return jsonResponse({
      leaderboard,
      pagination: {
        total,
        limit,
        offset,
        hasMore: offset + limit < total,
      },
    })
  } catch (error) {
    console.error('GET /api/leaderboard error:', error)
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
