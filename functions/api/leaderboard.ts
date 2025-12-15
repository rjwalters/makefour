/**
 * Leaderboard API endpoint
 *
 * GET /api/leaderboard - Get top players by ELO rating
 */

import { jsonResponse, errorResponse, validateSession } from '../lib/auth'

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
  is_bot: number
  bot_persona_id: string | null
}

// Bot persona info for joining
interface BotPersonaInfo {
  name: string
  description: string
}

/**
 * GET /api/leaderboard - Get top players by ELO rating
 *
 * Query params:
 * - limit: number of players to return (default 50, max 100)
 * - offset: pagination offset (default 0)
 * - includeBots: whether to include bot players (default true)
 */
export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    // Get query params for pagination and filtering
    const url = new URL(context.request.url)
    const limit = Math.min(parseInt(url.searchParams.get('limit') || '50'), 100)
    const offset = parseInt(url.searchParams.get('offset') || '0')
    const includeBots = url.searchParams.get('includeBots') !== 'false' // default true

    // Build WHERE clause based on whether to include bots
    const whereClause = includeBots
      ? `WHERE games_played > 0 AND (email_verified = 1 OR is_bot = 1)`
      : `WHERE games_played > 0 AND email_verified = 1 AND is_bot = 0`

    // Fetch top players by rating (verified users and optionally bots who have played at least 1 game)
    const players = await DB.prepare(`
      SELECT u.id, u.email, u.rating, u.games_played, u.wins, u.losses, u.draws,
             u.is_bot, u.bot_persona_id
      FROM users u
      ${whereClause}
      ORDER BY u.rating DESC, u.wins DESC
      LIMIT ? OFFSET ?
    `)
      .bind(limit, offset)
      .all<LeaderboardRow>()

    // Get bot persona info for bots in the results
    const botPersonaIds = players.results
      .filter(p => p.is_bot && p.bot_persona_id)
      .map(p => p.bot_persona_id)

    // Fetch bot persona details if there are bots
    const botPersonaMap = new Map<string, BotPersonaInfo>()
    if (botPersonaIds.length > 0) {
      const placeholders = botPersonaIds.map(() => '?').join(',')
      const personas = await DB.prepare(`
        SELECT id, name, description FROM bot_personas WHERE id IN (${placeholders})
      `)
        .bind(...botPersonaIds)
        .all<{ id: string; name: string; description: string }>()

      for (const persona of personas.results) {
        botPersonaMap.set(persona.id, { name: persona.name, description: persona.description })
      }
    }

    // Get total count for pagination
    const countResult = await DB.prepare(`
      SELECT COUNT(*) as count FROM users ${whereClause}
    `).first<{ count: number }>()

    const total = countResult?.count || 0

    // Map to public leaderboard entries
    const leaderboard = players.results.map((player, index) => {
      const isBot = player.is_bot === 1
      const personaInfo = player.bot_persona_id ? botPersonaMap.get(player.bot_persona_id) : null

      return {
        rank: offset + index + 1,
        userId: player.id,
        username: isBot && personaInfo
          ? personaInfo.name
          : player.email.split('@')[0], // Use email prefix as username for humans
        rating: player.rating,
        gamesPlayed: player.games_played,
        wins: player.wins,
        losses: player.losses,
        draws: player.draws,
        winRate:
          player.games_played > 0
            ? Math.round((player.wins / player.games_played) * 100)
            : 0,
        // Bot-specific fields
        isBot,
        botPersonaId: isBot ? player.bot_persona_id : null,
        botDescription: isBot && personaInfo ? personaInfo.description : null,
      }
    })

    // Check if user is authenticated (optional - don't require auth for leaderboard)
    let currentUser: {
      rank: number
      entry: typeof leaderboard[0]
    } | null = null

    const sessionResult = await validateSession(context.request, DB)
    if (sessionResult.valid) {
      const userId = sessionResult.userId

      // Check if user is already in the top 50 (on first page only)
      const userInList = offset === 0 && leaderboard.some((e) => e.userId === userId)

      if (!userInList) {
        // Get user's data and calculate global rank (always includes bots)
        const userRow = await DB.prepare(`
          SELECT u.id, u.email, u.rating, u.games_played, u.wins, u.losses, u.draws,
                 u.is_bot, u.bot_persona_id
          FROM users u
          WHERE u.id = ? AND u.games_played > 0 AND (u.email_verified = 1 OR u.is_bot = 1)
        `)
          .bind(userId)
          .first<LeaderboardRow>()

        if (userRow) {
          // Calculate global rank (always includes bots for consistent ranking)
          const rankResult = await DB.prepare(`
            SELECT COUNT(*) + 1 as rank
            FROM users
            WHERE rating > ?
              AND games_played > 0
              AND (email_verified = 1 OR is_bot = 1)
          `)
            .bind(userRow.rating)
            .first<{ rank: number }>()

          const userRank = rankResult?.rank || 1

          // Only include currentUser if they're outside the displayed range
          if (userRank > 50) {
            const isBot = userRow.is_bot === 1
            const personaInfo = userRow.bot_persona_id
              ? botPersonaMap.get(userRow.bot_persona_id)
              : null

            currentUser = {
              rank: userRank,
              entry: {
                rank: userRank,
                userId: userRow.id,
                username: isBot && personaInfo
                  ? personaInfo.name
                  : userRow.email.split('@')[0],
                rating: userRow.rating,
                gamesPlayed: userRow.games_played,
                wins: userRow.wins,
                losses: userRow.losses,
                draws: userRow.draws,
                winRate:
                  userRow.games_played > 0
                    ? Math.round((userRow.wins / userRow.games_played) * 100)
                    : 0,
                isBot,
                botPersonaId: isBot ? userRow.bot_persona_id : null,
                botDescription: isBot && personaInfo ? personaInfo.description : null,
              },
            }
          }
        }
      }
    }

    return jsonResponse({
      leaderboard,
      pagination: {
        total,
        limit,
        offset,
        hasMore: offset + limit < total,
      },
      currentUser,
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
