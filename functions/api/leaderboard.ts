/**
 * Leaderboard API endpoint
 *
 * GET /api/leaderboard - Get top players by ELO rating
 */

import { eq, and, or, gt, desc, asc, inArray, count, sql } from 'drizzle-orm'
import { createDb } from '../../shared/db/client'
import { users, botPersonas } from '../../shared/db/schema'
import { jsonResponse, errorResponse, validateSession } from '../lib/auth'

interface Env {
  DB: D1Database
}

// Schema for leaderboard entry from database
interface LeaderboardRow {
  id: string
  email: string
  username: string | null
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
  avatar_url: string | null
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
  const db = createDb(DB)

  try {
    // Get query params for pagination and filtering
    const url = new URL(context.request.url)
    const limit = Math.min(parseInt(url.searchParams.get('limit') || '50'), 100)
    const offset = parseInt(url.searchParams.get('offset') || '0')
    const includeBots = url.searchParams.get('includeBots') !== 'false' // default true

    // Build WHERE condition based on whether to include bots
    // Bots always appear (regardless of games played), humans need gamesPlayed > 0 and verified email
    const whereCondition = includeBots
      ? or(
          // Humans: must have played games and be verified
          and(gt(users.gamesPlayed, 0), eq(users.emailVerified, 1), eq(users.isBot, 0)),
          // Bots: always show (no games played requirement)
          eq(users.isBot, 1)
        )
      : and(
          gt(users.gamesPlayed, 0),
          eq(users.emailVerified, 1),
          eq(users.isBot, 0)
        )

    // Fetch top players by rating (verified users with games, plus all bots)
    const players = await db
      .select({
        id: users.id,
        email: users.email,
        username: users.username,
        rating: users.rating,
        games_played: users.gamesPlayed,
        wins: users.wins,
        losses: users.losses,
        draws: users.draws,
        is_bot: users.isBot,
        bot_persona_id: users.botPersonaId,
      })
      .from(users)
      .where(whereCondition)
      .orderBy(desc(users.rating), desc(users.wins))
      .limit(limit)
      .offset(offset)

    // Get bot persona info for bots in the results
    const botPersonaIds = players
      .filter(p => p.is_bot && p.bot_persona_id)
      .map(p => p.bot_persona_id!)

    // Fetch bot persona details if there are bots
    const botPersonaMap = new Map<string, BotPersonaInfo>()
    if (botPersonaIds.length > 0) {
      const personas = await db
        .select({
          id: botPersonas.id,
          name: botPersonas.name,
          description: botPersonas.description,
          avatar_url: botPersonas.avatarUrl,
        })
        .from(botPersonas)
        .where(inArray(botPersonas.id, botPersonaIds))

      for (const persona of personas) {
        botPersonaMap.set(persona.id, {
          name: persona.name,
          description: persona.description,
          avatar_url: persona.avatar_url,
        })
      }
    }

    // Get total count for pagination
    const countResult = await db
      .select({ count: count() })
      .from(users)
      .where(whereCondition)

    const total = countResult[0]?.count || 0

    // Map to public leaderboard entries
    const leaderboard = players.map((player, index) => {
      const isBot = player.is_bot === 1
      const personaInfo = player.bot_persona_id ? botPersonaMap.get(player.bot_persona_id) : null

      return {
        rank: offset + index + 1,
        userId: player.id,
        username: isBot && personaInfo
          ? personaInfo.name
          : player.username || player.email.split('@')[0], // Prefer username, fall back to email prefix
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
        botAvatarUrl: isBot && personaInfo ? personaInfo.avatar_url : null,
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
        const userRow = await db.query.users.findFirst({
          columns: {
            id: true,
            email: true,
            username: true,
            rating: true,
            gamesPlayed: true,
            wins: true,
            losses: true,
            draws: true,
            isBot: true,
            botPersonaId: true,
          },
          where: and(
            eq(users.id, userId),
            or(
              // Humans: must have played games and be verified
              and(gt(users.gamesPlayed, 0), eq(users.emailVerified, 1), eq(users.isBot, 0)),
              // Bots: always eligible
              eq(users.isBot, 1)
            )
          ),
        })

        if (userRow) {
          // Calculate global rank (includes all bots + verified humans with games)
          const rankResult = await db
            .select({
              rank: sql<number>`COUNT(*) + 1`,
            })
            .from(users)
            .where(
              and(
                gt(users.rating, userRow.rating),
                or(
                  // Humans: must have played games and be verified
                  and(gt(users.gamesPlayed, 0), eq(users.emailVerified, 1), eq(users.isBot, 0)),
                  // Bots: always counted
                  eq(users.isBot, 1)
                )
              )
            )

          const userRank = rankResult[0]?.rank || 1

          // Only include currentUser if they're outside the displayed range
          if (userRank > 50) {
            const isBot = userRow.isBot === 1
            const personaInfo = userRow.botPersonaId
              ? botPersonaMap.get(userRow.botPersonaId)
              : null

            currentUser = {
              rank: userRank,
              entry: {
                rank: userRank,
                userId: userRow.id,
                username: isBot && personaInfo
                  ? personaInfo.name
                  : userRow.username || userRow.email.split('@')[0],
                rating: userRow.rating,
                gamesPlayed: userRow.gamesPlayed,
                wins: userRow.wins,
                losses: userRow.losses,
                draws: userRow.draws,
                winRate:
                  userRow.gamesPlayed > 0
                    ? Math.round((userRow.wins / userRow.gamesPlayed) * 100)
                    : 0,
                isBot,
                botPersonaId: isBot ? userRow.botPersonaId : null,
                botDescription: isBot && personaInfo ? personaInfo.description : null,
              },
            }
          }
        }
      }
    }

    const response = jsonResponse({
      leaderboard,
      pagination: {
        total,
        limit,
        offset,
        hasMore: offset + limit < total,
      },
      currentUser,
    })
    // Prevent CDN caching to ensure fresh data for all users
    response.headers.set('Cache-Control', 'no-store, no-cache, must-revalidate')
    return response
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
