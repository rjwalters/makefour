/**
 * Bot Persona by ID API endpoint
 *
 * GET /api/bot/personas/:id - Get a specific bot persona with full profile
 */

import { jsonResponse, errorResponse } from '../../../lib/auth'
import { type BotPersonaRow } from '../../../lib/types'
import { createDb } from '../../../../shared/db/client'
import { users, games, botPersonas, ratingHistory } from '../../../../shared/db/schema'
import { eq, and, sql, desc } from 'drizzle-orm'

interface Env {
  DB: D1Database
}

interface BotUserRow {
  rating: number
  games_played: number
  wins: number
  losses: number
  draws: number
}

interface RecentGameRow {
  id: string
  outcome: string
  move_count: number
  rating_change: number
  opponent_type: string
  created_at: number
}

interface MatchupStatsRow {
  opponent_id: string
  total_games: number
  wins: number
  losses: number
  draws: number
  avg_moves: number
  last_game_at: number
}

interface OpponentPersonaRow {
  id: string
  name: string
  avatar_url: string | null
}

/**
 * GET /api/bot/personas/:id - Get a specific bot persona with full profile
 */
export async function onRequestGet(context: EventContext<Env, any, { id: string }>) {
  const { DB } = context.env
  const { id } = context.params

  try {
    const db = createDb(DB)

    const persona = await db.query.botPersonas.findFirst({
      where: and(eq(botPersonas.id, id), eq(botPersonas.isActive, 1)),
    })

    if (!persona) {
      return errorResponse('Bot persona not found', 404)
    }

    // Get bot user stats (from users table - may have different values)
    const botUserId = `bot_${id}`
    const botUser = await db.query.users.findFirst({
      where: and(eq(users.id, botUserId), eq(users.isBot, 1)),
      columns: {
        rating: true,
        gamesPlayed: true,
        wins: true,
        losses: true,
        draws: true,
      },
    })

    // Get recent games for this bot
    const recentGames = await db.query.games.findMany({
      where: eq(games.userId, botUserId),
      orderBy: [desc(games.createdAt)],
      limit: 10,
      columns: {
        id: true,
        outcome: true,
        moveCount: true,
        ratingChange: true,
        opponentType: true,
        createdAt: true,
      },
    })

    // Get rating history
    const ratingHistoryData = await db.query.ratingHistory.findMany({
      where: eq(ratingHistory.userId, botUserId),
      orderBy: [desc(ratingHistory.createdAt)],
      limit: 20,
      columns: {
        ratingAfter: true,
        createdAt: true,
      },
    })

    // Get head-to-head matchup stats against other bots
    // Using raw SQL for aggregation since Drizzle doesn't have a clean way to do this yet
    const matchupStatsResult = await db.select({
      opponentId: games.opponentId,
      totalGames: sql<number>`COUNT(*)`,
      wins: sql<number>`SUM(CASE WHEN ${games.outcome} = 'win' THEN 1 ELSE 0 END)`,
      losses: sql<number>`SUM(CASE WHEN ${games.outcome} = 'loss' THEN 1 ELSE 0 END)`,
      draws: sql<number>`SUM(CASE WHEN ${games.outcome} = 'draw' THEN 1 ELSE 0 END)`,
      avgMoves: sql<number>`AVG(${games.moveCount})`,
      lastGameAt: sql<number>`MAX(${games.createdAt})`,
    })
    .from(games)
    .where(
      and(
        eq(games.userId, botUserId),
        sql`${games.opponentId} IS NOT NULL`,
        sql`${games.opponentId} LIKE 'bot_%'`
      )
    )
    .groupBy(games.opponentId)
    .orderBy(sql`COUNT(*) DESC`)

    // Get persona info for all opponents
    const opponentIds = matchupStatsResult.map(m => m.opponentId?.replace('bot_', '')).filter(Boolean) as string[]
    let opponentPersonas: OpponentPersonaRow[] = []
    if (opponentIds.length > 0) {
      opponentPersonas = await db.select({
        id: botPersonas.id,
        name: botPersonas.name,
        avatarUrl: botPersonas.avatarUrl,
      })
      .from(botPersonas)
      .where(sql`${botPersonas.id} IN ${sql.raw(`(${opponentIds.map(() => '?').join(',')})`, opponentIds)}`)
    }

    // Build matchups array with persona info
    const matchups = matchupStatsResult.map(stat => {
      const personaId = stat.opponentId?.replace('bot_', '') ?? ''
      const opponentPersona = opponentPersonas.find(p => p.id === personaId)
      const winRate = stat.totalGames > 0
        ? Math.round((stat.wins / stat.totalGames) * 100)
        : 0
      return {
        opponentId: personaId,
        opponentName: opponentPersona?.name ?? personaId,
        opponentAvatarUrl: opponentPersona?.avatarUrl ?? null,
        totalGames: stat.totalGames,
        wins: stat.wins,
        losses: stat.losses,
        draws: stat.draws,
        winRate,
        avgMoves: Math.round(stat.avgMoves),
        lastGameAt: stat.lastGameAt,
      }
    })

    // Use bot user stats if available, otherwise fall back to persona stats
    const stats = botUser || {
      rating: persona.currentElo,
      gamesPlayed: persona.gamesPlayed,
      wins: persona.wins,
      losses: persona.losses,
      draws: persona.draws,
    }

    return jsonResponse({
      id: persona.id,
      name: persona.name,
      description: persona.description,
      avatarUrl: persona.avatarUrl,
      playStyle: persona.playStyle,
      aiEngine: persona.aiEngine,
      baseElo: persona.baseElo,
      createdAt: persona.createdAt,
      // Stats from bot user (or persona as fallback)
      rating: stats.rating,
      gamesPlayed: stats.gamesPlayed,
      wins: stats.wins,
      losses: stats.losses,
      draws: stats.draws,
      winRate:
        stats.gamesPlayed > 0
          ? Math.round((stats.wins / stats.gamesPlayed) * 100)
          : 0,
      // Include AI config for game creation
      aiConfig: JSON.parse(persona.aiConfig),
      chatPersonality: JSON.parse(persona.chatPersonality),
      // Recent games
      recentGames: recentGames.map(game => ({
        id: game.id,
        outcome: game.outcome,
        moveCount: game.moveCount,
        ratingChange: game.ratingChange,
        opponentType: game.opponentType,
        createdAt: game.createdAt,
      })),
      // Rating history (reversed for chronological order)
      ratingHistory: ratingHistoryData.reverse().map(r => ({
        rating: r.ratingAfter,
        created_at: r.createdAt,
      })),
      // Head-to-head matchup records against other bots
      matchups,
    })
  } catch (error) {
    console.error('GET /api/bot/personas/:id error:', error)
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
