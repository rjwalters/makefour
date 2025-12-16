/**
 * Bot Stats API endpoint for specific bot
 *
 * GET /api/users/me/bot-stats/:botId - Get player's stats against a specific bot
 */

import { validateSession, errorResponse, jsonResponse } from '../../../../lib/auth'
import { createDb } from '../../../../../shared/db/client'
import { playerBotStats, botPersonas } from '../../../../../shared/db/schema'
import { eq, and } from 'drizzle-orm'

interface Env {
  DB: D1Database
}

interface PlayerBotStatsRow {
  wins: number
  losses: number
  draws: number
  current_streak: number
  best_win_streak: number
  first_win_at: number | null
  last_played_at: number
}

interface BotPersonaRow {
  id: string
  name: string
  play_style: string
  current_elo: number
  description: string
}

/**
 * GET /api/users/me/bot-stats/:botId - Get player's stats against a specific bot
 */
export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB } = context.env
  const db = createDb(DB)
  const botId = context.params.botId as string

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Get bot info
    const bot = await db.query.botPersonas.findFirst({
      where: and(
        eq(botPersonas.id, botId),
        eq(botPersonas.isActive, 1)
      ),
      columns: {
        id: true,
        name: true,
        playStyle: true,
        currentElo: true,
        description: true,
      },
    })

    if (!bot) {
      return errorResponse('Bot not found', 404)
    }

    // Get stats for this user against this bot
    const stats = await db.query.playerBotStats.findFirst({
      where: and(
        eq(playerBotStats.userId, session.userId),
        eq(playerBotStats.botPersonaId, botId)
      ),
    })

    // If no stats exist, return default values
    const wins = stats?.wins ?? 0
    const losses = stats?.losses ?? 0
    const draws = stats?.draws ?? 0
    const totalGames = wins + losses + draws
    const winRate = totalGames > 0 ? (wins / totalGames) * 100 : 0

    return jsonResponse({
      botId: bot.id,
      botName: bot.name,
      botPlayStyle: bot.playStyle,
      botRating: bot.currentElo,
      botDescription: bot.description,
      wins,
      losses,
      draws,
      totalGames,
      winRate: Math.round(winRate * 10) / 10,
      currentStreak: stats?.currentStreak ?? 0,
      bestWinStreak: stats?.bestWinStreak ?? 0,
      firstWinAt: stats?.firstWinAt ?? null,
      lastPlayedAt: stats?.lastPlayedAt ?? null,
      hasPositiveRecord: wins > losses,
      isUndefeated: losses === 0 && wins > 0,
      isMastered: wins >= 10 && winRate > 60,
      hasPlayed: totalGames > 0,
    })
  } catch (error) {
    console.error('GET /api/users/me/bot-stats/:botId error:', error)
    return errorResponse('Internal server error', 500)
  }
}

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
