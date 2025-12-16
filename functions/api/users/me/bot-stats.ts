/**
 * Bot Stats API endpoint
 *
 * GET /api/users/me/bot-stats - Get player's stats against all bots
 */

import { validateSession, errorResponse, jsonResponse } from '../../../lib/auth'
import { createDb } from '../../../../shared/db/client'
import { playerBotStats, botPersonas } from '../../../../shared/db/schema'
import { eq, and, notInArray, sql } from 'drizzle-orm'

interface Env {
  DB: D1Database
}

interface PlayerBotStatsRow {
  user_id: string
  bot_persona_id: string
  wins: number
  losses: number
  draws: number
  current_streak: number
  best_win_streak: number
  first_win_at: number | null
  last_played_at: number
  // Joined from bot_personas
  bot_name: string
  bot_play_style: string
  bot_current_elo: number
}

interface BotStatsResponse {
  botId: string
  botName: string
  botPlayStyle: string
  botRating: number
  wins: number
  losses: number
  draws: number
  totalGames: number
  winRate: number
  currentStreak: number
  bestWinStreak: number
  firstWinAt: number | null
  lastPlayedAt: number
  // Derived status flags
  hasPositiveRecord: boolean
  isUndefeated: boolean
  isMastered: boolean // 10+ wins, >60% win rate
}

/**
 * GET /api/users/me/bot-stats - Get player's stats against all bots
 */
export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB } = context.env
  const db = createDb(DB)

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Get all bot stats for this user with bot persona info
    const stats = await db.query.playerBotStats.findMany({
      where: eq(playerBotStats.userId, session.userId),
      with: {
        botPersona: {
          columns: {
            name: true,
            playStyle: true,
            currentElo: true,
          },
        },
      },
      orderBy: (playerBotStats, { desc }) => [desc(playerBotStats.lastPlayedAt)],
    })

    // Transform to response format
    const botStats: BotStatsResponse[] = stats.map((row) => {
      const totalGames = row.wins + row.losses + row.draws
      const winRate = totalGames > 0 ? (row.wins / totalGames) * 100 : 0

      return {
        botId: row.botPersonaId,
        botName: row.botPersona.name,
        botPlayStyle: row.botPersona.playStyle,
        botRating: row.botPersona.currentElo,
        wins: row.wins,
        losses: row.losses,
        draws: row.draws,
        totalGames,
        winRate: Math.round(winRate * 10) / 10,
        currentStreak: row.currentStreak,
        bestWinStreak: row.bestWinStreak,
        firstWinAt: row.firstWinAt,
        lastPlayedAt: row.lastPlayedAt,
        hasPositiveRecord: row.wins > row.losses,
        isUndefeated: row.losses === 0 && row.wins > 0,
        isMastered: row.wins >= 10 && winRate > 60,
      }
    })

    // Get list of played bot IDs
    const playedBotIds = stats.map((s) => s.botPersonaId)

    // Get list of bots the user has never played
    const unplayedBotsQuery = playedBotIds.length > 0
      ? db.select({
          id: botPersonas.id,
          name: botPersonas.name,
          playStyle: botPersonas.playStyle,
          currentElo: botPersonas.currentElo,
        })
        .from(botPersonas)
        .where(and(
          eq(botPersonas.isActive, 1),
          notInArray(botPersonas.id, playedBotIds)
        ))
        .orderBy(botPersonas.currentElo)
      : db.select({
          id: botPersonas.id,
          name: botPersonas.name,
          playStyle: botPersonas.playStyle,
          currentElo: botPersonas.currentElo,
        })
        .from(botPersonas)
        .where(eq(botPersonas.isActive, 1))
        .orderBy(botPersonas.currentElo)

    const unplayedBots = await unplayedBotsQuery

    const unplayed = unplayedBots.map((bot) => ({
      botId: bot.id,
      botName: bot.name,
      botPlayStyle: bot.playStyle,
      botRating: bot.currentElo,
      wins: 0,
      losses: 0,
      draws: 0,
      totalGames: 0,
      winRate: 0,
      currentStreak: 0,
      bestWinStreak: 0,
      firstWinAt: null,
      lastPlayedAt: null,
      hasPositiveRecord: false,
      isUndefeated: false,
      isMastered: false,
    }))

    // Calculate summary stats
    const totalWins = botStats.reduce((sum, s) => sum + s.wins, 0)
    const totalLosses = botStats.reduce((sum, s) => sum + s.losses, 0)
    const totalDraws = botStats.reduce((sum, s) => sum + s.draws, 0)
    const totalGames = totalWins + totalLosses + totalDraws
    const botsPlayed = botStats.length
    const botsDefeated = botStats.filter((s) => s.wins > 0).length
    const botsMastered = botStats.filter((s) => s.isMastered).length

    return jsonResponse({
      stats: botStats,
      unplayed,
      summary: {
        totalGames,
        totalWins,
        totalLosses,
        totalDraws,
        overallWinRate: totalGames > 0 ? Math.round((totalWins / totalGames) * 1000) / 10 : 0,
        botsPlayed,
        botsDefeated,
        botsMastered,
        botsRemaining: unplayed.length,
      },
    })
  } catch (error) {
    console.error('GET /api/users/me/bot-stats error:', error)
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
