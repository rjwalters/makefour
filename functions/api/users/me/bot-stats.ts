/**
 * Bot Stats API endpoint
 *
 * GET /api/users/me/bot-stats - Get player's stats against all bots
 */

import { validateSession, errorResponse, jsonResponse } from '../../../lib/auth'

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

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Get all bot stats for this user, joined with bot personas for names
    const stats = await DB.prepare(`
      SELECT
        pbs.user_id,
        pbs.bot_persona_id,
        pbs.wins,
        pbs.losses,
        pbs.draws,
        pbs.current_streak,
        pbs.best_win_streak,
        pbs.first_win_at,
        pbs.last_played_at,
        bp.name as bot_name,
        bp.play_style as bot_play_style,
        bp.current_elo as bot_current_elo
      FROM player_bot_stats pbs
      JOIN bot_personas bp ON pbs.bot_persona_id = bp.id
      WHERE pbs.user_id = ?
      ORDER BY pbs.last_played_at DESC
    `)
      .bind(session.userId)
      .all<PlayerBotStatsRow>()

    // Transform to response format
    const botStats: BotStatsResponse[] = stats.results.map((row) => {
      const totalGames = row.wins + row.losses + row.draws
      const winRate = totalGames > 0 ? (row.wins / totalGames) * 100 : 0

      return {
        botId: row.bot_persona_id,
        botName: row.bot_name,
        botPlayStyle: row.bot_play_style,
        botRating: row.bot_current_elo,
        wins: row.wins,
        losses: row.losses,
        draws: row.draws,
        totalGames,
        winRate: Math.round(winRate * 10) / 10,
        currentStreak: row.current_streak,
        bestWinStreak: row.best_win_streak,
        firstWinAt: row.first_win_at,
        lastPlayedAt: row.last_played_at,
        hasPositiveRecord: row.wins > row.losses,
        isUndefeated: row.losses === 0 && row.wins > 0,
        isMastered: row.wins >= 10 && winRate > 60,
      }
    })

    // Also get list of bots the user has never played
    const unplayedBots = await DB.prepare(`
      SELECT bp.id, bp.name, bp.play_style, bp.current_elo
      FROM bot_personas bp
      WHERE bp.is_active = 1
        AND bp.id NOT IN (
          SELECT bot_persona_id FROM player_bot_stats WHERE user_id = ?
        )
      ORDER BY bp.current_elo ASC
    `)
      .bind(session.userId)
      .all<{ id: string; name: string; play_style: string; current_elo: number }>()

    const unplayed = unplayedBots.results.map((bot) => ({
      botId: bot.id,
      botName: bot.name,
      botPlayStyle: bot.play_style,
      botRating: bot.current_elo,
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
