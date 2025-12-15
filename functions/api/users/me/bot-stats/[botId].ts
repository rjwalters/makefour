/**
 * Bot Stats API endpoint for specific bot
 *
 * GET /api/users/me/bot-stats/:botId - Get player's stats against a specific bot
 */

import { validateSession, errorResponse, jsonResponse } from '../../../../lib/auth'

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
  const botId = context.params.botId as string

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Get bot info
    const bot = await DB.prepare(`
      SELECT id, name, play_style, current_elo, description
      FROM bot_personas
      WHERE id = ? AND is_active = 1
    `)
      .bind(botId)
      .first<BotPersonaRow>()

    if (!bot) {
      return errorResponse('Bot not found', 404)
    }

    // Get stats for this user against this bot
    const stats = await DB.prepare(`
      SELECT wins, losses, draws, current_streak, best_win_streak, first_win_at, last_played_at
      FROM player_bot_stats
      WHERE user_id = ? AND bot_persona_id = ?
    `)
      .bind(session.userId, botId)
      .first<PlayerBotStatsRow>()

    // If no stats exist, return default values
    const wins = stats?.wins ?? 0
    const losses = stats?.losses ?? 0
    const draws = stats?.draws ?? 0
    const totalGames = wins + losses + draws
    const winRate = totalGames > 0 ? (wins / totalGames) * 100 : 0

    return jsonResponse({
      botId: bot.id,
      botName: bot.name,
      botPlayStyle: bot.play_style,
      botRating: bot.current_elo,
      botDescription: bot.description,
      wins,
      losses,
      draws,
      totalGames,
      winRate: Math.round(winRate * 10) / 10,
      currentStreak: stats?.current_streak ?? 0,
      bestWinStreak: stats?.best_win_streak ?? 0,
      firstWinAt: stats?.first_win_at ?? null,
      lastPlayedAt: stats?.last_played_at ?? null,
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
