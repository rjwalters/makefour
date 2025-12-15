/**
 * Bot Persona by ID API endpoint
 *
 * GET /api/bot/personas/:id - Get a specific bot persona with full profile
 */

import { jsonResponse, errorResponse } from '../../../lib/auth'

interface Env {
  DB: D1Database
}

// Schema for bot persona from database
interface BotPersonaRow {
  id: string
  name: string
  description: string
  avatar_url: string | null
  ai_engine: string
  ai_config: string
  chat_personality: string
  play_style: string
  base_elo: number
  current_elo: number
  games_played: number
  wins: number
  losses: number
  draws: number
  is_active: number
  created_at: number
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
    const persona = await DB.prepare(`
      SELECT id, name, description, avatar_url, ai_engine, ai_config,
             chat_personality, play_style, base_elo, current_elo,
             games_played, wins, losses, draws, is_active, created_at
      FROM bot_personas
      WHERE id = ? AND is_active = 1
    `)
      .bind(id)
      .first<BotPersonaRow>()

    if (!persona) {
      return errorResponse('Bot persona not found', 404)
    }

    // Get bot user stats (from users table - may have different values)
    const botUserId = `bot_${id}`
    const botUser = await DB.prepare(`
      SELECT rating, games_played, wins, losses, draws
      FROM users
      WHERE id = ? AND is_bot = 1
    `)
      .bind(botUserId)
      .first<BotUserRow>()

    // Get recent games for this bot
    const recentGames = await DB.prepare(`
      SELECT id, outcome, move_count, rating_change, opponent_type, created_at
      FROM games
      WHERE user_id = ?
      ORDER BY created_at DESC
      LIMIT 10
    `)
      .bind(botUserId)
      .all<RecentGameRow>()

    // Get rating history
    const ratingHistory = await DB.prepare(`
      SELECT rating_after as rating, created_at
      FROM rating_history
      WHERE user_id = ?
      ORDER BY created_at DESC
      LIMIT 20
    `)
      .bind(botUserId)
      .all<{ rating: number; created_at: number }>()

    // Get head-to-head matchup stats against other bots
    const matchupStats = await DB.prepare(`
      SELECT
        opponent_id,
        COUNT(*) as total_games,
        SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
        SUM(CASE WHEN outcome = 'draw' THEN 1 ELSE 0 END) as draws,
        AVG(move_count) as avg_moves,
        MAX(created_at) as last_game_at
      FROM games
      WHERE user_id = ?
        AND opponent_id IS NOT NULL
        AND opponent_id LIKE 'bot_%'
      GROUP BY opponent_id
      ORDER BY total_games DESC
    `)
      .bind(botUserId)
      .all<MatchupStatsRow>()

    // Get persona info for all opponents
    const opponentIds = matchupStats.results.map(m => m.opponent_id.replace('bot_', ''))
    let opponentPersonas: OpponentPersonaRow[] = []
    if (opponentIds.length > 0) {
      const placeholders = opponentIds.map(() => '?').join(',')
      const opponentResult = await DB.prepare(`
        SELECT id, name, avatar_url
        FROM bot_personas
        WHERE id IN (${placeholders})
      `)
        .bind(...opponentIds)
        .all<OpponentPersonaRow>()
      opponentPersonas = opponentResult.results
    }

    // Build matchups array with persona info
    const matchups = matchupStats.results.map(stat => {
      const personaId = stat.opponent_id.replace('bot_', '')
      const opponentPersona = opponentPersonas.find(p => p.id === personaId)
      const winRate = stat.total_games > 0
        ? Math.round((stat.wins / stat.total_games) * 100)
        : 0
      return {
        opponentId: personaId,
        opponentName: opponentPersona?.name ?? personaId,
        opponentAvatarUrl: opponentPersona?.avatar_url ?? null,
        totalGames: stat.total_games,
        wins: stat.wins,
        losses: stat.losses,
        draws: stat.draws,
        winRate,
        avgMoves: Math.round(stat.avg_moves),
        lastGameAt: stat.last_game_at,
      }
    })

    // Use bot user stats if available, otherwise fall back to persona stats
    const stats = botUser || {
      rating: persona.current_elo,
      games_played: persona.games_played,
      wins: persona.wins,
      losses: persona.losses,
      draws: persona.draws,
    }

    return jsonResponse({
      id: persona.id,
      name: persona.name,
      description: persona.description,
      avatarUrl: persona.avatar_url,
      playStyle: persona.play_style,
      baseElo: persona.base_elo,
      createdAt: persona.created_at,
      // Stats from bot user (or persona as fallback)
      rating: stats.rating,
      gamesPlayed: stats.games_played,
      wins: stats.wins,
      losses: stats.losses,
      draws: stats.draws,
      winRate:
        stats.games_played > 0
          ? Math.round((stats.wins / stats.games_played) * 100)
          : 0,
      // Include AI config for game creation
      aiConfig: JSON.parse(persona.ai_config),
      chatPersonality: JSON.parse(persona.chat_personality),
      // Recent games
      recentGames: recentGames.results.map(game => ({
        id: game.id,
        outcome: game.outcome,
        moveCount: game.move_count,
        ratingChange: game.rating_change,
        opponentType: game.opponent_type,
        createdAt: game.created_at,
      })),
      // Rating history (reversed for chronological order)
      ratingHistory: ratingHistory.results.reverse(),
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
