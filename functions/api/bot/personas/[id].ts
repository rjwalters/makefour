/**
 * Bot Persona by ID API endpoint
 *
 * GET /api/bot/personas/:id - Get a specific bot persona
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
}

/**
 * GET /api/bot/personas/:id - Get a specific bot persona
 */
export async function onRequestGet(context: EventContext<Env, any, { id: string }>) {
  const { DB } = context.env
  const { id } = context.params

  try {
    const persona = await DB.prepare(`
      SELECT id, name, description, avatar_url, ai_engine, ai_config,
             chat_personality, play_style, base_elo, current_elo,
             games_played, wins, losses, draws, is_active
      FROM bot_personas
      WHERE id = ? AND is_active = 1
    `)
      .bind(id)
      .first<BotPersonaRow>()

    if (!persona) {
      return errorResponse('Bot persona not found', 404)
    }

    return jsonResponse({
      id: persona.id,
      name: persona.name,
      description: persona.description,
      avatarUrl: persona.avatar_url,
      playStyle: persona.play_style,
      rating: persona.current_elo,
      gamesPlayed: persona.games_played,
      wins: persona.wins,
      losses: persona.losses,
      draws: persona.draws,
      winRate:
        persona.games_played > 0
          ? Math.round((persona.wins / persona.games_played) * 100)
          : 0,
      // Include AI config for game creation
      aiConfig: JSON.parse(persona.ai_config),
      chatPersonality: JSON.parse(persona.chat_personality),
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
