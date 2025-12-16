/**
 * Bot Personas API endpoint
 *
 * GET /api/bot/personas - Get all active bot personas
 */

import { eq, asc } from 'drizzle-orm'
import { createDb } from '../../../shared/db/client'
import { botPersonas } from '../../../shared/db/schema'
import { jsonResponse, errorResponse } from '../../lib/auth'

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

// Public persona for API response
interface PublicBotPersona {
  id: string
  name: string
  description: string
  avatarUrl: string | null
  playStyle: string
  rating: number
  gamesPlayed: number
  wins: number
  losses: number
  draws: number
  winRate: number
}

/**
 * GET /api/bot/personas - Get all active bot personas
 *
 * Returns a list of all active bot personas, sorted by rating
 */
export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB } = context.env
  const db = createDb(DB)

  try {
    // Fetch all active personas, ordered by current ELO
    const personas = await db
      .select({
        id: botPersonas.id,
        name: botPersonas.name,
        description: botPersonas.description,
        avatar_url: botPersonas.avatarUrl,
        ai_engine: botPersonas.aiEngine,
        ai_config: botPersonas.aiConfig,
        chat_personality: botPersonas.chatPersonality,
        play_style: botPersonas.playStyle,
        base_elo: botPersonas.baseElo,
        current_elo: botPersonas.currentElo,
        games_played: botPersonas.gamesPlayed,
        wins: botPersonas.wins,
        losses: botPersonas.losses,
        draws: botPersonas.draws,
        is_active: botPersonas.isActive,
      })
      .from(botPersonas)
      .where(eq(botPersonas.isActive, 1))
      .orderBy(asc(botPersonas.currentElo))

    // Map to public persona format
    const publicPersonas: PublicBotPersona[] = personas.map((persona) => ({
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
    }))

    return jsonResponse({
      personas: publicPersonas,
    })
  } catch (error) {
    console.error('GET /api/bot/personas error:', error)
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
