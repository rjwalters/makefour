/**
 * Spectator Chat API endpoint
 *
 * GET /api/games/:id/chat - Get chat messages for spectating (no auth required for spectatable games)
 *
 * This endpoint allows spectators to view chat messages in real-time during
 * bot vs bot games and other spectatable matches.
 */

import { jsonResponse } from '../../../lib/auth'

interface Env {
  DB: D1Database
}

interface GameMessageRow {
  id: string
  game_id: string
  sender_id: string
  sender_type: 'human' | 'bot'
  content: string
  created_at: number
}

interface ActiveGameRow {
  id: string
  spectatable: number
  status: string
  is_bot_vs_bot: number
  bot1_persona_id: string | null
  bot2_persona_id: string | null
}

interface BotPersonaRow {
  id: string
  name: string
}

export interface SpectatorChatMessage {
  id: string
  senderName: string
  senderType: 'human' | 'bot'
  content: string
  createdAt: number
}

/**
 * GET /api/games/:id/chat - Get chat messages for spectating
 *
 * No authentication required for spectatable games.
 * Returns messages with sender names resolved.
 *
 * Query parameters:
 * - since: timestamp (only return messages after this time)
 * - limit: number (max 100, default 50)
 */
export async function onRequestGet(context: EventContext<Env, any, { id: string }>) {
  const { DB } = context.env
  const gameId = context.params.id
  const url = new URL(context.request.url)

  const since = parseInt(url.searchParams.get('since') || '0', 10)
  const limit = Math.min(parseInt(url.searchParams.get('limit') || '50', 10), 100)

  try {
    // Get the game to verify it's spectatable
    const game = await DB.prepare(`
      SELECT id, spectatable, status, is_bot_vs_bot, bot1_persona_id, bot2_persona_id
      FROM active_games
      WHERE id = ?
    `)
      .bind(gameId)
      .first<ActiveGameRow>()

    if (!game) {
      return jsonResponse({ error: 'Game not found' }, { status: 404 })
    }

    if (game.spectatable !== 1) {
      return jsonResponse({ error: 'This game is not available for spectating' }, { status: 403 })
    }

    // Get bot persona names for resolving sender names
    const botNames: Record<string, string> = {}

    if (game.is_bot_vs_bot === 1) {
      if (game.bot1_persona_id) {
        const bot1 = await DB.prepare('SELECT id, name FROM bot_personas WHERE id = ?')
          .bind(game.bot1_persona_id)
          .first<BotPersonaRow>()
        if (bot1) {
          botNames[`bot_${game.bot1_persona_id}`] = bot1.name
        }
      }
      if (game.bot2_persona_id) {
        const bot2 = await DB.prepare('SELECT id, name FROM bot_personas WHERE id = ?')
          .bind(game.bot2_persona_id)
          .first<BotPersonaRow>()
        if (bot2) {
          botNames[`bot_${game.bot2_persona_id}`] = bot2.name
        }
      }
    }

    // Get messages
    const messages = await DB.prepare(`
      SELECT id, game_id, sender_id, sender_type, content, created_at
      FROM game_messages
      WHERE game_id = ? AND created_at > ?
      ORDER BY created_at ASC
      LIMIT ?
    `)
      .bind(gameId, since, limit)
      .all<GameMessageRow>()

    // Transform messages with resolved sender names
    const chatMessages: SpectatorChatMessage[] = (messages.results || []).map((msg) => {
      let senderName = 'Player'

      if (msg.sender_type === 'bot') {
        // Check if we have a bot name for this sender
        senderName = botNames[msg.sender_id] || 'Bot'
      } else {
        // For human players, use a generic name for privacy
        senderName = 'Player'
      }

      return {
        id: msg.id,
        senderName,
        senderType: msg.sender_type,
        content: msg.content,
        createdAt: msg.created_at,
      }
    })

    return jsonResponse({
      messages: chatMessages,
      gameStatus: game.status,
    })
  } catch (error) {
    console.error('GET /api/games/:id/chat error:', error)
    return jsonResponse({ error: 'Internal server error' }, { status: 500 })
  }
}

export async function onRequestOptions() {
  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  })
}
