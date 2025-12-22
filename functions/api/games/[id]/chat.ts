/**
 * Spectator Chat API endpoint
 *
 * GET /api/games/:id/chat - Get chat messages for spectating (no auth required for spectatable games)
 *
 * This endpoint allows spectators to view chat messages in real-time during
 * bot vs bot games and other spectatable matches.
 */

import { jsonResponse } from '../../../lib/auth'
import { type ActiveGameRow, type BotPersonaRow } from '../../../lib/types'
import { createDb } from '../../../../shared/db/client'
import { activeGames, gameMessages, botPersonas } from '../../../../shared/db/schema'
import { eq, and, gt, asc } from 'drizzle-orm'

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
  const db = createDb(DB)
  const gameId = context.params.id
  const url = new URL(context.request.url)

  const since = parseInt(url.searchParams.get('since') || '0', 10)
  const limit = Math.min(parseInt(url.searchParams.get('limit') || '50', 10), 100)

  try {
    // Get the game to verify it's spectatable
    const game = await db.query.activeGames.findFirst({
      where: eq(activeGames.id, gameId),
      columns: {
        id: true,
        spectatable: true,
        status: true,
        isBotVsBot: true,
        bot1PersonaId: true,
        bot2PersonaId: true,
      },
    })

    if (!game) {
      return jsonResponse({ error: 'Game not found' }, { status: 404 })
    }

    if (game.spectatable !== 1) {
      return jsonResponse({ error: 'This game is not available for spectating' }, { status: 403 })
    }

    // Get bot persona names for resolving sender names
    const botNames: Record<string, string> = {}

    if (game.isBotVsBot === 1) {
      if (game.bot1PersonaId) {
        const bot1 = await db.query.botPersonas.findFirst({
          where: eq(botPersonas.id, game.bot1PersonaId),
          columns: { id: true, name: true },
        })
        if (bot1) {
          botNames[`bot_${game.bot1PersonaId}`] = bot1.name
        }
      }
      if (game.bot2PersonaId) {
        const bot2 = await db.query.botPersonas.findFirst({
          where: eq(botPersonas.id, game.bot2PersonaId),
          columns: { id: true, name: true },
        })
        if (bot2) {
          botNames[`bot_${game.bot2PersonaId}`] = bot2.name
        }
      }
    }

    // Get messages
    const messages = await db.query.gameMessages.findMany({
      where: and(eq(gameMessages.gameId, gameId), gt(gameMessages.createdAt, since)),
      orderBy: asc(gameMessages.createdAt),
      limit,
    })

    // Transform messages with resolved sender names
    const chatMessages: SpectatorChatMessage[] = messages.map((msg) => {
      let senderName = 'Player'

      if (msg.senderType === 'bot') {
        // Check if we have a bot name for this sender
        senderName = botNames[msg.senderId] || 'Bot'
      } else {
        // For human players, use a generic name for privacy
        senderName = 'Player'
      }

      return {
        id: msg.id,
        senderName,
        senderType: msg.senderType as 'human' | 'bot',
        content: msg.content,
        createdAt: msg.createdAt,
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
