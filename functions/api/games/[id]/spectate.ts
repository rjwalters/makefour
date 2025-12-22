/**
 * Spectate game API endpoint
 *
 * GET /api/games/:id/spectate - Get game state for spectating (no auth required)
 */

import { jsonResponse } from '../../../lib/auth'
import { replayMoves, createGameState } from '../../../lib/game'
import { type ActiveGameRow, type UserRow, safeParseMoves } from '../../../lib/types'
import { createDb } from '../../../../shared/db/client'
import { activeGames, users, botPersonas } from '../../../../shared/db/schema'
import { eq } from 'drizzle-orm'

interface Env {
  DB: D1Database
}

export interface SpectatorGameState {
  id: string
  player1: {
    rating: number
    displayName: string
    isBot?: boolean
    personaId?: string
  }
  player2: {
    rating: number
    displayName: string
    isBot?: boolean
    personaId?: string
  }
  currentTurn: 1 | 2
  moves: number[]
  board: (0 | 1 | 2)[][] | null
  status: 'active' | 'completed' | 'abandoned'
  winner: '1' | '2' | 'draw' | null
  mode: 'ranked' | 'casual'
  spectatorCount: number
  lastMoveAt: number
  createdAt: number
  // Timer fields
  timeControlMs: number | null
  player1TimeMs: number | null
  player2TimeMs: number | null
  turnStartedAt: number | null
  // Bot vs bot specific fields
  isBotVsBot?: boolean
  moveDelayMs?: number | null
  nextMoveAt?: number | null
}

/**
 * GET /api/games/:id/spectate - Get game state for spectating
 *
 * No authentication required, but game must be spectatable
 */
export async function onRequestGet(context: EventContext<Env, any, { id: string }>) {
  const { DB } = context.env
  const db = createDb(DB)
  const gameId = context.params.id

  try {
    // Get the game with bot personas using relational query
    const game = await db.query.activeGames.findFirst({
      where: eq(activeGames.id, gameId),
      with: {
        bot1Persona: {
          columns: {
            name: true,
          },
        },
        bot2Persona: {
          columns: {
            name: true,
          },
        },
      },
    })

    if (!game) {
      return jsonResponse({ error: 'Game not found' }, { status: 404 })
    }

    // Check if game is spectatable
    if (game.spectatable !== 1) {
      return jsonResponse({ error: 'This game is not available for spectating' }, { status: 403 })
    }

    // Get player info for display names
    const [player1, player2] = await Promise.all([
      db.query.users.findFirst({
        where: eq(users.id, game.player1Id),
        columns: {
          id: true,
          email: true,
          username: true,
        },
      }),
      db.query.users.findFirst({
        where: eq(users.id, game.player2Id),
        columns: {
          id: true,
          email: true,
          username: true,
        },
      }),
    ])

    const moves = safeParseMoves(game.moves)
    const isBotVsBot = game.isBotVsBot === 1

    // Reconstruct board state from moves
    const gameState = moves.length > 0 ? replayMoves(moves) : createGameState()

    // Calculate remaining time for display
    const now = Date.now()
    let player1TimeMs = game.player1TimeMs
    let player2TimeMs = game.player2TimeMs

    if (game.timeControlMs !== null && game.turnStartedAt !== null && game.status === 'active') {
      const elapsed = now - game.turnStartedAt
      if (game.currentTurn === 1) {
        player1TimeMs = Math.max(0, (game.player1TimeMs ?? 0) - elapsed)
      } else {
        player2TimeMs = Math.max(0, (game.player2TimeMs ?? 0) - elapsed)
      }
    }

    // For bot vs bot games, show bot names; otherwise prefer username, fall back to masked email
    const player1DisplayName = isBotVsBot && game.bot1Persona?.name
      ? game.bot1Persona.name
      : (player1 ? (player1.username || maskEmail(player1.email)) : 'Player 1')
    const player2DisplayName = isBotVsBot && game.bot2Persona?.name
      ? game.bot2Persona.name
      : (player2 ? (player2.username || maskEmail(player2.email)) : 'Player 2')

    const response: SpectatorGameState = {
      id: game.id,
      player1: {
        rating: game.player1Rating,
        displayName: player1DisplayName,
        isBot: isBotVsBot,
        personaId: game.bot1PersonaId || undefined,
      },
      player2: {
        rating: game.player2Rating,
        displayName: player2DisplayName,
        isBot: isBotVsBot,
        personaId: game.bot2PersonaId || undefined,
      },
      currentTurn: game.currentTurn as 1 | 2,
      moves,
      board: gameState?.board ?? null,
      status: game.status as 'active' | 'completed' | 'abandoned',
      winner: game.winner as '1' | '2' | 'draw' | null,
      mode: game.mode as 'ranked' | 'casual',
      spectatorCount: game.spectatorCount,
      lastMoveAt: game.lastMoveAt,
      createdAt: game.createdAt,
      timeControlMs: game.timeControlMs,
      player1TimeMs,
      player2TimeMs,
      turnStartedAt: game.turnStartedAt,
      isBotVsBot,
      moveDelayMs: isBotVsBot ? game.moveDelayMs : undefined,
      nextMoveAt: isBotVsBot ? game.nextMoveAt : undefined,
    }

    return jsonResponse(response)
  } catch (error) {
    console.error('GET /api/games/:id/spectate error:', error)
    return jsonResponse({ error: 'Internal server error' }, { status: 500 })
  }
}

/**
 * Mask email for display (e.g., "user@example.com" -> "us***@example.com")
 */
function maskEmail(email: string): string {
  const [local, domain] = email.split('@')
  if (!domain) return email
  if (local.length <= 2) return `${local}***@${domain}`
  return `${local.slice(0, 2)}***@${domain}`
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
