/**
 * Spectate game API endpoint
 *
 * GET /api/games/:id/spectate - Get game state for spectating (no auth required)
 */

import { jsonResponse } from '../../../lib/auth'
import { replayMoves, createGameState } from '../../../lib/game'

interface Env {
  DB: D1Database
}

interface ActiveGameRow {
  id: string
  player1_id: string
  player2_id: string
  moves: string
  current_turn: number
  status: string
  mode: string
  winner: string | null
  player1_rating: number
  player2_rating: number
  spectatable: number
  spectator_count: number
  last_move_at: number
  time_control_ms: number | null
  player1_time_ms: number | null
  player2_time_ms: number | null
  turn_started_at: number | null
  created_at: number
  updated_at: number
  // Bot vs bot fields
  is_bot_vs_bot: number
  bot1_persona_id: string | null
  bot2_persona_id: string | null
  move_delay_ms: number | null
  next_move_at: number | null
}

interface UserRow {
  id: string
  email: string
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
  const gameId = context.params.id

  try {
    // Get the game with player info and bot persona names
    const game = await DB.prepare(`
      SELECT
        ag.id,
        ag.player1_id,
        ag.player2_id,
        ag.moves,
        ag.current_turn,
        ag.status,
        ag.mode,
        ag.winner,
        ag.player1_rating,
        ag.player2_rating,
        ag.spectatable,
        ag.spectator_count,
        ag.last_move_at,
        ag.time_control_ms,
        ag.player1_time_ms,
        ag.player2_time_ms,
        ag.turn_started_at,
        ag.created_at,
        ag.updated_at,
        ag.is_bot_vs_bot,
        ag.bot1_persona_id,
        ag.bot2_persona_id,
        ag.move_delay_ms,
        ag.next_move_at,
        bp1.name as bot1_name,
        bp2.name as bot2_name
      FROM active_games ag
      LEFT JOIN bot_personas bp1 ON ag.bot1_persona_id = bp1.id
      LEFT JOIN bot_personas bp2 ON ag.bot2_persona_id = bp2.id
      WHERE ag.id = ?
    `)
      .bind(gameId)
      .first<ActiveGameRow & { bot1_name: string | null; bot2_name: string | null }>()

    if (!game) {
      return jsonResponse({ error: 'Game not found' }, { status: 404 })
    }

    // Check if game is spectatable
    if (game.spectatable !== 1) {
      return jsonResponse({ error: 'This game is not available for spectating' }, { status: 403 })
    }

    // Get player emails for display names
    const [player1, player2] = await Promise.all([
      DB.prepare('SELECT id, email FROM users WHERE id = ?')
        .bind(game.player1_id)
        .first<UserRow>(),
      DB.prepare('SELECT id, email FROM users WHERE id = ?')
        .bind(game.player2_id)
        .first<UserRow>(),
    ])

    const moves = JSON.parse(game.moves) as number[]
    const isBotVsBot = game.is_bot_vs_bot === 1

    // Reconstruct board state from moves
    const gameState = moves.length > 0 ? replayMoves(moves) : createGameState()

    // Calculate remaining time for display
    const now = Date.now()
    let player1TimeMs = game.player1_time_ms
    let player2TimeMs = game.player2_time_ms

    if (game.time_control_ms !== null && game.turn_started_at !== null && game.status === 'active') {
      const elapsed = now - game.turn_started_at
      if (game.current_turn === 1) {
        player1TimeMs = Math.max(0, (game.player1_time_ms ?? 0) - elapsed)
      } else {
        player2TimeMs = Math.max(0, (game.player2_time_ms ?? 0) - elapsed)
      }
    }

    // For bot vs bot games, show bot names instead of masked emails
    const player1DisplayName = isBotVsBot && game.bot1_name
      ? game.bot1_name
      : (player1 ? maskEmail(player1.email) : 'Player 1')
    const player2DisplayName = isBotVsBot && game.bot2_name
      ? game.bot2_name
      : (player2 ? maskEmail(player2.email) : 'Player 2')

    const response: SpectatorGameState = {
      id: game.id,
      player1: {
        rating: game.player1_rating,
        displayName: player1DisplayName,
        isBot: isBotVsBot,
        personaId: game.bot1_persona_id || undefined,
      },
      player2: {
        rating: game.player2_rating,
        displayName: player2DisplayName,
        isBot: isBotVsBot,
        personaId: game.bot2_persona_id || undefined,
      },
      currentTurn: game.current_turn as 1 | 2,
      moves,
      board: gameState?.board ?? null,
      status: game.status as 'active' | 'completed' | 'abandoned',
      winner: game.winner as '1' | '2' | 'draw' | null,
      mode: game.mode as 'ranked' | 'casual',
      spectatorCount: game.spectator_count,
      lastMoveAt: game.last_move_at,
      createdAt: game.created_at,
      timeControlMs: game.time_control_ms,
      player1TimeMs,
      player2TimeMs,
      turnStartedAt: game.turn_started_at,
      isBotVsBot,
      moveDelayMs: isBotVsBot ? game.move_delay_ms : undefined,
      nextMoveAt: isBotVsBot ? game.next_move_at : undefined,
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
