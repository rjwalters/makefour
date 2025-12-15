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
  }
  player2: {
    rating: number
    displayName: string
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
    // Get the game with player info
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
        ag.updated_at
      FROM active_games ag
      WHERE ag.id = ?
    `)
      .bind(gameId)
      .first<ActiveGameRow>()

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

    const response: SpectatorGameState = {
      id: game.id,
      player1: {
        rating: game.player1_rating,
        displayName: player1 ? maskEmail(player1.email) : 'Player 1',
      },
      player2: {
        rating: game.player2_rating,
        displayName: player2 ? maskEmail(player2.email) : 'Player 2',
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
