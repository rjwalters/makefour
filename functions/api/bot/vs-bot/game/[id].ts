/**
 * Bot vs Bot Game API endpoint for specific games
 *
 * GET /api/bot/vs-bot/game/:id - Get game state (for spectating)
 * POST /api/bot/vs-bot/game/:id - Advance the game (make next bot move)
 */

import { jsonResponse } from '../../../../lib/auth'
import {
  replayMoves,
  createGameState,
} from '../../../../lib/game'
import {
  advanceGame,
  type ActiveGameRow,
} from '../../../../lib/botVsBotGame'

interface Env {
  DB: D1Database
  BOT_TICK_SECRET?: string
}

interface ActiveGameWithNames extends ActiveGameRow {
  mode: string
  last_move_at: number
  time_control_ms: number | null
  is_bot_game: number
  is_bot_vs_bot: number
  spectator_count: number
  created_at: number
  updated_at: number
  bot1_name: string | null
  bot2_name: string | null
}

/**
 * GET /api/bot/vs-bot/game/:id - Get game state for spectating
 */
export async function onRequestGet(context: EventContext<Env, any, { id: string }>) {
  const { DB } = context.env
  const gameId = context.params.id

  try {
    const game = await DB.prepare(`
      SELECT
        ag.id, ag.player1_id, ag.player2_id, ag.moves, ag.current_turn,
        ag.status, ag.mode, ag.winner, ag.player1_rating, ag.player2_rating,
        ag.spectator_count, ag.move_delay_ms, ag.next_move_at,
        ag.time_control_ms, ag.player1_time_ms, ag.player2_time_ms,
        ag.last_move_at, ag.created_at, ag.updated_at,
        ag.bot1_persona_id, ag.bot2_persona_id,
        bp1.name as bot1_name,
        bp2.name as bot2_name
      FROM active_games ag
      LEFT JOIN bot_personas bp1 ON ag.bot1_persona_id = bp1.id
      LEFT JOIN bot_personas bp2 ON ag.bot2_persona_id = bp2.id
      WHERE ag.id = ? AND ag.is_bot_vs_bot = 1
    `)
      .bind(gameId)
      .first<ActiveGameWithNames>()

    if (!game) {
      return jsonResponse({ error: 'Game not found' }, { status: 404 })
    }

    const moves = JSON.parse(game.moves) as number[]
    const gameState = moves.length > 0 ? replayMoves(moves) : createGameState()

    return jsonResponse({
      id: game.id,
      bot1: {
        personaId: game.bot1_persona_id,
        name: game.bot1_name || 'Bot 1',
        rating: game.player1_rating,
      },
      bot2: {
        personaId: game.bot2_persona_id,
        name: game.bot2_name || 'Bot 2',
        rating: game.player2_rating,
      },
      currentTurn: game.current_turn,
      moves,
      board: gameState?.board ?? null,
      status: game.status,
      winner: game.winner,
      spectatorCount: game.spectator_count,
      moveDelayMs: game.move_delay_ms,
      nextMoveAt: game.next_move_at,
      timeControlMs: game.time_control_ms,
      player1TimeMs: game.player1_time_ms,
      player2TimeMs: game.player2_time_ms,
      lastMoveAt: game.last_move_at,
      createdAt: game.created_at,
      updatedAt: game.updated_at,
    })
  } catch (error) {
    console.error('GET /api/bot/vs-bot/game/:id error:', error)
    return jsonResponse({ error: 'Internal server error' }, { status: 500 })
  }
}

/**
 * POST /api/bot/vs-bot/game/:id - Advance the game (make next bot move)
 *
 * This endpoint is called to make the next move in a bot vs bot game.
 * It can be triggered by a scheduled worker or manually.
 * Requires authentication via BOT_TICK_SECRET for security.
 *
 * Returns the new game state after the move is made.
 */
export async function onRequestPost(context: EventContext<Env, any, { id: string }>) {
  const { DB, BOT_TICK_SECRET } = context.env
  const gameId = context.params.id

  // Authenticate the request
  if (BOT_TICK_SECRET) {
    const authHeader = context.request.headers.get('Authorization')
    const expectedAuth = `Bearer ${BOT_TICK_SECRET}`

    if (!authHeader || authHeader !== expectedAuth) {
      return jsonResponse({ error: 'Unauthorized' }, { status: 401 })
    }
  }

  try {
    // Get the game
    const game = await DB.prepare(`
      SELECT
        id, player1_id, player2_id, moves, current_turn,
        status, winner, player1_rating, player2_rating,
        player1_time_ms, player2_time_ms, turn_started_at,
        bot1_persona_id, bot2_persona_id,
        move_delay_ms, next_move_at
      FROM active_games
      WHERE id = ? AND is_bot_vs_bot = 1
    `)
      .bind(gameId)
      .first<ActiveGameRow>()

    if (!game) {
      return jsonResponse({ error: 'Game not found' }, { status: 404 })
    }

    if (game.status !== 'active') {
      return jsonResponse({ error: 'Game is not active', status: game.status }, { status: 400 })
    }

    const now = Date.now()

    // Check if it's time for the next move (with some tolerance)
    if (game.next_move_at && now < game.next_move_at - 100) {
      return jsonResponse({
        error: 'Too early for next move',
        nextMoveAt: game.next_move_at,
        waitMs: game.next_move_at - now,
      }, { status: 425 }) // 425 Too Early
    }

    // Advance the game using shared logic
    const result = await advanceGame(DB, game, now)

    if (result.status === 'error') {
      return jsonResponse({ error: result.error }, { status: 500 })
    }

    // Get updated game state to return
    const updatedGame = await DB.prepare(`
      SELECT moves, current_turn, status, winner, next_move_at,
             player1_time_ms, player2_time_ms
      FROM active_games
      WHERE id = ?
    `)
      .bind(gameId)
      .first<{
        moves: string
        current_turn: number
        status: string
        winner: string | null
        next_move_at: number | null
        player1_time_ms: number | null
        player2_time_ms: number | null
      }>()

    if (!updatedGame) {
      return jsonResponse({ error: 'Game state not found after update' }, { status: 500 })
    }

    const moves = JSON.parse(updatedGame.moves) as number[]
    const gameState = moves.length > 0 ? replayMoves(moves) : createGameState()

    return jsonResponse({
      success: true,
      move: result.move,
      moves,
      board: gameState?.board ?? null,
      currentTurn: updatedGame.current_turn,
      status: updatedGame.status,
      winner: updatedGame.winner,
      nextMoveAt: updatedGame.next_move_at,
      player1TimeMs: updatedGame.player1_time_ms,
      player2TimeMs: updatedGame.player2_time_ms,
      chatMessage: result.chatMessage,
    })
  } catch (error) {
    console.error('POST /api/bot/vs-bot/game/:id error:', error)
    return jsonResponse({ error: 'Internal server error' }, { status: 500 })
  }
}

export async function onRequestOptions() {
  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  })
}
