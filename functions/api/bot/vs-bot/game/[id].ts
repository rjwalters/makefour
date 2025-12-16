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
import { createDb } from '../../../../../shared/db/client'
import { activeGames } from '../../../../../shared/db/schema'
import { eq, and, sql } from 'drizzle-orm'

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
    const db = createDb(DB)

    const game = await db.select({
      id: activeGames.id,
      player1Id: activeGames.player1Id,
      player2Id: activeGames.player2Id,
      moves: activeGames.moves,
      currentTurn: activeGames.currentTurn,
      status: activeGames.status,
      mode: activeGames.mode,
      winner: activeGames.winner,
      player1Rating: activeGames.player1Rating,
      player2Rating: activeGames.player2Rating,
      spectatorCount: activeGames.spectatorCount,
      moveDelayMs: activeGames.moveDelayMs,
      nextMoveAt: activeGames.nextMoveAt,
      timeControlMs: activeGames.timeControlMs,
      player1TimeMs: activeGames.player1TimeMs,
      player2TimeMs: activeGames.player2TimeMs,
      lastMoveAt: activeGames.lastMoveAt,
      createdAt: activeGames.createdAt,
      updatedAt: activeGames.updatedAt,
      bot1PersonaId: activeGames.bot1PersonaId,
      bot2PersonaId: activeGames.bot2PersonaId,
      bot1Name: sql<string>`bp1.name`,
      bot2Name: sql<string>`bp2.name`,
    })
    .from(activeGames)
    .leftJoin(sql`bot_personas bp1`, sql`${activeGames.bot1PersonaId} = bp1.id`)
    .leftJoin(sql`bot_personas bp2`, sql`${activeGames.bot2PersonaId} = bp2.id`)
    .where(and(eq(activeGames.id, gameId), eq(activeGames.isBotVsBot, 1)))
    .then(rows => rows[0])

    if (!game) {
      return jsonResponse({ error: 'Game not found' }, { status: 404 })
    }

    const moves = JSON.parse(game.moves) as number[]
    const gameState = moves.length > 0 ? replayMoves(moves) : createGameState()

    return jsonResponse({
      id: game.id,
      bot1: {
        personaId: game.bot1PersonaId,
        name: game.bot1Name || 'Bot 1',
        rating: game.player1Rating,
      },
      bot2: {
        personaId: game.bot2PersonaId,
        name: game.bot2Name || 'Bot 2',
        rating: game.player2Rating,
      },
      currentTurn: game.currentTurn,
      moves,
      board: gameState?.board ?? null,
      status: game.status,
      winner: game.winner,
      spectatorCount: game.spectatorCount,
      moveDelayMs: game.moveDelayMs,
      nextMoveAt: game.nextMoveAt,
      timeControlMs: game.timeControlMs,
      player1TimeMs: game.player1TimeMs,
      player2TimeMs: game.player2TimeMs,
      lastMoveAt: game.lastMoveAt,
      createdAt: game.createdAt,
      updatedAt: game.updatedAt,
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
    const db = createDb(DB)

    // Get the game
    const game = await db.query.activeGames.findFirst({
      where: and(eq(activeGames.id, gameId), eq(activeGames.isBotVsBot, 1)),
      columns: {
        id: true,
        player1Id: true,
        player2Id: true,
        moves: true,
        currentTurn: true,
        status: true,
        winner: true,
        player1Rating: true,
        player2Rating: true,
        player1TimeMs: true,
        player2TimeMs: true,
        turnStartedAt: true,
        bot1PersonaId: true,
        bot2PersonaId: true,
        moveDelayMs: true,
        nextMoveAt: true,
      },
    })

    if (!game) {
      return jsonResponse({ error: 'Game not found' }, { status: 404 })
    }

    if (game.status !== 'active') {
      return jsonResponse({ error: 'Game is not active', status: game.status }, { status: 400 })
    }

    const now = Date.now()

    // Check if it's time for the next move (with some tolerance)
    if (game.nextMoveAt && now < game.nextMoveAt - 100) {
      return jsonResponse({
        error: 'Too early for next move',
        nextMoveAt: game.nextMoveAt,
        waitMs: game.nextMoveAt - now,
      }, { status: 425 }) // 425 Too Early
    }

    // Convert to ActiveGameRow format for advanceGame
    const gameRow: ActiveGameRow = {
      id: game.id,
      player1_id: game.player1Id,
      player2_id: game.player2Id,
      moves: game.moves,
      current_turn: game.currentTurn,
      status: game.status,
      winner: game.winner,
      player1_rating: game.player1Rating,
      player2_rating: game.player2Rating,
      player1_time_ms: game.player1TimeMs,
      player2_time_ms: game.player2TimeMs,
      turn_started_at: game.turnStartedAt,
      bot1_persona_id: game.bot1PersonaId,
      bot2_persona_id: game.bot2PersonaId,
      move_delay_ms: game.moveDelayMs,
      next_move_at: game.nextMoveAt,
    }

    // Advance the game using shared logic
    const result = await advanceGame(DB, gameRow, now)

    if (result.status === 'error') {
      return jsonResponse({ error: result.error }, { status: 500 })
    }

    // Get updated game state to return
    const updatedGame = await db.query.activeGames.findFirst({
      where: eq(activeGames.id, gameId),
      columns: {
        moves: true,
        currentTurn: true,
        status: true,
        winner: true,
        nextMoveAt: true,
        player1TimeMs: true,
        player2TimeMs: true,
      },
    })

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
      currentTurn: updatedGame.currentTurn,
      status: updatedGame.status,
      winner: updatedGame.winner,
      nextMoveAt: updatedGame.nextMoveAt,
      player1TimeMs: updatedGame.player1TimeMs,
      player2TimeMs: updatedGame.player2TimeMs,
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
