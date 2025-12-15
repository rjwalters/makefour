/**
 * Bot Game API endpoint for specific games
 *
 * GET /api/bot/game/:id - Get game state
 * POST /api/bot/game/:id - Submit a move (human move, bot responds)
 */

import { validateSession, errorResponse, jsonResponse } from '../../../lib/auth'
import {
  replayMoves,
  makeMove,
  isValidMove,
  createGameState,
  type Board,
  type Player,
} from '../../../lib/game'
import { calculateNewRating, type GameOutcome } from '../../../lib/elo'
import {
  suggestMoveWithEngine,
  calculateTimeBudget,
  type DifficultyLevel,
  type BotPersonaConfig,
  type AIConfig,
} from '../../../lib/bot'
import { z } from 'zod'

interface Env {
  DB: D1Database
}

const BOT_USER_ID = 'bot-opponent'

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
  last_move_at: number
  time_control_ms: number | null
  player1_time_ms: number | null
  player2_time_ms: number | null
  turn_started_at: number | null
  is_bot_game: number
  bot_difficulty: string | null
  bot_persona_id: string | null
  created_at: number
  updated_at: number
}

interface BotPersonaRow {
  id: string
  ai_config: string
}

const moveSchema = z.object({
  column: z.number().int().min(0).max(6),
})

/**
 * Calculate current time remaining for display
 */
function calculateTimeRemaining(
  game: ActiveGameRow,
  now: number
): { player1TimeMs: number | null; player2TimeMs: number | null } {
  if (game.time_control_ms === null || game.turn_started_at === null) {
    return { player1TimeMs: null, player2TimeMs: null }
  }

  const elapsed = now - game.turn_started_at

  if (game.current_turn === 1) {
    return {
      player1TimeMs: Math.max(0, (game.player1_time_ms ?? 0) - elapsed),
      player2TimeMs: game.player2_time_ms,
    }
  } else {
    return {
      player1TimeMs: game.player1_time_ms,
      player2TimeMs: Math.max(0, (game.player2_time_ms ?? 0) - elapsed),
    }
  }
}

/**
 * GET /api/bot/game/:id - Get current game state
 */
export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB } = context.env
  const gameId = context.params.id as string

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    const game = await DB.prepare(`
      SELECT id, player1_id, player2_id, moves, current_turn, status, mode,
             winner, player1_rating, player2_rating, last_move_at,
             time_control_ms, player1_time_ms, player2_time_ms, turn_started_at,
             is_bot_game, bot_difficulty, bot_persona_id, created_at, updated_at
      FROM active_games
      WHERE id = ? AND is_bot_game = 1
    `)
      .bind(gameId)
      .first<ActiveGameRow>()

    if (!game) {
      return errorResponse('Game not found', 404)
    }

    // Verify user is the human player
    const isPlayer1 = game.player1_id === session.userId
    const isPlayer2 = game.player2_id === session.userId
    if (!isPlayer1 && !isPlayer2) {
      return errorResponse('You are not a participant in this game', 403)
    }

    const now = Date.now()
    const playerNumber = isPlayer1 ? 1 : 2
    const moves = JSON.parse(game.moves) as number[]
    const gameState = moves.length > 0 ? replayMoves(moves) : createGameState()
    const timeRemaining = calculateTimeRemaining(game, now)

    return jsonResponse({
      id: game.id,
      playerNumber,
      currentTurn: game.current_turn,
      moves,
      board: gameState?.board ?? null,
      status: game.status,
      winner: game.winner,
      mode: game.mode,
      opponentRating: playerNumber === 1 ? game.player2_rating : game.player1_rating,
      lastMoveAt: game.last_move_at,
      createdAt: game.created_at,
      isYourTurn: game.status === 'active' && game.current_turn === playerNumber,
      timeControlMs: game.time_control_ms,
      player1TimeMs: timeRemaining.player1TimeMs,
      player2TimeMs: timeRemaining.player2TimeMs,
      turnStartedAt: game.turn_started_at,
      isBotGame: true,
      botDifficulty: game.bot_difficulty,
      botPersonaId: game.bot_persona_id,
    })
  } catch (error) {
    console.error('GET /api/bot/game/:id error:', error)
    return errorResponse('Internal server error', 500)
  }
}

/**
 * POST /api/bot/game/:id - Submit a move (human move, bot responds)
 */
export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB } = context.env
  const gameId = context.params.id as string

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    const body = await context.request.json()
    const parseResult = moveSchema.safeParse(body)

    if (!parseResult.success) {
      return errorResponse(parseResult.error.errors[0].message, 400)
    }

    const { column } = parseResult.data

    const game = await DB.prepare(`
      SELECT id, player1_id, player2_id, moves, current_turn, status, mode,
             winner, player1_rating, player2_rating, last_move_at,
             time_control_ms, player1_time_ms, player2_time_ms, turn_started_at,
             is_bot_game, bot_difficulty, bot_persona_id, created_at, updated_at
      FROM active_games
      WHERE id = ? AND is_bot_game = 1
    `)
      .bind(gameId)
      .first<ActiveGameRow>()

    if (!game) {
      return errorResponse('Game not found', 404)
    }

    // Verify user is the human player
    const isPlayer1 = game.player1_id === session.userId
    const isPlayer2 = game.player2_id === session.userId
    if (!isPlayer1 && !isPlayer2) {
      return errorResponse('You are not a participant in this game', 403)
    }

    if (game.status !== 'active') {
      return errorResponse('Game is not active', 400)
    }

    // Look up persona AI config if available
    let personaAIConfig: AIConfig | null = null
    if (game.bot_persona_id) {
      const persona = await DB.prepare(`
        SELECT id, ai_config FROM bot_personas WHERE id = ?
      `)
        .bind(game.bot_persona_id)
        .first<BotPersonaRow>()

      if (persona) {
        personaAIConfig = JSON.parse(persona.ai_config)
      }
    }

    const playerNumber = isPlayer1 ? 1 : 2
    const botPlayerNumber = isPlayer1 ? 2 : 1
    const now = Date.now()

    // Check it's the human's turn
    if (game.current_turn !== playerNumber) {
      return errorResponse('Not your turn', 400)
    }

    // Time tracking
    let player1Time = game.player1_time_ms
    let player2Time = game.player2_time_ms

    if (game.time_control_ms !== null && game.turn_started_at !== null) {
      const elapsed = now - game.turn_started_at
      const currentPlayerTime = playerNumber === 1 ? player1Time : player2Time

      if (currentPlayerTime !== null) {
        const timeRemaining = currentPlayerTime - elapsed

        if (timeRemaining <= 0) {
          // Human ran out of time - bot wins
          const winner = String(botPlayerNumber)

          await DB.prepare(`
            UPDATE active_games
            SET status = 'completed', winner = ?,
                player1_time_ms = ?, player2_time_ms = ?, updated_at = ?
            WHERE id = ?
          `).bind(
            winner,
            playerNumber === 1 ? 0 : player1Time,
            playerNumber === 2 ? 0 : player2Time,
            now,
            gameId
          ).run()

          // Update user's rating
          await updateUserRating(DB, session.userId, game, 'loss', now)

          return errorResponse('Time expired', 400)
        }

        // Deduct time
        if (playerNumber === 1) {
          player1Time = timeRemaining
        } else {
          player2Time = timeRemaining
        }
      }
    }

    // Validate and apply human's move
    const moves = JSON.parse(game.moves) as number[]
    const currentState = moves.length > 0 ? replayMoves(moves) : createGameState()

    if (!currentState) {
      return errorResponse('Invalid game state', 500)
    }

    if (!isValidMove(currentState.board, column)) {
      return errorResponse('Invalid move: column is full or out of bounds', 400)
    }

    const afterHumanMove = makeMove(currentState, column)
    if (!afterHumanMove) {
      return errorResponse('Invalid move', 400)
    }

    let newMoves = [...moves, column]
    let newStatus = 'active'
    let winner: string | null = null
    let board = afterHumanMove.board
    let currentTurn = botPlayerNumber

    // Check if human's move ended the game
    if (afterHumanMove.winner !== null) {
      newStatus = 'completed'
      winner = afterHumanMove.winner === 'draw' ? 'draw' : String(afterHumanMove.winner)
    } else {
      // Bot makes its move using engine-based API
      const difficulty = (game.bot_difficulty || 'intermediate') as DifficultyLevel
      const botTimeRemaining = botPlayerNumber === 1 ? player1Time : player2Time

      if (botTimeRemaining !== null && botTimeRemaining > 0) {
        const timeBudget = calculateTimeBudget(botTimeRemaining, newMoves.length, difficulty)
        const startTime = Date.now()

        // Use engine-based move suggestion (defaults to minimax)
        const botPersonaConfig: BotPersonaConfig = {
          difficulty,
          engine: 'minimax', // Default engine, can be extended to use stored engine preference
        }

        const moveResult = await suggestMoveWithEngine(
          afterHumanMove.board,
          botPlayerNumber as Player,
          botPersonaConfig,
          timeBudget
        )

        const botElapsed = Date.now() - startTime

        // Deduct bot's time
        if (botPlayerNumber === 1) {
          player1Time = Math.max(0, (player1Time ?? 0) - botElapsed)
        } else {
          player2Time = Math.max(0, (player2Time ?? 0) - botElapsed)
        }

        // Check if bot ran out of time
        const newBotTime = botPlayerNumber === 1 ? player1Time : player2Time
        if (newBotTime !== null && newBotTime <= 0) {
          // Bot ran out of time - human wins
          newStatus = 'completed'
          winner = String(playerNumber)
        } else {
          // Apply bot's move
          const afterBotMove = makeMove(afterHumanMove, moveResult.column)
          if (afterBotMove) {
            newMoves = [...newMoves, moveResult.column]
            board = afterBotMove.board

            if (afterBotMove.winner !== null) {
              newStatus = 'completed'
              winner = afterBotMove.winner === 'draw' ? 'draw' : String(afterBotMove.winner)
            } else {
              currentTurn = playerNumber // Back to human's turn
            }
          }
        }
      }
    }

    // Update the game
    const turnStartedAt = newStatus === 'active' ? Date.now() : game.turn_started_at

    await DB.prepare(`
      UPDATE active_games
      SET moves = ?, current_turn = ?, status = ?, winner = ?,
          last_move_at = ?, updated_at = ?,
          player1_time_ms = ?, player2_time_ms = ?, turn_started_at = ?
      WHERE id = ?
    `).bind(
      JSON.stringify(newMoves),
      currentTurn,
      newStatus,
      winner,
      now,
      now,
      player1Time,
      player2Time,
      turnStartedAt,
      gameId
    ).run()

    // Update rating if game completed
    if (newStatus === 'completed') {
      let outcome: GameOutcome
      if (winner === 'draw') {
        outcome = 'draw'
      } else if (winner === String(playerNumber)) {
        outcome = 'win'
      } else {
        outcome = 'loss'
      }
      await updateUserRating(DB, session.userId, game, outcome, now)
    }

    return jsonResponse({
      success: true,
      moves: newMoves,
      board,
      currentTurn,
      status: newStatus,
      winner,
      isYourTurn: newStatus === 'active' && currentTurn === playerNumber,
      timeControlMs: game.time_control_ms,
      player1TimeMs: player1Time,
      player2TimeMs: player2Time,
      turnStartedAt,
    })
  } catch (error) {
    console.error('POST /api/bot/game/:id error:', error)
    return errorResponse('Internal server error', 500)
  }
}

/**
 * Update user's rating after bot game
 */
async function updateUserRating(
  DB: D1Database,
  userId: string,
  game: ActiveGameRow,
  outcome: GameOutcome,
  now: number
) {
  const user = await DB.prepare(`
    SELECT rating, games_played FROM users WHERE id = ?
  `).bind(userId).first<{ rating: number; games_played: number }>()

  if (!user) return

  const isPlayer1 = game.player1_id === userId
  const userRating = isPlayer1 ? game.player1_rating : game.player2_rating
  const botRating = isPlayer1 ? game.player2_rating : game.player1_rating

  const result = calculateNewRating(userRating, botRating, outcome, user.games_played)

  const gameId = crypto.randomUUID()
  const ratingHistoryId = crypto.randomUUID()
  const moves = JSON.parse(game.moves) as number[]

  await DB.batch([
    // Game record
    DB.prepare(`
      INSERT INTO games (id, user_id, outcome, moves, move_count, rating_change,
                        opponent_type, ai_difficulty, player_number, created_at)
      VALUES (?, ?, ?, ?, ?, ?, 'ai', ?, ?, ?)
    `).bind(
      gameId,
      userId,
      outcome,
      JSON.stringify(moves),
      moves.length,
      result.ratingChange,
      game.bot_difficulty,
      isPlayer1 ? 1 : 2,
      now
    ),

    // Update user stats
    DB.prepare(`
      UPDATE users SET
        rating = ?,
        games_played = games_played + 1,
        wins = wins + ?,
        losses = losses + ?,
        draws = draws + ?,
        updated_at = ?
      WHERE id = ?
    `).bind(
      result.newRating,
      outcome === 'win' ? 1 : 0,
      outcome === 'loss' ? 1 : 0,
      outcome === 'draw' ? 1 : 0,
      now,
      userId
    ),

    // Rating history
    DB.prepare(`
      INSERT INTO rating_history (id, user_id, game_id, rating_before, rating_after, rating_change, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `).bind(
      ratingHistoryId,
      userId,
      gameId,
      userRating,
      result.newRating,
      result.ratingChange,
      now
    ),
  ])
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
