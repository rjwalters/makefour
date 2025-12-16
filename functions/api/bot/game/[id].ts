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
import { createDb } from '../../../../shared/db/client'
import { users, activeGames, botPersonas, games, ratingHistory, playerBotStats } from '../../../../shared/db/schema'
import { eq, and } from 'drizzle-orm'

interface Env {
  DB: D1Database
}

/**
 * Check if a user ID is a bot user ID
 */
function isBotUserId(userId: string): boolean {
  return userId.startsWith('bot_')
}

/**
 * Get the bot user ID (player1 or player2 that is a bot)
 */
function getBotUserId(game: ActiveGameRow): string | null {
  if (isBotUserId(game.player1_id)) return game.player1_id
  if (isBotUserId(game.player2_id)) return game.player2_id
  return null
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
 * Check if the current player's time has expired and handle timeout
 * Returns the updated game if timeout occurred, null otherwise
 */
async function checkAndHandleTimeout(
  db: ReturnType<typeof createDb>,
  game: ActiveGameRow,
  now: number
): Promise<{ timedOut: boolean; winner?: string }> {
  // Only check timed games that are active
  if (game.time_control_ms === null || game.status !== 'active' || game.turn_started_at === null) {
    return { timedOut: false }
  }

  const elapsed = now - game.turn_started_at
  const currentPlayerTime = game.current_turn === 1 ? game.player1_time_ms : game.player2_time_ms

  if (currentPlayerTime === null) {
    return { timedOut: false }
  }

  const timeRemaining = currentPlayerTime - elapsed

  if (timeRemaining <= 0) {
    // Current player has timed out - opponent wins
    const winner = game.current_turn === 1 ? '2' : '1'

    await db.update(activeGames)
      .set({
        status: 'completed',
        winner,
        player1TimeMs: game.current_turn === 1 ? 0 : game.player1_time_ms,
        player2TimeMs: game.current_turn === 2 ? 0 : game.player2_time_ms,
        updatedAt: now,
      })
      .where(eq(activeGames.id, game.id))

    // Update user's rating if ranked bot game
    if (game.mode === 'ranked') {
      // Determine which player is human and which is bot
      const humanIsPlayer1 = !isBotUserId(game.player1_id)
      const humanUserId = humanIsPlayer1 ? game.player1_id : game.player2_id
      const humanPlayerNumber = humanIsPlayer1 ? 1 : 2

      // Human wins if bot timed out, loses if human timed out
      const humanTimedOut = game.current_turn === humanPlayerNumber
      const outcome: GameOutcome = humanTimedOut ? 'loss' : 'win'

      await updateUserRating(db, humanUserId, game, outcome, now)
    }

    return { timedOut: true, winner }
  }

  return { timedOut: false }
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

    const db = createDb(DB)
    const game = await db.query.activeGames.findFirst({
      where: and(eq(activeGames.id, gameId), eq(activeGames.isBotGame, 1)),
    })

    if (!game) {
      return errorResponse('Game not found', 404)
    }

    // Verify user is the human player
    const isPlayer1 = game.player1Id === session.userId
    const isPlayer2 = game.player2Id === session.userId
    if (!isPlayer1 && !isPlayer2) {
      return errorResponse('You are not a participant in this game', 403)
    }

    const now = Date.now()
    const playerNumber = isPlayer1 ? 1 : 2
    const moves = JSON.parse(game.moves) as number[]
    const gameState = moves.length > 0 ? replayMoves(moves) : createGameState()
    const gameRow: ActiveGameRow = {
      id: game.id,
      player1_id: game.player1Id,
      player2_id: game.player2Id,
      moves: game.moves,
      current_turn: game.currentTurn,
      status: game.status,
      mode: game.mode,
      winner: game.winner,
      player1_rating: game.player1Rating,
      player2_rating: game.player2Rating,
      last_move_at: game.lastMoveAt,
      time_control_ms: game.timeControlMs,
      player1_time_ms: game.player1TimeMs,
      player2_time_ms: game.player2TimeMs,
      turn_started_at: game.turnStartedAt,
      is_bot_game: game.isBotGame,
      bot_difficulty: game.botDifficulty,
      bot_persona_id: game.botPersonaId,
      created_at: game.createdAt,
      updated_at: game.updatedAt,
    }

    // Check for timeout (this may end the game)
    const timeoutResult = await checkAndHandleTimeout(db, gameRow, now)

    // If timeout occurred, update game state for response
    let status = game.status
    let winner = game.winner
    if (timeoutResult.timedOut) {
      status = 'completed'
      winner = timeoutResult.winner ?? null
    }

    const timeRemaining = calculateTimeRemaining(gameRow, now)

    return jsonResponse({
      id: game.id,
      playerNumber,
      currentTurn: game.currentTurn,
      moves,
      board: gameState?.board ?? null,
      status,
      winner,
      mode: game.mode,
      opponentRating: playerNumber === 1 ? game.player2Rating : game.player1Rating,
      lastMoveAt: game.lastMoveAt,
      createdAt: game.createdAt,
      isYourTurn: status === 'active' && game.currentTurn === playerNumber,
      timeControlMs: game.timeControlMs,
      player1TimeMs: timeoutResult.timedOut
        ? (game.currentTurn === 1 ? 0 : game.player1TimeMs)
        : timeRemaining.player1TimeMs,
      player2TimeMs: timeoutResult.timedOut
        ? (game.currentTurn === 2 ? 0 : game.player2TimeMs)
        : timeRemaining.player2TimeMs,
      turnStartedAt: game.turnStartedAt,
      isBotGame: true,
      botDifficulty: game.botDifficulty,
      botPersonaId: game.botPersonaId,
    })
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error)
    console.error('GET /api/bot/game/:id error:', {
      error: errorMessage,
      stack: error instanceof Error ? error.stack : undefined,
      gameId,
    })
    return errorResponse(`Internal server error: ${errorMessage}`, 500)
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

    const db = createDb(DB)
    const game = await db.query.activeGames.findFirst({
      where: and(eq(activeGames.id, gameId), eq(activeGames.isBotGame, 1)),
    })

    if (!game) {
      return errorResponse('Game not found', 404)
    }

    // Verify user is the human player
    const isPlayer1 = game.player1Id === session.userId
    const isPlayer2 = game.player2Id === session.userId
    if (!isPlayer1 && !isPlayer2) {
      return errorResponse('You are not a participant in this game', 403)
    }

    if (game.status !== 'active') {
      return errorResponse('Game is not active', 400)
    }

    // Look up persona AI config if available
    let personaAIConfig: AIConfig | null = null
    if (game.botPersonaId) {
      const persona = await db.query.botPersonas.findFirst({
        where: eq(botPersonas.id, game.botPersonaId),
        columns: {
          id: true,
          aiConfig: true,
        },
      })

      if (persona) {
        personaAIConfig = JSON.parse(persona.aiConfig)
      }
    }

    const playerNumber = isPlayer1 ? 1 : 2
    const botPlayerNumber = isPlayer1 ? 2 : 1
    const now = Date.now()

    // Check it's the human's turn
    if (game.currentTurn !== playerNumber) {
      return errorResponse('Not your turn', 400)
    }

    // Time tracking
    let player1Time = game.player1TimeMs
    let player2Time = game.player2TimeMs

    if (game.timeControlMs !== null && game.turnStartedAt !== null) {
      const elapsed = now - game.turnStartedAt
      const currentPlayerTime = playerNumber === 1 ? player1Time : player2Time

      if (currentPlayerTime !== null) {
        const timeRemaining = currentPlayerTime - elapsed

        if (timeRemaining <= 0) {
          // Human ran out of time - bot wins
          const winner = String(botPlayerNumber)

          await db.update(activeGames)
            .set({
              status: 'completed',
              winner,
              player1TimeMs: playerNumber === 1 ? 0 : player1Time,
              player2TimeMs: playerNumber === 2 ? 0 : player2Time,
              updatedAt: now,
            })
            .where(eq(activeGames.id, gameId))

          // Update user's rating
          const gameRow: ActiveGameRow = {
            id: game.id,
            player1_id: game.player1Id,
            player2_id: game.player2Id,
            moves: game.moves,
            current_turn: game.currentTurn,
            status: game.status,
            mode: game.mode,
            winner: game.winner,
            player1_rating: game.player1Rating,
            player2_rating: game.player2Rating,
            last_move_at: game.lastMoveAt,
            time_control_ms: game.timeControlMs,
            player1_time_ms: game.player1TimeMs,
            player2_time_ms: game.player2TimeMs,
            turn_started_at: game.turnStartedAt,
            is_bot_game: game.isBotGame,
            bot_difficulty: game.botDifficulty,
            bot_persona_id: game.botPersonaId,
            created_at: game.createdAt,
            updated_at: game.updatedAt,
          }
          await updateUserRating(db, session.userId, gameRow, 'loss', now)

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
    const turnStartedAt = newStatus === 'active' ? Date.now() : game.turnStartedAt

    await db.update(activeGames)
      .set({
        moves: JSON.stringify(newMoves),
        currentTurn,
        status: newStatus,
        winner,
        lastMoveAt: now,
        updatedAt: now,
        player1TimeMs: player1Time,
        player2TimeMs: player2Time,
        turnStartedAt,
      })
      .where(eq(activeGames.id, gameId))

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
      const gameRow: ActiveGameRow = {
        id: game.id,
        player1_id: game.player1Id,
        player2_id: game.player2Id,
        moves: game.moves,
        current_turn: game.currentTurn,
        status: game.status,
        mode: game.mode,
        winner: game.winner,
        player1_rating: game.player1Rating,
        player2_rating: game.player2Rating,
        last_move_at: game.lastMoveAt,
        time_control_ms: game.timeControlMs,
        player1_time_ms: game.player1TimeMs,
        player2_time_ms: game.player2TimeMs,
        turn_started_at: game.turnStartedAt,
        is_bot_game: game.isBotGame,
        bot_difficulty: game.botDifficulty,
        bot_persona_id: game.botPersonaId,
        created_at: game.createdAt,
        updated_at: game.updatedAt,
      }
      await updateUserRating(db, session.userId, gameRow, outcome, now)
    }

    return jsonResponse({
      success: true,
      moves: newMoves,
      board,
      currentTurn,
      status: newStatus,
      winner,
      isYourTurn: newStatus === 'active' && currentTurn === playerNumber,
      timeControlMs: game.timeControlMs,
      player1TimeMs: player1Time,
      player2TimeMs: player2Time,
      turnStartedAt,
    })
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error)
    const errorStack = error instanceof Error ? error.stack : undefined
    console.error('POST /api/bot/game/:id error:', {
      error: errorMessage,
      stack: errorStack,
      gameId,
    })
    return errorResponse(`Internal server error: ${errorMessage}`, 500)
  }
}

/**
 * Update both human and bot ratings after a bot game
 */
async function updateUserRating(
  db: ReturnType<typeof createDb>,
  userId: string,
  game: ActiveGameRow,
  outcome: GameOutcome,
  now: number
) {
  const user = await db.query.users.findFirst({
    where: eq(users.id, userId),
    columns: {
      rating: true,
      gamesPlayed: true,
    },
  })

  if (!user) return

  const isPlayer1 = game.player1_id === userId
  const userRating = isPlayer1 ? game.player1_rating : game.player2_rating
  const botRating = isPlayer1 ? game.player2_rating : game.player1_rating

  const result = calculateNewRating(userRating, botRating, outcome, user.gamesPlayed)

  const gameId = crypto.randomUUID()
  const ratingHistoryId = crypto.randomUUID()
  const moves = JSON.parse(game.moves) as number[]

  // Build player_bot_stats update if we have a persona
  if (game.bot_persona_id) {
    // Get existing stats to calculate streaks
    const existingStats = await db.query.playerBotStats.findFirst({
      where: and(
        eq(playerBotStats.userId, userId),
        eq(playerBotStats.botPersonaId, game.bot_persona_id)
      ),
    })

    if (existingStats) {
      // Update existing record
      let newStreak = existingStats.currentStreak
      if (outcome === 'win') {
        newStreak = newStreak >= 0 ? newStreak + 1 : 1
      } else if (outcome === 'loss') {
        newStreak = newStreak <= 0 ? newStreak - 1 : -1
      } else {
        newStreak = 0 // Draw resets streak
      }

      const newBestWinStreak = Math.max(existingStats.bestWinStreak, newStreak > 0 ? newStreak : 0)
      const firstWinAt = outcome === 'win' && !existingStats.firstWinAt ? now : existingStats.firstWinAt

      await db.update(playerBotStats)
        .set({
          wins: existingStats.wins + (outcome === 'win' ? 1 : 0),
          losses: existingStats.losses + (outcome === 'loss' ? 1 : 0),
          draws: existingStats.draws + (outcome === 'draw' ? 1 : 0),
          currentStreak: newStreak,
          bestWinStreak: newBestWinStreak,
          firstWinAt,
          lastPlayedAt: now,
        })
        .where(
          and(
            eq(playerBotStats.userId, userId),
            eq(playerBotStats.botPersonaId, game.bot_persona_id)
          )
        )
    } else {
      // Insert new record
      const initialStreak = outcome === 'win' ? 1 : outcome === 'loss' ? -1 : 0
      const firstWinAt = outcome === 'win' ? now : null

      await db.insert(playerBotStats).values({
        userId,
        botPersonaId: game.bot_persona_id,
        wins: outcome === 'win' ? 1 : 0,
        losses: outcome === 'loss' ? 1 : 0,
        draws: outcome === 'draw' ? 1 : 0,
        currentStreak: initialStreak,
        bestWinStreak: outcome === 'win' ? 1 : 0,
        firstWinAt,
        lastPlayedAt: now,
      })
    }
  }

  // Update human game record and stats
  await db.insert(games).values({
    id: gameId,
    userId,
    outcome,
    moves: JSON.stringify(moves),
    moveCount: moves.length,
    ratingChange: result.ratingChange,
    opponentType: 'ai',
    aiDifficulty: game.bot_difficulty,
    playerNumber: isPlayer1 ? 1 : 2,
    botPersonaId: game.bot_persona_id,
    createdAt: now,
  })

  await db.update(users)
    .set({
      rating: result.newRating,
      gamesPlayed: user.gamesPlayed + 1,
      wins: user.wins + (outcome === 'win' ? 1 : 0),
      losses: user.losses + (outcome === 'loss' ? 1 : 0),
      draws: user.draws + (outcome === 'draw' ? 1 : 0),
      updatedAt: now,
    })
    .where(eq(users.id, userId))

  await db.insert(ratingHistory).values({
    id: ratingHistoryId,
    userId,
    gameId,
    ratingBefore: userRating,
    ratingAfter: result.newRating,
    ratingChange: result.ratingChange,
    createdAt: now,
  })

  // Update bot's rating if this is a real bot user (not legacy bot-opponent)
  const botUserId = getBotUserId(game)
  if (botUserId && botUserId !== 'bot-opponent') {
    // Get bot's current stats for K-factor calculation
    const botUser = await db.query.users.findFirst({
      where: and(eq(users.id, botUserId), eq(users.isBot, 1)),
      columns: {
        rating: true,
        gamesPlayed: true,
        wins: true,
        losses: true,
        draws: true,
      },
    })

    if (botUser) {
      // Calculate bot's rating change (opposite outcome)
      const botOutcome: GameOutcome = outcome === 'win' ? 'loss' : outcome === 'loss' ? 'win' : 'draw'
      const botResult = calculateNewRating(botRating, userRating, botOutcome, botUser.gamesPlayed)

      const botGameId = crypto.randomUUID()
      const botRatingHistoryId = crypto.randomUUID()

      // Game record for bot
      await db.insert(games).values({
        id: botGameId,
        userId: botUserId,
        outcome: botOutcome,
        moves: JSON.stringify(moves),
        moveCount: moves.length,
        ratingChange: botResult.ratingChange,
        opponentType: 'human',
        opponentId: userId,
        playerNumber: isPlayer1 ? 2 : 1,
        createdAt: now,
      })

      // Update bot user stats
      await db.update(users)
        .set({
          rating: botResult.newRating,
          gamesPlayed: botUser.gamesPlayed + 1,
          wins: botUser.wins + (botOutcome === 'win' ? 1 : 0),
          losses: botUser.losses + (botOutcome === 'loss' ? 1 : 0),
          draws: botUser.draws + (botOutcome === 'draw' ? 1 : 0),
          updatedAt: now,
        })
        .where(and(eq(users.id, botUserId), eq(users.isBot, 1)))

      // Rating history for bot
      await db.insert(ratingHistory).values({
        id: botRatingHistoryId,
        userId: botUserId,
        gameId: botGameId,
        ratingBefore: botRating,
        ratingAfter: botResult.newRating,
        ratingChange: botResult.ratingChange,
        createdAt: now,
      })

      // Also update bot_personas table to keep it in sync
      if (game.bot_persona_id) {
        const persona = await db.query.botPersonas.findFirst({
          where: eq(botPersonas.id, game.bot_persona_id),
          columns: {
            gamesPlayed: true,
            wins: true,
            losses: true,
            draws: true,
          },
        })

        if (persona) {
          await db.update(botPersonas)
            .set({
              currentElo: botResult.newRating,
              gamesPlayed: persona.gamesPlayed + 1,
              wins: persona.wins + (botOutcome === 'win' ? 1 : 0),
              losses: persona.losses + (botOutcome === 'loss' ? 1 : 0),
              draws: persona.draws + (botOutcome === 'draw' ? 1 : 0),
              updatedAt: now,
            })
            .where(eq(botPersonas.id, game.bot_persona_id))
        }
      }
    }
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
