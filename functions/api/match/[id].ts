/**
 * Match API endpoint for active online games
 *
 * GET /api/match/:id - Get game state (poll for updates)
 * POST /api/match/:id - Submit a move
 */

import { validateSession, errorResponse, jsonResponse } from '../../lib/auth'
import { replayMoves, makeMove, isValidMove, createGameState } from '../../lib/game'
import { calculateNewRating, type GameOutcome } from '../../lib/elo'
import { z } from 'zod'
import { createDb } from '../../../shared/db/client'
import { users, games, activeGames, ratingHistory } from '../../../shared/db/schema'
import { eq, and } from 'drizzle-orm'

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
  last_move_at: number
  // Timer fields (null = untimed game)
  time_control_ms: number | null
  player1_time_ms: number | null
  player2_time_ms: number | null
  turn_started_at: number | null
  // Bot game fields
  is_bot_game: number
  bot_difficulty: string | null
  created_at: number
  updated_at: number
}

interface UserRow {
  id: string
  rating: number
  games_played: number
}

const moveSchema = z.object({
  column: z.number().int().min(0).max(6),
})

/**
 * Check if the current player's time has expired and handle timeout
 * Returns the updated game if timeout occurred, null otherwise
 */
async function checkAndHandleTimeout(
  DB: D1Database,
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

    const db = createDb(DB)
    await db.update(activeGames)
      .set({
        status: 'completed',
        winner,
        player1TimeMs: game.current_turn === 1 ? 0 : game.player1_time_ms,
        player2TimeMs: game.current_turn === 2 ? 0 : game.player2_time_ms,
        updatedAt: now,
      })
      .where(eq(activeGames.id, game.id))

    // Update ELO if ranked
    if (game.mode === 'ranked') {
      const moves = JSON.parse(game.moves) as number[]
      await updateRatings(DB, { ...game, status: 'completed', winner }, winner, moves, now)
    }

    return { timedOut: true, winner }
  }

  return { timedOut: false }
}

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

  // Active player's time decreases, inactive player's time stays the same
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
 * GET /api/match/:id - Get current game state
 */
export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB } = context.env
  const db = createDb(DB)
  const gameId = context.params.id as string

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Get the game
    const game = await db.query.activeGames.findFirst({
      where: eq(activeGames.id, gameId),
    })

    if (!game) {
      return errorResponse('Game not found', 404)
    }

    // Verify user is a participant
    if (game.player1Id !== session.userId && game.player2Id !== session.userId) {
      return errorResponse('You are not a participant in this game', 403)
    }

    const now = Date.now()
    const playerNumber = game.player1Id === session.userId ? 1 : 2
    const opponentId = playerNumber === 1 ? game.player2Id : game.player1Id

    // Get opponent's username for rematch functionality
    let opponentUsername: string | null = null
    if (!game.isBotGame) {
      const opponent = await db.query.users.findFirst({
        where: eq(users.id, opponentId),
        columns: { username: true },
      })
      opponentUsername = opponent?.username ?? null
    }

    // Convert to ActiveGameRow format for compatibility with helper functions
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
      created_at: game.createdAt,
      updated_at: game.updatedAt,
    }

    // Check for timeout (this may end the game)
    const timeoutResult = await checkAndHandleTimeout(DB, gameRow, now)

    // If timeout occurred, update game state for response
    let status = game.status
    let winner = game.winner
    if (timeoutResult.timedOut) {
      status = 'completed'
      winner = timeoutResult.winner ?? null
    }

    const moves = JSON.parse(game.moves) as number[]

    // Reconstruct board state from moves
    const gameState = moves.length > 0 ? replayMoves(moves) : createGameState()

    // Calculate current time remaining
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
      opponentUsername,
      lastMoveAt: game.lastMoveAt,
      createdAt: game.createdAt,
      isYourTurn: status === 'active' && game.currentTurn === playerNumber,
      // Timer fields
      timeControlMs: game.timeControlMs,
      player1TimeMs: timeoutResult.timedOut
        ? (game.currentTurn === 1 ? 0 : game.player1TimeMs)
        : timeRemaining.player1TimeMs,
      player2TimeMs: timeoutResult.timedOut
        ? (game.currentTurn === 2 ? 0 : game.player2TimeMs)
        : timeRemaining.player2TimeMs,
      turnStartedAt: game.turnStartedAt,
      // Bot game fields
      isBotGame: game.isBotGame === 1,
      botDifficulty: game.botDifficulty,
    })
  } catch (error) {
    console.error('GET /api/match/:id error:', error)
    return errorResponse('Internal server error', 500)
  }
}

/**
 * POST /api/match/:id - Submit a move
 */
export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB } = context.env
  const db = createDb(DB)
  const gameId = context.params.id as string

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Parse request body
    const body = await context.request.json()
    const parseResult = moveSchema.safeParse(body)

    if (!parseResult.success) {
      return errorResponse(parseResult.error.errors[0].message, 400)
    }

    const { column } = parseResult.data

    // Get the game
    const game = await db.query.activeGames.findFirst({
      where: eq(activeGames.id, gameId),
    })

    if (!game) {
      return errorResponse('Game not found', 404)
    }

    // Verify user is a participant
    if (game.player1Id !== session.userId && game.player2Id !== session.userId) {
      return errorResponse('You are not a participant in this game', 403)
    }

    // Check game is active
    if (game.status !== 'active') {
      return errorResponse('Game is not active', 400)
    }

    const playerNumber = game.player1Id === session.userId ? 1 : 2
    const now = Date.now()

    // Check it's their turn
    if (game.currentTurn !== playerNumber) {
      return errorResponse('Not your turn', 400)
    }

    // Convert to ActiveGameRow format for compatibility with helper functions
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
      created_at: game.createdAt,
      updated_at: game.updatedAt,
    }

    // Time tracking for timed games
    let newPlayer1Time = game.player1TimeMs
    let newPlayer2Time = game.player2TimeMs

    if (game.timeControlMs !== null && game.turnStartedAt !== null) {
      const elapsed = now - game.turnStartedAt
      const currentPlayerTime = playerNumber === 1 ? game.player1TimeMs : game.player2TimeMs

      if (currentPlayerTime !== null) {
        const timeRemaining = currentPlayerTime - elapsed

        // Check if player ran out of time
        if (timeRemaining <= 0) {
          // Time expired - opponent wins
          const winner = playerNumber === 1 ? '2' : '1'

          await db.update(activeGames)
            .set({
              status: 'completed',
              winner,
              player1TimeMs: playerNumber === 1 ? 0 : game.player1TimeMs,
              player2TimeMs: playerNumber === 2 ? 0 : game.player2TimeMs,
              updatedAt: now,
            })
            .where(eq(activeGames.id, gameId))

          // Update ELO if ranked
          if (game.mode === 'ranked') {
            const moves = JSON.parse(game.moves) as number[]
            await updateRatings(DB, { ...gameRow, status: 'completed', winner }, winner, moves, now)
          }

          return errorResponse('Time expired', 400)
        }

        // Deduct elapsed time from current player
        if (playerNumber === 1) {
          newPlayer1Time = timeRemaining
        } else {
          newPlayer2Time = timeRemaining
        }
      }
    }

    // Validate and apply the move
    const moves = JSON.parse(game.moves) as number[]
    const currentState = moves.length > 0 ? replayMoves(moves) : createGameState()

    if (!currentState) {
      return errorResponse('Invalid game state', 500)
    }

    if (!isValidMove(currentState.board, column)) {
      return errorResponse('Invalid move: column is full or out of bounds', 400)
    }

    const newState = makeMove(currentState, column)
    if (!newState) {
      return errorResponse('Invalid move', 400)
    }

    const newMoves = [...moves, column]
    const nextTurn = playerNumber === 1 ? 2 : 1

    // Check for game end
    let newStatus = 'active'
    let winner: string | null = null

    if (newState.winner !== null) {
      newStatus = 'completed'
      winner = newState.winner === 'draw' ? 'draw' : String(newState.winner)
    }

    // Update the game (including timer fields)
    await db.update(activeGames)
      .set({
        moves: JSON.stringify(newMoves),
        currentTurn: nextTurn,
        status: newStatus,
        winner,
        lastMoveAt: now,
        updatedAt: now,
        player1TimeMs: newPlayer1Time,
        player2TimeMs: newPlayer2Time,
        turnStartedAt: newStatus === 'active' ? now : game.turnStartedAt,
      })
      .where(eq(activeGames.id, gameId))

    // If game is completed, update ELO ratings
    if (newStatus === 'completed' && game.mode === 'ranked') {
      await updateRatings(DB, gameRow, winner, newMoves, now)
    }

    // Calculate time remaining for response
    const responsePlayer1Time = newStatus === 'completed' ? newPlayer1Time : newPlayer1Time
    const responsePlayer2Time = newStatus === 'completed' ? newPlayer2Time : newPlayer2Time

    return jsonResponse({
      success: true,
      moves: newMoves,
      board: newState.board,
      currentTurn: nextTurn,
      status: newStatus,
      winner,
      isYourTurn: newStatus === 'active' && nextTurn === playerNumber,
      // Timer fields
      timeControlMs: game.timeControlMs,
      player1TimeMs: responsePlayer1Time,
      player2TimeMs: responsePlayer2Time,
      turnStartedAt: newStatus === 'active' ? now : game.turnStartedAt,
    })
  } catch (error) {
    console.error('POST /api/match/:id error:', error)
    return errorResponse('Internal server error', 500)
  }
}

/**
 * Update ELO ratings after game completion
 */
async function updateRatings(
  DB: D1Database,
  game: ActiveGameRow,
  winner: string | null,
  moves: number[],
  now: number
) {
  const db = createDb(DB)

  // Get both players' current stats
  const [player1, player2] = await Promise.all([
    db.query.users.findFirst({
      where: eq(users.id, game.player1_id),
      columns: { id: true, rating: true, gamesPlayed: true, wins: true, losses: true, draws: true },
    }),
    db.query.users.findFirst({
      where: eq(users.id, game.player2_id),
      columns: { id: true, rating: true, gamesPlayed: true, wins: true, losses: true, draws: true },
    }),
  ])

  if (!player1 || !player2) {
    console.error('Could not find players for rating update')
    return
  }

  // Determine outcomes
  let player1Outcome: GameOutcome
  let player2Outcome: GameOutcome

  if (winner === 'draw') {
    player1Outcome = 'draw'
    player2Outcome = 'draw'
  } else if (winner === '1') {
    player1Outcome = 'win'
    player2Outcome = 'loss'
  } else {
    player1Outcome = 'loss'
    player2Outcome = 'win'
  }

  // Calculate new ratings
  const player1Result = calculateNewRating(
    game.player1_rating,
    game.player2_rating,
    player1Outcome,
    player1.gamesPlayed
  )
  const player2Result = calculateNewRating(
    game.player2_rating,
    game.player1_rating,
    player2Outcome,
    player2.gamesPlayed
  )

  // Generate IDs for records
  const game1Id = crypto.randomUUID()
  const game2Id = crypto.randomUUID()
  const ratingHistory1Id = crypto.randomUUID()
  const ratingHistory2Id = crypto.randomUUID()

  // Update stats for both players
  await db.batch([
    // Player 1 game record
    db.insert(games).values({
      id: game1Id,
      userId: game.player1_id,
      outcome: player1Outcome,
      moves: JSON.stringify(moves),
      moveCount: moves.length,
      ratingChange: player1Result.ratingChange,
      opponentType: 'human',
      playerNumber: 1,
      createdAt: now,
    }),

    // Player 2 game record
    db.insert(games).values({
      id: game2Id,
      userId: game.player2_id,
      outcome: player2Outcome,
      moves: JSON.stringify(moves),
      moveCount: moves.length,
      ratingChange: player2Result.ratingChange,
      opponentType: 'human',
      playerNumber: 2,
      createdAt: now,
    }),

    // Update player 1 stats
    db.update(users)
      .set({
        rating: player1Result.newRating,
        gamesPlayed: player1.gamesPlayed + 1,
        wins: player1.wins + (player1Outcome === 'win' ? 1 : 0),
        losses: player1.losses + (player1Outcome === 'loss' ? 1 : 0),
        draws: player1.draws + (player1Outcome === 'draw' ? 1 : 0),
        updatedAt: now,
      })
      .where(eq(users.id, game.player1_id)),

    // Update player 2 stats
    db.update(users)
      .set({
        rating: player2Result.newRating,
        gamesPlayed: player2.gamesPlayed + 1,
        wins: player2.wins + (player2Outcome === 'win' ? 1 : 0),
        losses: player2.losses + (player2Outcome === 'loss' ? 1 : 0),
        draws: player2.draws + (player2Outcome === 'draw' ? 1 : 0),
        updatedAt: now,
      })
      .where(eq(users.id, game.player2_id)),

    // Rating history for player 1
    db.insert(ratingHistory).values({
      id: ratingHistory1Id,
      userId: game.player1_id,
      gameId: game1Id,
      ratingBefore: game.player1_rating,
      ratingAfter: player1Result.newRating,
      ratingChange: player1Result.ratingChange,
      createdAt: now,
    }),

    // Rating history for player 2
    db.insert(ratingHistory).values({
      id: ratingHistory2Id,
      userId: game.player2_id,
      gameId: game2Id,
      ratingBefore: game.player2_rating,
      ratingAfter: player2Result.newRating,
      ratingChange: player2Result.ratingChange,
      createdAt: now,
    }),
  ] as any)
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
