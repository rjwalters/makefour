/**
 * Match API endpoint for active online games
 *
 * GET /api/match/:id - Get game state (poll for updates)
 * POST /api/match/:id - Submit a move
 */

import { validateSession, errorResponse, jsonResponse } from '../../lib/auth'
import { replayMoves, makeMove, isValidMove, createGameState } from '../../lib/game'
import { calculateNewRating, GameOutcome } from '../../lib/elo'
import { z } from 'zod'

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
 * GET /api/match/:id - Get current game state
 */
export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB } = context.env
  const gameId = context.params.id as string

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Get the game
    const game = await DB.prepare(`
      SELECT id, player1_id, player2_id, moves, current_turn, status, mode,
             winner, player1_rating, player2_rating, last_move_at, created_at, updated_at
      FROM active_games
      WHERE id = ?
    `)
      .bind(gameId)
      .first<ActiveGameRow>()

    if (!game) {
      return errorResponse('Game not found', 404)
    }

    // Verify user is a participant
    if (game.player1_id !== session.userId && game.player2_id !== session.userId) {
      return errorResponse('You are not a participant in this game', 403)
    }

    const playerNumber = game.player1_id === session.userId ? 1 : 2
    const moves = JSON.parse(game.moves) as number[]

    // Reconstruct board state from moves
    const gameState = moves.length > 0 ? replayMoves(moves) : createGameState()

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
    const game = await DB.prepare(`
      SELECT id, player1_id, player2_id, moves, current_turn, status, mode,
             winner, player1_rating, player2_rating, last_move_at, created_at, updated_at
      FROM active_games
      WHERE id = ?
    `)
      .bind(gameId)
      .first<ActiveGameRow>()

    if (!game) {
      return errorResponse('Game not found', 404)
    }

    // Verify user is a participant
    if (game.player1_id !== session.userId && game.player2_id !== session.userId) {
      return errorResponse('You are not a participant in this game', 403)
    }

    // Check game is active
    if (game.status !== 'active') {
      return errorResponse('Game is not active', 400)
    }

    const playerNumber = game.player1_id === session.userId ? 1 : 2

    // Check it's their turn
    if (game.current_turn !== playerNumber) {
      return errorResponse('Not your turn', 400)
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

    const now = Date.now()
    const newMoves = [...moves, column]
    const nextTurn = playerNumber === 1 ? 2 : 1

    // Check for game end
    let newStatus = 'active'
    let winner: string | null = null

    if (newState.winner !== null) {
      newStatus = 'completed'
      winner = newState.winner === 'draw' ? 'draw' : String(newState.winner)
    }

    // Update the game
    await DB.prepare(`
      UPDATE active_games
      SET moves = ?, current_turn = ?, status = ?, winner = ?,
          last_move_at = ?, updated_at = ?
      WHERE id = ?
    `).bind(
      JSON.stringify(newMoves),
      nextTurn,
      newStatus,
      winner,
      now,
      now,
      gameId
    ).run()

    // If game is completed, update ELO ratings
    if (newStatus === 'completed' && game.mode === 'ranked') {
      await updateRatings(DB, game, winner, newMoves, now)
    }

    return jsonResponse({
      success: true,
      moves: newMoves,
      board: newState.board,
      currentTurn: nextTurn,
      status: newStatus,
      winner,
      isYourTurn: newStatus === 'active' && nextTurn === playerNumber,
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
  // Get both players' current stats
  const [player1, player2] = await Promise.all([
    DB.prepare('SELECT id, rating, games_played FROM users WHERE id = ?')
      .bind(game.player1_id)
      .first<UserRow>(),
    DB.prepare('SELECT id, rating, games_played FROM users WHERE id = ?')
      .bind(game.player2_id)
      .first<UserRow>(),
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
    player1.games_played
  )
  const player2Result = calculateNewRating(
    game.player2_rating,
    game.player1_rating,
    player2Outcome,
    player2.games_played
  )

  // Generate IDs for records
  const game1Id = crypto.randomUUID()
  const game2Id = crypto.randomUUID()
  const ratingHistory1Id = crypto.randomUUID()
  const ratingHistory2Id = crypto.randomUUID()

  // Update stats for both players
  await DB.batch([
    // Player 1 game record
    DB.prepare(`
      INSERT INTO games (id, user_id, outcome, moves, move_count, rating_change, opponent_type, player_number, created_at)
      VALUES (?, ?, ?, ?, ?, ?, 'human', 1, ?)
    `).bind(
      game1Id,
      game.player1_id,
      player1Outcome,
      JSON.stringify(moves),
      moves.length,
      player1Result.ratingChange,
      now
    ),

    // Player 2 game record
    DB.prepare(`
      INSERT INTO games (id, user_id, outcome, moves, move_count, rating_change, opponent_type, player_number, created_at)
      VALUES (?, ?, ?, ?, ?, ?, 'human', 2, ?)
    `).bind(
      game2Id,
      game.player2_id,
      player2Outcome,
      JSON.stringify(moves),
      moves.length,
      player2Result.ratingChange,
      now
    ),

    // Update player 1 stats
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
      player1Result.newRating,
      player1Outcome === 'win' ? 1 : 0,
      player1Outcome === 'loss' ? 1 : 0,
      player1Outcome === 'draw' ? 1 : 0,
      now,
      game.player1_id
    ),

    // Update player 2 stats
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
      player2Result.newRating,
      player2Outcome === 'win' ? 1 : 0,
      player2Outcome === 'loss' ? 1 : 0,
      player2Outcome === 'draw' ? 1 : 0,
      now,
      game.player2_id
    ),

    // Rating history for player 1
    DB.prepare(`
      INSERT INTO rating_history (id, user_id, game_id, rating_before, rating_after, rating_change, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `).bind(
      ratingHistory1Id,
      game.player1_id,
      game1Id,
      game.player1_rating,
      player1Result.newRating,
      player1Result.ratingChange,
      now
    ),

    // Rating history for player 2
    DB.prepare(`
      INSERT INTO rating_history (id, user_id, game_id, rating_before, rating_after, rating_change, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `).bind(
      ratingHistory2Id,
      game.player2_id,
      game2Id,
      game.player2_rating,
      player2Result.newRating,
      player2Result.ratingChange,
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
