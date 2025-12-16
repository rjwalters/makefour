/**
 * POST /api/match/:id/resign - Resign from an active game
 */

import { validateSession, errorResponse, jsonResponse } from '../../../lib/auth'
import { calculateNewRating, type GameOutcome } from '../../../lib/elo'

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
}

interface UserRow {
  id: string
  rating: number
  games_played: number
}

export async function onRequestPost(context: EventContext<Env, any, any>) {
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
             winner, player1_rating, player2_rating
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
    const winner = playerNumber === 1 ? '2' : '1' // Opponent wins
    const now = Date.now()

    // Update the game
    await DB.prepare(`
      UPDATE active_games
      SET status = 'completed', winner = ?, updated_at = ?
      WHERE id = ?
    `).bind(winner, now, gameId).run()

    // Update ELO ratings for ranked games
    if (game.mode === 'ranked') {
      await updateRatingsOnResign(DB, game, playerNumber, now)
    }

    return jsonResponse({
      success: true,
      status: 'completed',
      winner,
      resigned: true,
    })
  } catch (error) {
    console.error('POST /api/match/:id/resign error:', error)
    return errorResponse('Internal server error', 500)
  }
}

async function updateRatingsOnResign(
  DB: D1Database,
  game: ActiveGameRow,
  resigningPlayer: number,
  now: number
) {
  const moves = JSON.parse(game.moves) as number[]

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

  // Resigning player loses
  const player1Outcome: GameOutcome = resigningPlayer === 1 ? 'loss' : 'win'
  const player2Outcome: GameOutcome = resigningPlayer === 2 ? 'loss' : 'win'

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

  const game1Id = crypto.randomUUID()
  const game2Id = crypto.randomUUID()
  const ratingHistory1Id = crypto.randomUUID()
  const ratingHistory2Id = crypto.randomUUID()

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
        updated_at = ?
      WHERE id = ?
    `).bind(
      player1Result.newRating,
      player1Outcome === 'win' ? 1 : 0,
      player1Outcome === 'loss' ? 1 : 0,
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
        updated_at = ?
      WHERE id = ?
    `).bind(
      player2Result.newRating,
      player2Outcome === 'win' ? 1 : 0,
      player2Outcome === 'loss' ? 1 : 0,
      now,
      game.player2_id
    ),

    // Rating history
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
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  })
}
