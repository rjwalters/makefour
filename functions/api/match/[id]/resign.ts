/**
 * POST /api/match/:id/resign - Resign from an active game
 */

import { validateSession, errorResponse, jsonResponse } from '../../../lib/auth'
import { calculateNewRating, type GameOutcome } from '../../../lib/elo'
import { type ActiveGameRow, type UserRow, safeParseMoves } from '../../../lib/types'
import { createDb } from '../../../../shared/db/client'
import { users, games, activeGames, ratingHistory } from '../../../../shared/db/schema'
import { eq } from 'drizzle-orm'

interface Env {
  DB: D1Database
}

export async function onRequestPost(context: EventContext<Env, any, any>) {
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

    // Check game is active
    if (game.status !== 'active') {
      return errorResponse('Game is not active', 400)
    }

    const playerNumber = game.player1Id === session.userId ? 1 : 2
    const winner = playerNumber === 1 ? '2' : '1' // Opponent wins
    const now = Date.now()

    // Update the game
    await db.update(activeGames)
      .set({
        status: 'completed',
        winner,
        updatedAt: now,
      })
      .where(eq(activeGames.id, gameId))

    // Convert to ActiveGameRow format for compatibility
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
    }

    // Update ELO ratings for ranked games
    if (game.mode === 'ranked') {
      await updateRatingsOnResign(DB, gameRow, playerNumber, now)
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
  const db = createDb(DB)
  const moves = safeParseMoves(game.moves)

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

  // Resigning player loses
  const player1Outcome: GameOutcome = resigningPlayer === 1 ? 'loss' : 'win'
  const player2Outcome: GameOutcome = resigningPlayer === 2 ? 'loss' : 'win'

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

  const game1Id = crypto.randomUUID()
  const game2Id = crypto.randomUUID()
  const ratingHistory1Id = crypto.randomUUID()
  const ratingHistory2Id = crypto.randomUUID()

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
        updatedAt: now,
      })
      .where(eq(users.id, game.player2_id)),

    // Rating history
    db.insert(ratingHistory).values({
      id: ratingHistory1Id,
      userId: game.player1_id,
      gameId: game1Id,
      ratingBefore: game.player1_rating,
      ratingAfter: player1Result.newRating,
      ratingChange: player1Result.ratingChange,
      createdAt: now,
    }),

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
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  })
}
