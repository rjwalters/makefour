/**
 * Games API endpoint
 *
 * GET /api/games - List user's games
 * POST /api/games - Save a completed game
 */

import { validateSession, errorResponse, jsonResponse } from '../lib/auth'
import { calculateNewRating, type GameOutcome } from '../lib/elo'
import { z } from 'zod'
import { createDb } from '../../shared/db/client'
import { users, games, ratingHistory } from '../../shared/db/schema'
import { eq, desc, count } from 'drizzle-orm'

interface Env {
  DB: D1Database
}

// AI difficulty ratings for ELO calculations (mapped to new difficulty names)
const AI_DIFFICULTY_RATINGS: Record<string, number> = {
  beginner: 800,
  intermediate: 1200,
  expert: 1600,
  perfect: 2000,
}

// Schema for creating a new game
const createGameSchema = z.object({
  outcome: z.enum(['win', 'loss', 'draw']),
  moves: z.array(z.number().int().min(0).max(6)),
  opponentType: z.enum(['human', 'ai']).default('ai'),
  aiDifficulty: z.enum(['beginner', 'intermediate', 'expert', 'perfect']).nullable().optional(),
  playerNumber: z.number().int().min(1).max(2).default(1),
})

// Schema for game from database
interface GameRow {
  id: string
  user_id: string
  outcome: string
  moves: string
  move_count: number
  rating_change: number | null
  opponent_type: string
  ai_difficulty: string | null
  player_number: number
  created_at: number
}

// Schema for user rating data from database
interface UserRatingRow {
  rating: number
  games_played: number
  wins: number
  losses: number
  draws: number
}

/**
 * GET /api/games - List user's games
 */
export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB } = context.env
  const db = createDb(DB)

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Get query params for pagination
    const url = new URL(context.request.url)
    const limit = Math.min(parseInt(url.searchParams.get('limit') || '20'), 100)
    const offset = parseInt(url.searchParams.get('offset') || '0')

    // Fetch games for user, ordered by most recent first
    const userGames = await db.query.games.findMany({
      where: eq(games.userId, session.userId),
      orderBy: desc(games.createdAt),
      limit,
      offset,
    })

    // Get total count for pagination
    const countResult = await db
      .select({ count: count() })
      .from(games)
      .where(eq(games.userId, session.userId))

    const total = countResult[0]?.count || 0

    // Parse moves JSON for each game
    const parsedGames = userGames.map((game) => ({
      id: game.id,
      outcome: game.outcome,
      moves: JSON.parse(game.moves),
      moveCount: game.moveCount,
      ratingChange: game.ratingChange ?? 0,
      opponentType: game.opponentType,
      aiDifficulty: game.aiDifficulty,
      playerNumber: game.playerNumber,
      createdAt: game.createdAt,
    }))

    return jsonResponse({
      games: parsedGames,
      pagination: {
        total,
        limit,
        offset,
        hasMore: offset + limit < total,
      },
    })
  } catch (error) {
    console.error('GET /api/games error:', error)
    return errorResponse('Internal server error', 500)
  }
}

/**
 * POST /api/games - Save a completed game and update ELO rating
 */
export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB } = context.env
  const db = createDb(DB)

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Parse and validate request body
    const body = await context.request.json()
    const parseResult = createGameSchema.safeParse(body)

    if (!parseResult.success) {
      // Zod v4 uses `issues` instead of `errors`
      const issues = parseResult.error.issues || (parseResult.error as unknown as { errors: unknown[] }).errors || []
      const message = (issues[0] as { message?: string })?.message || 'Invalid request data'
      return errorResponse(message, 400)
    }

    const { outcome, moves, opponentType, aiDifficulty, playerNumber } = parseResult.data

    // Get user's current rating and stats
    const user = await db.query.users.findFirst({
      where: eq(users.id, session.userId),
      columns: {
        rating: true,
        gamesPlayed: true,
        wins: true,
        losses: true,
        draws: true,
      },
    })

    if (!user) {
      return errorResponse('User not found', 404)
    }

    // Calculate new ELO rating (only for AI games)
    let ratingChange = 0
    let newRating = user.rating

    if (opponentType === 'ai' && aiDifficulty) {
      const opponentRating = AI_DIFFICULTY_RATINGS[aiDifficulty] ?? 1200
      const eloResult = calculateNewRating(
        user.rating,
        opponentRating,
        outcome as GameOutcome,
        user.gamesPlayed
      )
      ratingChange = eloResult.ratingChange
      newRating = eloResult.newRating
    }

    // Generate UUID for the game
    const gameId = crypto.randomUUID()
    const ratingHistoryId = crypto.randomUUID()
    const now = Date.now()

    // Update stats based on outcome
    const winsIncrement = outcome === 'win' ? 1 : 0
    const lossesIncrement = outcome === 'loss' ? 1 : 0
    const drawsIncrement = outcome === 'draw' ? 1 : 0

    // Use a batch to ensure all updates happen atomically
    const batchOperations = [
      // Insert the game with all fields
      db.insert(games).values({
        id: gameId,
        userId: session.userId,
        outcome,
        moves: JSON.stringify(moves),
        moveCount: moves.length,
        ratingChange,
        opponentType,
        aiDifficulty: aiDifficulty ?? null,
        playerNumber,
        createdAt: now,
      }),

      // Update user's rating and stats
      db.update(users)
        .set({
          rating: newRating,
          gamesPlayed: user.gamesPlayed + 1,
          wins: user.wins + winsIncrement,
          losses: user.losses + lossesIncrement,
          draws: user.draws + drawsIncrement,
          updatedAt: now,
        })
        .where(eq(users.id, session.userId)),
    ]

    // Record rating history (only if rating changed)
    if (ratingChange !== 0) {
      batchOperations.push(
        db.insert(ratingHistory).values({
          id: ratingHistoryId,
          userId: session.userId,
          gameId,
          ratingBefore: user.rating,
          ratingAfter: newRating,
          ratingChange,
          createdAt: now,
        })
      )
    }

    await db.batch(batchOperations as any)

    return jsonResponse(
      {
        id: gameId,
        outcome,
        moves,
        moveCount: moves.length,
        ratingChange,
        newRating,
        opponentType,
        aiDifficulty,
        playerNumber,
        createdAt: now,
      },
      201
    )
  } catch (error) {
    console.error('POST /api/games error:', error)
    return errorResponse('Internal server error', 500)
  }
}

/**
 * Handle OPTIONS for CORS preflight
 */
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
