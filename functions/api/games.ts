/**
 * Games API endpoint
 *
 * GET /api/games - List user's games
 * POST /api/games - Save a completed game
 */

import { validateSession, errorResponse, jsonResponse } from '../lib/auth'
import { calculateNewRating, GameOutcome } from '../lib/elo'
import { z } from 'zod'

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
    const games = await DB.prepare(`
      SELECT id, outcome, moves, move_count, rating_change, opponent_type, ai_difficulty, player_number, created_at
      FROM games
      WHERE user_id = ?
      ORDER BY created_at DESC
      LIMIT ? OFFSET ?
    `)
      .bind(session.userId, limit, offset)
      .all<GameRow>()

    // Get total count for pagination
    const countResult = await DB.prepare(`
      SELECT COUNT(*) as count FROM games WHERE user_id = ?
    `)
      .bind(session.userId)
      .first<{ count: number }>()

    const total = countResult?.count || 0

    // Parse moves JSON for each game
    const parsedGames = games.results.map((game) => ({
      id: game.id,
      outcome: game.outcome,
      moves: JSON.parse(game.moves),
      moveCount: game.move_count,
      ratingChange: game.rating_change ?? 0,
      opponentType: game.opponent_type,
      aiDifficulty: game.ai_difficulty,
      playerNumber: game.player_number,
      createdAt: game.created_at,
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

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Parse and validate request body
    const body = await context.request.json()
    const parseResult = createGameSchema.safeParse(body)

    if (!parseResult.success) {
      return errorResponse(parseResult.error.errors[0].message, 400)
    }

    const { outcome, moves, opponentType, aiDifficulty, playerNumber } = parseResult.data

    // Get user's current rating and stats
    const user = await DB.prepare(`
      SELECT rating, games_played, wins, losses, draws
      FROM users WHERE id = ?
    `)
      .bind(session.userId)
      .first<UserRatingRow>()

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
        user.games_played
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
    await DB.batch([
      // Insert the game with all fields
      DB.prepare(`
        INSERT INTO games (id, user_id, outcome, moves, move_count, rating_change, opponent_type, ai_difficulty, player_number, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).bind(
        gameId,
        session.userId,
        outcome,
        JSON.stringify(moves),
        moves.length,
        ratingChange,
        opponentType,
        aiDifficulty ?? null,
        playerNumber,
        now
      ),

      // Update user's rating and stats
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
        newRating,
        winsIncrement,
        lossesIncrement,
        drawsIncrement,
        now,
        session.userId
      ),

      // Record rating history (only if rating changed)
      ...(ratingChange !== 0
        ? [
            DB.prepare(`
              INSERT INTO rating_history (id, user_id, game_id, rating_before, rating_after, rating_change, created_at)
              VALUES (?, ?, ?, ?, ?, ?, ?)
            `).bind(
              ratingHistoryId,
              session.userId,
              gameId,
              user.rating,
              newRating,
              ratingChange,
              now
            ),
          ]
        : []),
    ])

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
