/**
 * GET /api/matchmaking/status - Check matchmaking status and try to find a match
 *
 * This endpoint is polled by the client to:
 * 1. Check if still in queue
 * 2. Attempt to find a suitable opponent
 * 3. Create a game if match is found
 */

import { validateSession, errorResponse, jsonResponse } from '../../lib/auth'

interface Env {
  DB: D1Database
}

interface QueueEntry {
  id: string
  user_id: string
  rating: number
  mode: string
  initial_tolerance: number
  spectatable: number
  joined_at: number
}

interface UserRow {
  id: string
  email: string
  rating: number
}

interface ActiveGameRow {
  id: string
  player1_id: string
  player2_id: string
  status: string
}

// Tolerance expands by 50 rating points every 10 seconds
const TOLERANCE_EXPANSION_RATE = 50
const TOLERANCE_EXPANSION_INTERVAL = 10000 // 10 seconds
const MAX_TOLERANCE = 500

// Default time control: 5 minutes per player
const DEFAULT_TIME_CONTROL_MS = 300000

export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // First check if user is already in an active game
    const activeGame = await DB.prepare(`
      SELECT id, player1_id, player2_id, status
      FROM active_games
      WHERE (player1_id = ? OR player2_id = ?)
      AND status = 'active'
    `)
      .bind(session.userId, session.userId)
      .first<ActiveGameRow>()

    if (activeGame) {
      // User already has an active game - return it
      return jsonResponse({
        status: 'matched',
        gameId: activeGame.id,
        playerNumber: activeGame.player1_id === session.userId ? 1 : 2,
      })
    }

    // Check if user is in queue
    const queueEntry = await DB.prepare(`
      SELECT id, user_id, rating, mode, initial_tolerance, spectatable, joined_at
      FROM matchmaking_queue
      WHERE user_id = ?
    `)
      .bind(session.userId)
      .first<QueueEntry>()

    if (!queueEntry) {
      return jsonResponse({
        status: 'not_queued',
      })
    }

    const now = Date.now()
    const waitTime = now - queueEntry.joined_at

    // Calculate current tolerance based on wait time
    const toleranceExpansions = Math.floor(waitTime / TOLERANCE_EXPANSION_INTERVAL)
    const currentTolerance = Math.min(
      queueEntry.initial_tolerance + toleranceExpansions * TOLERANCE_EXPANSION_RATE,
      MAX_TOLERANCE
    )

    // Try to find a match
    // Look for other players in the same mode within rating tolerance
    // Exclude self, order by closest rating then longest wait
    const potentialMatches = await DB.prepare(`
      SELECT id, user_id, rating, mode, initial_tolerance, spectatable, joined_at
      FROM matchmaking_queue
      WHERE user_id != ?
      AND mode = ?
      AND ABS(rating - ?) <= ?
      ORDER BY ABS(rating - ?) ASC, joined_at ASC
      LIMIT 10
    `)
      .bind(
        session.userId,
        queueEntry.mode,
        queueEntry.rating,
        currentTolerance,
        queueEntry.rating
      )
      .all<QueueEntry>()

    // Find first match where both players are within each other's tolerance
    let matchedOpponent: QueueEntry | null = null

    for (const opponent of potentialMatches.results) {
      const opponentWaitTime = now - opponent.joined_at
      const opponentExpansions = Math.floor(opponentWaitTime / TOLERANCE_EXPANSION_INTERVAL)
      const opponentTolerance = Math.min(
        opponent.initial_tolerance + opponentExpansions * TOLERANCE_EXPANSION_RATE,
        MAX_TOLERANCE
      )

      // Check if we're within each other's tolerance
      const ratingDiff = Math.abs(queueEntry.rating - opponent.rating)
      if (ratingDiff <= currentTolerance && ratingDiff <= opponentTolerance) {
        matchedOpponent = opponent
        break
      }
    }

    if (!matchedOpponent) {
      // No match found yet
      return jsonResponse({
        status: 'queued',
        waitTime,
        currentTolerance,
        mode: queueEntry.mode,
        rating: queueEntry.rating,
      })
    }

    // Match found! Create the game
    const gameId = crypto.randomUUID()

    // Randomly assign player 1 and player 2
    const player1IsUser = Math.random() < 0.5
    const player1Id = player1IsUser ? session.userId : matchedOpponent.user_id
    const player2Id = player1IsUser ? matchedOpponent.user_id : session.userId
    const player1Rating = player1IsUser ? queueEntry.rating : matchedOpponent.rating
    const player2Rating = player1IsUser ? matchedOpponent.rating : queueEntry.rating
    // Game is spectatable only if both players allow it
    const gameSpectatable = queueEntry.spectatable === 1 && matchedOpponent.spectatable === 1 ? 1 : 0

    // Get opponent info for response
    const opponent = await DB.prepare(`
      SELECT id, email, rating FROM users WHERE id = ?
    `)
      .bind(matchedOpponent.user_id)
      .first<UserRow>()

    try {
      // Use a batch to atomically create game and remove both from queue
      await DB.batch([
        // Create the active game with timer initialized
        DB.prepare(`
          INSERT INTO active_games (
            id, player1_id, player2_id, moves, current_turn, status, mode,
            player1_rating, player2_rating, spectatable, spectator_count,
            last_move_at, time_control_ms, player1_time_ms, player2_time_ms,
            turn_started_at, is_bot_game, created_at, updated_at
          )
          VALUES (?, ?, ?, '[]', 1, 'active', ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, 0, ?, ?)
        `).bind(
          gameId,
          player1Id,
          player2Id,
          queueEntry.mode,
          player1Rating,
          player2Rating,
          gameSpectatable,
          now,
          DEFAULT_TIME_CONTROL_MS,
          DEFAULT_TIME_CONTROL_MS,
          DEFAULT_TIME_CONTROL_MS,
          now, // turn_started_at - clock starts immediately
          now,
          now
        ),

        // Remove both players from queue
        DB.prepare(`DELETE FROM matchmaking_queue WHERE user_id = ?`).bind(session.userId),
        DB.prepare(`DELETE FROM matchmaking_queue WHERE user_id = ?`).bind(matchedOpponent.user_id),
      ])

      return jsonResponse({
        status: 'matched',
        gameId,
        playerNumber: player1Id === session.userId ? 1 : 2,
        opponent: opponent
          ? {
              rating: opponent.rating,
            }
          : null,
        mode: queueEntry.mode,
      })
    } catch (dbError) {
      // If batch fails (e.g., race condition where opponent was already matched),
      // just return queued status - client will retry
      console.error('Match creation failed:', dbError)
      return jsonResponse({
        status: 'queued',
        waitTime,
        currentTolerance,
        mode: queueEntry.mode,
        rating: queueEntry.rating,
      })
    }
  } catch (error) {
    console.error('GET /api/matchmaking/status error:', error)
    return errorResponse('Internal server error', 500)
  }
}

export async function onRequestOptions() {
  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  })
}
