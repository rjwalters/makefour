/**
 * GET /api/matchmaking/status - Check matchmaking status and try to find a match
 *
 * This endpoint is polled by the client to:
 * 1. Check if still in queue
 * 2. Attempt to find a suitable opponent
 * 3. Create a game if match is found
 */

import { validateSession, errorResponse, jsonResponse } from '../../lib/auth'
import { createDb } from '../../../shared/db/client'
import { users, matchmakingQueue, activeGames } from '../../../shared/db/schema'
import { eq, and, or, ne, sql } from 'drizzle-orm'
import { DEFAULT_TIME_CONTROL_MS } from '../../lib/types'

interface Env {
  DB: D1Database
}

// Tolerance expands by 50 rating points every 10 seconds
const TOLERANCE_EXPANSION_RATE = 50
const TOLERANCE_EXPANSION_INTERVAL = 10000 // 10 seconds
const MAX_TOLERANCE = 500

// Time after which bot match becomes available
const BOT_READY_THRESHOLD_MS = 5000 // 5 seconds

export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    const db = createDb(DB)

    // First check if user is already in an active game
    const activeGame = await db.query.activeGames.findFirst({
      where: and(
        or(
          eq(activeGames.player1Id, session.userId),
          eq(activeGames.player2Id, session.userId)
        ),
        eq(activeGames.status, 'active')
      ),
      columns: { id: true, player1Id: true, player2Id: true }
    })

    if (activeGame) {
      // User already has an active game - return it
      return jsonResponse({
        status: 'matched',
        gameId: activeGame.id,
        playerNumber: activeGame.player1Id === session.userId ? 1 : 2,
      })
    }

    // Check if user is in queue
    const queueEntry = await db.query.matchmakingQueue.findFirst({
      where: eq(matchmakingQueue.userId, session.userId)
    })

    if (!queueEntry) {
      return jsonResponse({
        status: 'not_queued',
      })
    }

    const now = Date.now()
    const waitTime = now - queueEntry.joinedAt

    // Calculate current tolerance based on wait time
    const toleranceExpansions = Math.floor(waitTime / TOLERANCE_EXPANSION_INTERVAL)
    const currentTolerance = Math.min(
      queueEntry.initialTolerance + toleranceExpansions * TOLERANCE_EXPANSION_RATE,
      MAX_TOLERANCE
    )

    // Try to find a match
    // Look for other players in the same mode within rating tolerance
    // Exclude self, order by closest rating then longest wait
    const potentialMatches = await db
      .select()
      .from(matchmakingQueue)
      .where(
        and(
          ne(matchmakingQueue.userId, session.userId),
          eq(matchmakingQueue.mode, queueEntry.mode),
          sql`ABS(${matchmakingQueue.rating} - ${queueEntry.rating}) <= ${currentTolerance}`
        )
      )
      .orderBy(
        sql`ABS(${matchmakingQueue.rating} - ${queueEntry.rating}) ASC`,
        matchmakingQueue.joinedAt
      )
      .limit(10)

    // Find first match where both players are within each other's tolerance
    let matchedOpponent = null

    for (const opponent of potentialMatches) {
      const opponentWaitTime = now - opponent.joinedAt
      const opponentExpansions = Math.floor(opponentWaitTime / TOLERANCE_EXPANSION_INTERVAL)
      const opponentTolerance = Math.min(
        opponent.initialTolerance + opponentExpansions * TOLERANCE_EXPANSION_RATE,
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
      // Check if bot match is ready (waited 60 seconds)
      const botMatchReady = waitTime >= BOT_READY_THRESHOLD_MS

      return jsonResponse({
        status: 'queued',
        waitTime,
        currentTolerance,
        mode: queueEntry.mode,
        rating: queueEntry.rating,
        botMatchReady,
      })
    }

    // Match found! Create the game
    const gameId = crypto.randomUUID()

    // Randomly assign player 1 and player 2
    const player1IsUser = Math.random() < 0.5
    const player1Id = player1IsUser ? session.userId : matchedOpponent.userId
    const player2Id = player1IsUser ? matchedOpponent.userId : session.userId
    const player1Rating = player1IsUser ? queueEntry.rating : matchedOpponent.rating
    const player2Rating = player1IsUser ? matchedOpponent.rating : queueEntry.rating
    // Game is spectatable only if both players allow it
    const gameSpectatable = queueEntry.spectatable === 1 && matchedOpponent.spectatable === 1 ? 1 : 0

    // Get opponent info for response
    const opponent = await db.query.users.findFirst({
      where: eq(users.id, matchedOpponent.userId),
      columns: { id: true, rating: true }
    })

    try {
      // Use a batch to atomically create game and remove both from queue
      await db.batch([
        // Create the active game with timer initialized
        db.insert(activeGames).values({
          id: gameId,
          player1Id,
          player2Id,
          moves: '[]',
          currentTurn: 1,
          status: 'active',
          mode: queueEntry.mode,
          player1Rating,
          player2Rating,
          spectatable: gameSpectatable,
          spectatorCount: 0,
          lastMoveAt: now,
          timeControlMs: DEFAULT_TIME_CONTROL_MS,
          player1TimeMs: DEFAULT_TIME_CONTROL_MS,
          player2TimeMs: DEFAULT_TIME_CONTROL_MS,
          turnStartedAt: now,
          isBotGame: 0,
          createdAt: now,
          updatedAt: now,
        }),

        // Remove both players from queue
        db.delete(matchmakingQueue).where(eq(matchmakingQueue.userId, session.userId)),
        db.delete(matchmakingQueue).where(eq(matchmakingQueue.userId, matchedOpponent.userId)),
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
