/**
 * POST /api/challenges/:id/accept - Accept an incoming challenge
 */

import { validateSession, errorResponse, jsonResponse } from '../../../lib/auth'
import { createDb } from '../../../../shared/db/client'
import { users, challenges, activeGames } from '../../../../shared/db/schema'
import { eq, and, or, lt, sql } from 'drizzle-orm'

interface Env {
  DB: D1Database
}

// Get display name from user: prefer username, fall back to email prefix
function getDisplayName(user: { username: string | null; email: string }): string {
  return user.username || user.email.split('@')[0]
}

export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB } = context.env
  const challengeId = context.params.id as string

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    const db = createDb(DB)
    const now = Date.now()

    // Get the challenge
    const challenge = await db.query.challenges.findFirst({
      where: eq(challenges.id, challengeId)
    })

    if (!challenge) {
      return errorResponse('Challenge not found', 404)
    }

    if (challenge.status !== 'pending') {
      return errorResponse(`Challenge is ${challenge.status}`, 400)
    }

    if (challenge.expiresAt < now) {
      await db.update(challenges)
        .set({ status: 'expired' })
        .where(eq(challenges.id, challengeId))
      return errorResponse('Challenge has expired', 400)
    }

    // Get acceptor info
    const acceptor = await db.query.users.findFirst({
      where: eq(users.id, session.userId),
      columns: { id: true, email: true, username: true, rating: true }
    })

    if (!acceptor) {
      return errorResponse('User not found', 404)
    }

    const acceptorDisplayName = getDisplayName(acceptor).toLowerCase()

    // Verify the acceptor is the target
    if (
      challenge.targetId !== session.userId &&
      challenge.targetUsername.toLowerCase() !== acceptorDisplayName
    ) {
      return errorResponse('You are not the target of this challenge', 403)
    }

    // Check if acceptor is already in a game
    const activeGame = await db.query.activeGames.findFirst({
      where: and(
        or(
          eq(activeGames.player1Id, session.userId),
          eq(activeGames.player2Id, session.userId)
        ),
        eq(activeGames.status, 'active')
      ),
      columns: { id: true }
    })

    if (activeGame) {
      return errorResponse('You are already in an active game', 409)
    }

    // Check if challenger is already in a game
    const challengerActiveGame = await db.query.activeGames.findFirst({
      where: and(
        or(
          eq(activeGames.player1Id, challenge.challengerId),
          eq(activeGames.player2Id, challenge.challengerId)
        ),
        eq(activeGames.status, 'active')
      ),
      columns: { id: true }
    })

    if (challengerActiveGame) {
      // Cancel the challenge - challenger is busy
      await db.update(challenges)
        .set({ status: 'cancelled' })
        .where(eq(challenges.id, challengeId))
      return errorResponse('Challenger is already in a game', 409)
    }

    // Create the game
    const gameId = crypto.randomUUID()
    const TIME_CONTROL_MS = 300000 // 5 minutes

    await db.insert(activeGames).values({
      id: gameId,
      player1Id: challenge.challengerId, // Challenger is player 1 (red)
      player2Id: session.userId, // Acceptor is player 2 (yellow)
      moves: '[]',
      currentTurn: 1,
      status: 'active',
      mode: 'ranked',
      player1Rating: challenge.challengerRating,
      player2Rating: acceptor.rating,
      spectatable: 1,
      lastMoveAt: now,
      timeControlMs: TIME_CONTROL_MS,
      player1TimeMs: TIME_CONTROL_MS,
      player2TimeMs: TIME_CONTROL_MS,
      turnStartedAt: now,
      createdAt: now,
      updatedAt: now,
    })

    // Update challenge status
    await db.update(challenges)
      .set({
        status: 'accepted',
        gameId,
        targetId: session.userId,
        targetRating: acceptor.rating,
      })
      .where(eq(challenges.id, challengeId))

    return jsonResponse({
      status: 'accepted',
      gameId,
      playerNumber: 2, // Acceptor is always player 2
      opponent: {
        username: challenge.challengerUsername,
        rating: challenge.challengerRating,
      },
    })
  } catch (error) {
    console.error('POST /api/challenges/:id/accept error:', error)
    return errorResponse('Internal server error', 500)
  }
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
