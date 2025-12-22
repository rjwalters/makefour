/**
 * Challenges API
 * POST /api/challenges - Create a new challenge to a specific user
 * GET /api/challenges - Get user's outgoing challenges
 */

import { validateSession, errorResponse, jsonResponse } from '../../lib/auth'
import { z } from 'zod'
import { createDb } from '../../../shared/db/client'
import { users, challenges, activeGames } from '../../../shared/db/schema'
import { eq, and, or, gt, lt, sql, desc } from 'drizzle-orm'
import { DEFAULT_TIME_CONTROL_MS } from '../../lib/types'

interface Env {
  DB: D1Database
}

const CHALLENGE_EXPIRY_MS = 5 * 60 * 1000 // 5 minutes

const createChallengeSchema = z.object({
  targetUsername: z.string().min(1).max(100),
})

// Get display name from user: prefer username, fall back to email prefix
function getDisplayName(user: { username: string | null; email: string }): string {
  return user.username || user.email.split('@')[0]
}

export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    const body = await context.request.json().catch(() => ({}))
    const parseResult = createChallengeSchema.safeParse(body)

    if (!parseResult.success) {
      return errorResponse(parseResult.error.errors[0].message, 400)
    }

    const { targetUsername } = parseResult.data
    const normalizedTarget = targetUsername.toLowerCase().trim()
    const db = createDb(DB)

    // Get challenger info
    const challenger = await db.query.users.findFirst({
      where: eq(users.id, session.userId),
      columns: { id: true, email: true, username: true, rating: true, emailVerified: true, isBot: true }
    })

    if (!challenger) {
      return errorResponse('User not found', 404)
    }

    // Require email verification
    if (challenger.emailVerified !== 1) {
      return errorResponse('Email verification required for challenges', 403)
    }

    const challengerUsername = getDisplayName(challenger)

    // Cannot challenge yourself
    if (challengerUsername.toLowerCase() === normalizedTarget) {
      return errorResponse('Cannot challenge yourself', 400)
    }

    // Check if already in an active game
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

    // Check for existing pending challenge to this user
    const existingChallenge = await db.query.challenges.findFirst({
      where: and(
        eq(challenges.challengerId, session.userId),
        sql`LOWER(${challenges.targetUsername}) = ${normalizedTarget}`,
        eq(challenges.status, 'pending'),
        gt(challenges.expiresAt, Date.now())
      ),
      columns: { id: true }
    })

    if (existingChallenge) {
      return errorResponse('You already have a pending challenge to this user', 409)
    }

    // Find target user by username or email prefix
    const targetUser = await db.query.users.findFirst({
      where: and(
        or(
          sql`LOWER(${users.username}) = ${normalizedTarget}`,
          sql`LOWER(SUBSTR(${users.email}, 1, INSTR(${users.email}, '@') - 1)) = ${normalizedTarget}`
        ),
        eq(users.isBot, 0)
      ),
      columns: { id: true, email: true, username: true, rating: true, isBot: true }
    })

    // Check if target has a pending challenge TO the challenger (mutual challenge = start game)
    if (targetUser) {
      const mutualChallenge = await db.query.challenges.findFirst({
        where: and(
          eq(challenges.challengerId, targetUser.id),
          eq(challenges.targetId, session.userId),
          eq(challenges.status, 'pending'),
          gt(challenges.expiresAt, Date.now())
        ),
        columns: { id: true }
      })

      if (mutualChallenge) {
        // Mutual challenge! Create game immediately
        const gameId = crypto.randomUUID()
        const now = Date.now()

        // Create active game
        await db.insert(activeGames).values({
          id: gameId,
          player1Id: session.userId,
          player2Id: targetUser.id,
          moves: '[]',
          currentTurn: 1,
          status: 'active',
          mode: 'ranked',
          player1Rating: challenger.rating,
          player2Rating: targetUser.rating,
          spectatable: 1,
          lastMoveAt: now,
          timeControlMs: DEFAULT_TIME_CONTROL_MS,
          player1TimeMs: DEFAULT_TIME_CONTROL_MS,
          player2TimeMs: DEFAULT_TIME_CONTROL_MS,
          turnStartedAt: now,
          createdAt: now,
          updatedAt: now,
        })

        // Update the mutual challenge as accepted
        await db.update(challenges)
          .set({ status: 'accepted', gameId })
          .where(eq(challenges.id, mutualChallenge.id))

        return jsonResponse({
          status: 'matched',
          gameId,
          opponent: {
            id: targetUser.id,
            username: getDisplayName(targetUser),
            rating: targetUser.rating,
          },
        })
      }
    }

    // No mutual challenge - create a pending challenge
    const challengeId = crypto.randomUUID()
    const now = Date.now()
    const expiresAt = now + CHALLENGE_EXPIRY_MS

    await db.insert(challenges).values({
      id: challengeId,
      challengerId: session.userId,
      challengerUsername,
      challengerRating: challenger.rating,
      targetId: targetUser?.id || null,
      targetUsername, // Keep original casing for display
      targetRating: targetUser?.rating || null,
      status: 'pending',
      createdAt: now,
      expiresAt,
    })

    return jsonResponse({
      status: 'pending',
      challengeId,
      target: {
        username: targetUsername,
        rating: targetUser?.rating || null,
        exists: !!targetUser,
      },
      expiresAt,
    })
  } catch (error) {
    console.error('POST /api/challenges error:', error)
    return errorResponse('Internal server error', 500)
  }
}

export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    const db = createDb(DB)
    const now = Date.now()

    // Get user's outgoing challenges
    const outgoing = await db
      .select()
      .from(challenges)
      .where(
        and(
          eq(challenges.challengerId, session.userId),
          or(
            eq(challenges.status, 'pending'),
            and(
              eq(challenges.status, 'accepted'),
              gt(challenges.createdAt, now - 60000)
            )
          )
        )
      )
      .orderBy(desc(challenges.createdAt))
      .limit(10)

    return jsonResponse({
      challenges: outgoing.map((c) => ({
        id: c.id,
        targetUsername: c.targetUsername,
        targetRating: c.targetRating,
        targetExists: !!c.targetId,
        status: c.status,
        createdAt: c.createdAt,
        expiresAt: c.expiresAt,
        gameId: c.gameId,
      })),
    })
  } catch (error) {
    console.error('GET /api/challenges error:', error)
    return errorResponse('Internal server error', 500)
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
