/**
 * POST /api/matchmaking/join - Join the matchmaking queue
 */

import { validateSession, errorResponse, jsonResponse } from '../../lib/auth'
import { z } from 'zod'
import { createDb } from '../../../shared/db/client'
import { users, matchmakingQueue, activeGames } from '../../../shared/db/schema'
import { eq, and, or } from 'drizzle-orm'

interface Env {
  DB: D1Database
}

const joinQueueSchema = z.object({
  mode: z.enum(['ranked', 'casual']).default('ranked'),
  spectatable: z.boolean().default(true),
})

export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Parse request body
    const body = await context.request.json().catch(() => ({}))
    const parseResult = joinQueueSchema.safeParse(body)

    if (!parseResult.success) {
      return errorResponse(parseResult.error.errors[0].message, 400)
    }

    const { mode, spectatable } = parseResult.data
    const db = createDb(DB)

    // Check if user is already in an active game
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

    // Check if user is already in queue
    const existingEntry = await db.query.matchmakingQueue.findFirst({
      where: eq(matchmakingQueue.userId, session.userId),
      columns: { id: true }
    })

    if (existingEntry) {
      return errorResponse('Already in matchmaking queue', 409)
    }

    // Get user's current rating and verification status
    const user = await db.query.users.findFirst({
      where: eq(users.id, session.userId),
      columns: { id: true, rating: true, emailVerified: true }
    })

    if (!user) {
      return errorResponse('User not found', 404)
    }

    // Require email verification for online matchmaking
    if (user.emailVerified !== 1) {
      return errorResponse(
        'Email verification required for online matchmaking. Please verify your email first.',
        403
      )
    }

    // Add to queue
    const queueId = crypto.randomUUID()
    const now = Date.now()

    await db.insert(matchmakingQueue).values({
      id: queueId,
      userId: session.userId,
      rating: user.rating,
      mode,
      initialTolerance: 100,
      spectatable: spectatable ? 1 : 0,
      joinedAt: now,
    })

    return jsonResponse({
      status: 'queued',
      queueId,
      mode,
      rating: user.rating,
      joinedAt: now,
    })
  } catch (error) {
    console.error('POST /api/matchmaking/join error:', error)
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
