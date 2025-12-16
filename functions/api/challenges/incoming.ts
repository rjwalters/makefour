/**
 * GET /api/challenges/incoming - Poll for incoming challenges
 */

import { validateSession, errorResponse, jsonResponse } from '../../lib/auth'
import { createDb } from '../../../shared/db/client'
import { users, challenges } from '../../../shared/db/schema'
import { eq, and, or, gt, lt, sql, desc } from 'drizzle-orm'

interface Env {
  DB: D1Database
}

// Get display name from user: prefer username, fall back to email prefix
function getDisplayName(user: { username: string | null; email: string }): string {
  return user.username || user.email.split('@')[0]
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

    // Get user info to find challenges by username
    const user = await db.query.users.findFirst({
      where: eq(users.id, session.userId),
      columns: { id: true, email: true, username: true }
    })

    if (!user) {
      return errorResponse('User not found', 404)
    }

    const displayName = getDisplayName(user).toLowerCase()

    // Expire old challenges first
    await db.update(challenges)
      .set({ status: 'expired' })
      .where(
        and(
          eq(challenges.status, 'pending'),
          lt(challenges.expiresAt, now)
        )
      )

    // Get incoming challenges (where this user is the target)
    const incoming = await db
      .select()
      .from(challenges)
      .where(
        and(
          or(
            eq(challenges.targetId, session.userId),
            sql`LOWER(${challenges.targetUsername}) = ${displayName}`
          ),
          eq(challenges.status, 'pending'),
          gt(challenges.expiresAt, now)
        )
      )
      .orderBy(desc(challenges.createdAt))
      .limit(10)

    // Also check if any of our outgoing challenges were matched
    const matched = await db.query.challenges.findFirst({
      where: and(
        eq(challenges.challengerId, session.userId),
        eq(challenges.status, 'accepted'),
        sql`${challenges.gameId} IS NOT NULL`,
        gt(challenges.createdAt, now - 60000)
      ),
      columns: { id: true, targetUsername: true, targetRating: true, gameId: true },
      orderBy: desc(challenges.createdAt)
    })

    return jsonResponse({
      incoming: incoming.map((c) => ({
        id: c.id,
        challengerUsername: c.challengerUsername,
        challengerRating: c.challengerRating,
        createdAt: c.createdAt,
        expiresAt: c.expiresAt,
      })),
      matchedGame: matched
        ? {
            gameId: matched.gameId,
            opponentUsername: matched.targetUsername,
            opponentRating: matched.targetRating,
          }
        : null,
    })
  } catch (error) {
    console.error('GET /api/challenges/incoming error:', error)
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
