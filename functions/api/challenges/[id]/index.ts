/**
 * DELETE /api/challenges/:id - Cancel or decline a challenge
 */

import { validateSession, errorResponse, jsonResponse } from '../../../lib/auth'
import { createDb } from '../../../../shared/db/client'
import { users, challenges } from '../../../../shared/db/schema'
import { eq, sql } from 'drizzle-orm'

interface Env {
  DB: D1Database
}

// Get display name from user: prefer username, fall back to email prefix
function getDisplayName(user: { username: string | null; email: string }): string {
  return user.username || user.email.split('@')[0]
}

export async function onRequestDelete(context: EventContext<Env, any, any>) {
  const { DB } = context.env
  const challengeId = context.params.id as string

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    const db = createDb(DB)

    // Get the challenge
    const challenge = await db.query.challenges.findFirst({
      where: eq(challenges.id, challengeId),
      columns: { id: true, challengerId: true, targetId: true, targetUsername: true, status: true }
    })

    if (!challenge) {
      return errorResponse('Challenge not found', 404)
    }

    if (challenge.status !== 'pending') {
      return errorResponse(`Challenge is already ${challenge.status}`, 400)
    }

    // Get user info
    const user = await db.query.users.findFirst({
      where: eq(users.id, session.userId),
      columns: { id: true, email: true, username: true }
    })

    if (!user) {
      return errorResponse('User not found', 404)
    }

    const displayName = getDisplayName(user).toLowerCase()

    // Check if user is challenger or target
    const isChallenger = challenge.challengerId === session.userId
    const isTarget =
      challenge.targetId === session.userId ||
      challenge.targetUsername.toLowerCase() === displayName

    if (!isChallenger && !isTarget) {
      return errorResponse('You are not part of this challenge', 403)
    }

    // Update status based on who cancelled
    const newStatus = isChallenger ? 'cancelled' : 'declined'

    await db.update(challenges)
      .set({ status: newStatus })
      .where(eq(challenges.id, challengeId))

    return jsonResponse({
      status: newStatus,
      message: isChallenger ? 'Challenge cancelled' : 'Challenge declined',
    })
  } catch (error) {
    console.error('DELETE /api/challenges/:id error:', error)
    return errorResponse('Internal server error', 500)
  }
}

export async function onRequestOptions() {
  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  })
}
