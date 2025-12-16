/**
 * DELETE /api/challenges/:id - Cancel or decline a challenge
 */

import { validateSession, errorResponse, jsonResponse } from '../../../lib/auth'

interface Env {
  DB: D1Database
}

interface UserRow {
  id: string
  email: string
}

interface ChallengeRow {
  id: string
  challenger_id: string
  target_id: string | null
  target_username: string
  status: string
}

// Extract username from email (before @)
function getUsernameFromEmail(email: string): string {
  return email.split('@')[0]
}

export async function onRequestDelete(context: EventContext<Env, any, any>) {
  const { DB } = context.env
  const challengeId = context.params.id as string

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Get the challenge
    const challenge = await DB.prepare(`
      SELECT id, challenger_id, target_id, target_username, status
      FROM challenges WHERE id = ?
    `)
      .bind(challengeId)
      .first<ChallengeRow>()

    if (!challenge) {
      return errorResponse('Challenge not found', 404)
    }

    if (challenge.status !== 'pending') {
      return errorResponse(`Challenge is already ${challenge.status}`, 400)
    }

    // Get user info
    const user = await DB.prepare(`SELECT id, email FROM users WHERE id = ?`)
      .bind(session.userId)
      .first<UserRow>()

    if (!user) {
      return errorResponse('User not found', 404)
    }

    const username = getUsernameFromEmail(user.email).toLowerCase()

    // Check if user is challenger or target
    const isChallenger = challenge.challenger_id === session.userId
    const isTarget =
      challenge.target_id === session.userId ||
      challenge.target_username.toLowerCase() === username

    if (!isChallenger && !isTarget) {
      return errorResponse('You are not part of this challenge', 403)
    }

    // Update status based on who cancelled
    const newStatus = isChallenger ? 'cancelled' : 'declined'

    await DB.prepare(`UPDATE challenges SET status = ? WHERE id = ?`)
      .bind(newStatus, challengeId)
      .run()

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
