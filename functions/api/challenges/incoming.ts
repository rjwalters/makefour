/**
 * GET /api/challenges/incoming - Poll for incoming challenges
 */

import { validateSession, errorResponse, jsonResponse } from '../../lib/auth'

interface Env {
  DB: D1Database
}

interface UserRow {
  id: string
  email: string
  username: string | null
}

interface ChallengeRow {
  id: string
  challenger_id: string
  challenger_username: string
  challenger_rating: number
  target_id: string | null
  target_username: string
  status: string
  created_at: number
  expires_at: number
  game_id: string | null
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

    const now = Date.now()

    // Get user info to find challenges by username
    const user = await DB.prepare(`SELECT id, email, username FROM users WHERE id = ?`)
      .bind(session.userId)
      .first<UserRow>()

    if (!user) {
      return errorResponse('User not found', 404)
    }

    const displayName = getDisplayName(user).toLowerCase()

    // Expire old challenges first
    await DB.prepare(`
      UPDATE challenges
      SET status = 'expired'
      WHERE status = 'pending'
      AND expires_at < ?
    `).bind(now).run()

    // Get incoming challenges (where this user is the target)
    const incoming = await DB.prepare(`
      SELECT id, challenger_id, challenger_username, challenger_rating,
             target_id, target_username, status, created_at, expires_at, game_id
      FROM challenges
      WHERE (target_id = ? OR LOWER(target_username) = ?)
      AND status = 'pending'
      AND expires_at > ?
      ORDER BY created_at DESC
      LIMIT 10
    `)
      .bind(session.userId, displayName, now)
      .all<ChallengeRow>()

    // Also check if any of our outgoing challenges were matched
    const matched = await DB.prepare(`
      SELECT id, target_username, target_rating, game_id
      FROM challenges
      WHERE challenger_id = ?
      AND status = 'accepted'
      AND game_id IS NOT NULL
      AND created_at > ?
      ORDER BY created_at DESC
      LIMIT 1
    `)
      .bind(session.userId, now - 60000) // Check last minute
      .first<ChallengeRow>()

    return jsonResponse({
      incoming: incoming.results.map((c) => ({
        id: c.id,
        challengerUsername: c.challenger_username,
        challengerRating: c.challenger_rating,
        createdAt: c.created_at,
        expiresAt: c.expires_at,
      })),
      matchedGame: matched
        ? {
            gameId: matched.game_id,
            opponentUsername: matched.target_username,
            opponentRating: matched.target_rating,
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
