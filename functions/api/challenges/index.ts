/**
 * Challenges API
 * POST /api/challenges - Create a new challenge to a specific user
 * GET /api/challenges - Get user's outgoing challenges
 */

import { validateSession, errorResponse, jsonResponse } from '../../lib/auth'
import { z } from 'zod'

interface Env {
  DB: D1Database
}

const CHALLENGE_EXPIRY_MS = 5 * 60 * 1000 // 5 minutes

const createChallengeSchema = z.object({
  targetUsername: z.string().min(1).max(100),
})

interface UserRow {
  id: string
  email: string
  rating: number
  email_verified: number
  is_bot: number
}

interface ChallengeRow {
  id: string
  challenger_id: string
  challenger_username: string
  challenger_rating: number
  target_id: string | null
  target_username: string
  target_rating: number | null
  status: string
  created_at: number
  expires_at: number
  game_id: string | null
}

interface ActiveGameRow {
  id: string
}

// Extract username from email (before @)
function getUsernameFromEmail(email: string): string {
  return email.split('@')[0]
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

    // Get challenger info
    const challenger = await DB.prepare(`
      SELECT id, email, rating, email_verified, is_bot FROM users WHERE id = ?
    `)
      .bind(session.userId)
      .first<UserRow>()

    if (!challenger) {
      return errorResponse('User not found', 404)
    }

    // Require email verification
    if (challenger.email_verified !== 1) {
      return errorResponse('Email verification required for challenges', 403)
    }

    const challengerUsername = getUsernameFromEmail(challenger.email)

    // Cannot challenge yourself
    if (challengerUsername.toLowerCase() === normalizedTarget) {
      return errorResponse('Cannot challenge yourself', 400)
    }

    // Check if already in an active game
    const activeGame = await DB.prepare(`
      SELECT id FROM active_games
      WHERE (player1_id = ? OR player2_id = ?)
      AND status = 'active'
    `)
      .bind(session.userId, session.userId)
      .first<ActiveGameRow>()

    if (activeGame) {
      return errorResponse('You are already in an active game', 409)
    }

    // Check for existing pending challenge to this user
    const existingChallenge = await DB.prepare(`
      SELECT id FROM challenges
      WHERE challenger_id = ?
      AND LOWER(target_username) = ?
      AND status = 'pending'
      AND expires_at > ?
    `)
      .bind(session.userId, normalizedTarget, Date.now())
      .first<ChallengeRow>()

    if (existingChallenge) {
      return errorResponse('You already have a pending challenge to this user', 409)
    }

    // Find target user by username (email prefix)
    const targetUser = await DB.prepare(`
      SELECT id, email, rating, is_bot FROM users
      WHERE LOWER(SUBSTR(email, 1, INSTR(email, '@') - 1)) = ?
      AND is_bot = 0
    `)
      .bind(normalizedTarget)
      .first<UserRow>()

    // Check if target has a pending challenge TO the challenger (mutual challenge = start game)
    if (targetUser) {
      const mutualChallenge = await DB.prepare(`
        SELECT id FROM challenges
        WHERE challenger_id = ?
        AND target_id = ?
        AND status = 'pending'
        AND expires_at > ?
      `)
        .bind(targetUser.id, session.userId, Date.now())
        .first<ChallengeRow>()

      if (mutualChallenge) {
        // Mutual challenge! Create game immediately
        const gameId = crypto.randomUUID()
        const now = Date.now()
        const TIME_CONTROL_MS = 300000 // 5 minutes

        // Create active game
        await DB.prepare(`
          INSERT INTO active_games (
            id, player1_id, player2_id, moves, current_turn, status, mode,
            player1_rating, player2_rating, spectatable, last_move_at,
            time_control_ms, player1_time_ms, player2_time_ms, turn_started_at,
            created_at, updated_at
          )
          VALUES (?, ?, ?, '[]', 1, 'active', 'ranked', ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
        `).bind(
          gameId,
          session.userId,
          targetUser.id,
          challenger.rating,
          targetUser.rating,
          now,
          TIME_CONTROL_MS,
          TIME_CONTROL_MS,
          TIME_CONTROL_MS,
          now,
          now,
          now
        ).run()

        // Update the mutual challenge as accepted
        await DB.prepare(`
          UPDATE challenges SET status = 'accepted', game_id = ? WHERE id = ?
        `).bind(gameId, mutualChallenge.id).run()

        return jsonResponse({
          status: 'matched',
          gameId,
          opponent: {
            id: targetUser.id,
            username: getUsernameFromEmail(targetUser.email),
            rating: targetUser.rating,
          },
        })
      }
    }

    // No mutual challenge - create a pending challenge
    const challengeId = crypto.randomUUID()
    const now = Date.now()
    const expiresAt = now + CHALLENGE_EXPIRY_MS

    await DB.prepare(`
      INSERT INTO challenges (
        id, challenger_id, challenger_username, challenger_rating,
        target_id, target_username, target_rating,
        status, created_at, expires_at
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
    `).bind(
      challengeId,
      session.userId,
      challengerUsername,
      challenger.rating,
      targetUser?.id || null,
      targetUsername, // Keep original casing for display
      targetUser?.rating || null,
      now,
      expiresAt
    ).run()

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

    const now = Date.now()

    // Get user's outgoing challenges
    const outgoing = await DB.prepare(`
      SELECT id, challenger_id, challenger_username, challenger_rating,
             target_id, target_username, target_rating, status, created_at, expires_at, game_id
      FROM challenges
      WHERE challenger_id = ?
      AND (status = 'pending' OR (status = 'accepted' AND created_at > ?))
      ORDER BY created_at DESC
      LIMIT 10
    `)
      .bind(session.userId, now - 60000) // Include recently accepted
      .all<ChallengeRow>()

    return jsonResponse({
      challenges: outgoing.results.map((c) => ({
        id: c.id,
        targetUsername: c.target_username,
        targetRating: c.target_rating,
        targetExists: !!c.target_id,
        status: c.status,
        createdAt: c.created_at,
        expiresAt: c.expires_at,
        gameId: c.game_id,
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
