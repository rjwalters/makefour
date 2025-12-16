/**
 * POST /api/challenges/:id/accept - Accept an incoming challenge
 */

import { validateSession, errorResponse, jsonResponse } from '../../../lib/auth'

interface Env {
  DB: D1Database
}

interface UserRow {
  id: string
  email: string
  username: string | null
  rating: number
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
}

interface ActiveGameRow {
  id: string
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

    const now = Date.now()

    // Get the challenge
    const challenge = await DB.prepare(`
      SELECT id, challenger_id, challenger_username, challenger_rating,
             target_id, target_username, target_rating, status, created_at, expires_at
      FROM challenges WHERE id = ?
    `)
      .bind(challengeId)
      .first<ChallengeRow>()

    if (!challenge) {
      return errorResponse('Challenge not found', 404)
    }

    if (challenge.status !== 'pending') {
      return errorResponse(`Challenge is ${challenge.status}`, 400)
    }

    if (challenge.expires_at < now) {
      await DB.prepare(`UPDATE challenges SET status = 'expired' WHERE id = ?`)
        .bind(challengeId)
        .run()
      return errorResponse('Challenge has expired', 400)
    }

    // Get acceptor info
    const acceptor = await DB.prepare(`
      SELECT id, email, username, rating FROM users WHERE id = ?
    `)
      .bind(session.userId)
      .first<UserRow>()

    if (!acceptor) {
      return errorResponse('User not found', 404)
    }

    const acceptorDisplayName = getDisplayName(acceptor).toLowerCase()

    // Verify the acceptor is the target
    if (
      challenge.target_id !== session.userId &&
      challenge.target_username.toLowerCase() !== acceptorDisplayName
    ) {
      return errorResponse('You are not the target of this challenge', 403)
    }

    // Check if acceptor is already in a game
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

    // Check if challenger is already in a game
    const challengerActiveGame = await DB.prepare(`
      SELECT id FROM active_games
      WHERE (player1_id = ? OR player2_id = ?)
      AND status = 'active'
    `)
      .bind(challenge.challenger_id, challenge.challenger_id)
      .first<ActiveGameRow>()

    if (challengerActiveGame) {
      // Cancel the challenge - challenger is busy
      await DB.prepare(`UPDATE challenges SET status = 'cancelled' WHERE id = ?`)
        .bind(challengeId)
        .run()
      return errorResponse('Challenger is already in a game', 409)
    }

    // Create the game
    const gameId = crypto.randomUUID()
    const TIME_CONTROL_MS = 300000 // 5 minutes

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
      challenge.challenger_id, // Challenger is player 1 (red)
      session.userId, // Acceptor is player 2 (yellow)
      challenge.challenger_rating,
      acceptor.rating,
      now,
      TIME_CONTROL_MS,
      TIME_CONTROL_MS,
      TIME_CONTROL_MS,
      now,
      now,
      now
    ).run()

    // Update challenge status
    await DB.prepare(`
      UPDATE challenges SET status = 'accepted', game_id = ?, target_id = ?, target_rating = ?
      WHERE id = ?
    `).bind(gameId, session.userId, acceptor.rating, challengeId).run()

    return jsonResponse({
      status: 'accepted',
      gameId,
      playerNumber: 2, // Acceptor is always player 2
      opponent: {
        username: challenge.challenger_username,
        rating: challenge.challenger_rating,
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
