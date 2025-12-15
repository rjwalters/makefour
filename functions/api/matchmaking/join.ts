/**
 * POST /api/matchmaking/join - Join the matchmaking queue
 */

import { validateSession, errorResponse, jsonResponse } from '../../lib/auth'
import { z } from 'zod'

interface Env {
  DB: D1Database
}

const joinQueueSchema = z.object({
  mode: z.enum(['ranked', 'casual']).default('ranked'),
})

interface UserRow {
  id: string
  rating: number
}

interface QueueEntry {
  id: string
  user_id: string
  rating: number
  mode: string
  joined_at: number
}

interface ActiveGameRow {
  id: string
}

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

    const { mode } = parseResult.data

    // Check if user is already in an active game
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

    // Check if user is already in queue
    const existingEntry = await DB.prepare(`
      SELECT id FROM matchmaking_queue WHERE user_id = ?
    `)
      .bind(session.userId)
      .first<QueueEntry>()

    if (existingEntry) {
      return errorResponse('Already in matchmaking queue', 409)
    }

    // Get user's current rating
    const user = await DB.prepare(`
      SELECT id, rating FROM users WHERE id = ?
    `)
      .bind(session.userId)
      .first<UserRow>()

    if (!user) {
      return errorResponse('User not found', 404)
    }

    // Add to queue
    const queueId = crypto.randomUUID()
    const now = Date.now()

    await DB.prepare(`
      INSERT INTO matchmaking_queue (id, user_id, rating, mode, initial_tolerance, joined_at)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind(queueId, session.userId, user.rating, mode, 100, now).run()

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
