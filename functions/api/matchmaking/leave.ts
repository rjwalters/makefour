/**
 * POST /api/matchmaking/leave - Leave the matchmaking queue
 */

import { validateSession, errorResponse, jsonResponse } from '../../lib/auth'
import { createDb } from '../../../shared/db/client'
import { matchmakingQueue } from '../../../shared/db/schema'
import { eq } from 'drizzle-orm'

interface Env {
  DB: D1Database
}

export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    const db = createDb(DB)

    // Remove user from queue
    const result = await db.delete(matchmakingQueue)
      .where(eq(matchmakingQueue.userId, session.userId))

    if (result.rowsAffected === 0) {
      return errorResponse('Not in matchmaking queue', 404)
    }

    return jsonResponse({
      status: 'left',
      message: 'Successfully left the matchmaking queue',
    })
  } catch (error) {
    console.error('POST /api/matchmaking/leave error:', error)
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
