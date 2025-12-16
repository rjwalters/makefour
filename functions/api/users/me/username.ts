import { and, eq, ne, sql } from 'drizzle-orm'
import { createDb } from '../../../../shared/db/client'
import { users } from '../../../../shared/db/schema'
import { errorResponse, jsonResponse, validateSession } from '../../../lib/auth'

interface Env {
  DB: D1Database
}

// 30 days in milliseconds
const USERNAME_COOLDOWN_MS = 30 * 24 * 60 * 60 * 1000

// Username validation: 3-20 chars, alphanumeric + underscores, must start with letter
const USERNAME_REGEX = /^[a-zA-Z][a-zA-Z0-9_]{2,19}$/

function validateUsername(username: string): { valid: boolean; error?: string } {
  if (typeof username !== 'string') {
    return { valid: false, error: 'Username must be a string' }
  }
  if (username.length < 3) {
    return { valid: false, error: 'Username must be at least 3 characters' }
  }
  if (username.length > 20) {
    return { valid: false, error: 'Username must be at most 20 characters' }
  }
  if (!USERNAME_REGEX.test(username)) {
    return {
      valid: false,
      error: 'Username must start with a letter and contain only letters, numbers, and underscores',
    }
  }
  return { valid: true }
}

/**
 * GET /api/users/me/username - Get current username and eligibility to change
 */
export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    const db = createDb(DB)
    const user = await db.query.users.findFirst({
      where: eq(users.id, session.userId),
      columns: {
        username: true,
        usernameChangedAt: true,
        email: true,
      },
    })

    if (!user) {
      return errorResponse('User not found', 404)
    }

    const now = Date.now()
    const canChange = !user.usernameChangedAt || now - user.usernameChangedAt >= USERNAME_COOLDOWN_MS
    const nextChangeAt = user.usernameChangedAt ? user.usernameChangedAt + USERNAME_COOLDOWN_MS : null

    return jsonResponse({
      username: user.username,
      displayName: user.username || user.email.split('@')[0],
      canChange,
      nextChangeAt: canChange ? null : nextChangeAt,
      cooldownDays: 30,
    })
  } catch (error) {
    console.error('Get username error:', error)
    return errorResponse('Internal server error', 500)
  }
}

/**
 * PUT /api/users/me/username - Set or update username
 *
 * Request body:
 * - username: New username (3-20 chars, alphanumeric + underscores, starts with letter)
 */
export async function onRequestPut(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Parse request body
    let body: { username?: string }
    try {
      body = await context.request.json()
    } catch {
      return errorResponse('Invalid JSON body', 400)
    }

    const { username } = body

    // Validate username format
    if (!username) {
      return errorResponse('Username is required', 400)
    }

    const validation = validateUsername(username)
    if (!validation.valid) {
      return errorResponse(validation.error!, 400)
    }

    const now = Date.now()
    const db = createDb(DB)

    // Get current user data
    const user = await db.query.users.findFirst({
      where: eq(users.id, session.userId),
      columns: {
        username: true,
        usernameChangedAt: true,
      },
    })

    if (!user) {
      return errorResponse('User not found', 404)
    }

    // Check cooldown (only if they've changed it before)
    if (user.usernameChangedAt) {
      const timeSinceChange = now - user.usernameChangedAt
      if (timeSinceChange < USERNAME_COOLDOWN_MS) {
        const daysRemaining = Math.ceil((USERNAME_COOLDOWN_MS - timeSinceChange) / (24 * 60 * 60 * 1000))
        return errorResponse(`You can change your username again in ${daysRemaining} day(s)`, 429)
      }
    }

    // Check if username is already taken (case-insensitive)
    const existing = await db.query.users.findFirst({
      where: and(sql`LOWER(${users.username}) = LOWER(${username})`, ne(users.id, session.userId)),
      columns: {
        id: true,
      },
    })

    if (existing) {
      return errorResponse('This username is already taken', 409)
    }

    // Update username
    await db
      .update(users)
      .set({
        username: username,
        usernameChangedAt: now,
        updatedAt: now,
      })
      .where(eq(users.id, session.userId))

    return jsonResponse({
      success: true,
      username,
      nextChangeAt: now + USERNAME_COOLDOWN_MS,
    })
  } catch (error) {
    console.error('Update username error:', error)
    return errorResponse('Internal server error', 500)
  }
}

export async function onRequestOptions() {
  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, PUT, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  })
}
