import { validateSession, errorResponse, jsonResponse } from '../../../lib/auth'

interface Env {
  DB: D1Database
}

interface UserRow {
  username: string | null
  username_changed_at: number | null
  email: string
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

    const user = await DB.prepare(
      'SELECT username, username_changed_at, email FROM users WHERE id = ?'
    )
      .bind(session.userId)
      .first<UserRow>()

    if (!user) {
      return errorResponse('User not found', 404)
    }

    const now = Date.now()
    const canChange =
      !user.username_changed_at || now - user.username_changed_at >= USERNAME_COOLDOWN_MS
    const nextChangeAt = user.username_changed_at
      ? user.username_changed_at + USERNAME_COOLDOWN_MS
      : null

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

    // Get current user data
    const user = await DB.prepare(
      'SELECT username, username_changed_at FROM users WHERE id = ?'
    )
      .bind(session.userId)
      .first<{ username: string | null; username_changed_at: number | null }>()

    if (!user) {
      return errorResponse('User not found', 404)
    }

    // Check cooldown (only if they've changed it before)
    if (user.username_changed_at) {
      const timeSinceChange = now - user.username_changed_at
      if (timeSinceChange < USERNAME_COOLDOWN_MS) {
        const daysRemaining = Math.ceil(
          (USERNAME_COOLDOWN_MS - timeSinceChange) / (24 * 60 * 60 * 1000)
        )
        return errorResponse(`You can change your username again in ${daysRemaining} day(s)`, 429)
      }
    }

    // Check if username is already taken (case-insensitive)
    const existing = await DB.prepare(
      'SELECT id FROM users WHERE LOWER(username) = LOWER(?) AND id != ?'
    )
      .bind(username, session.userId)
      .first()

    if (existing) {
      return errorResponse('This username is already taken', 409)
    }

    // Update username
    await DB.prepare(
      'UPDATE users SET username = ?, username_changed_at = ?, updated_at = ? WHERE id = ?'
    )
      .bind(username, now, now, session.userId)
      .run()

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
