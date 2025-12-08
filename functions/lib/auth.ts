/**
 * Authentication helpers for Cloudflare Workers
 */

interface SessionValidationResult {
  valid: true
  userId: string
  sessionId: string
}

interface SessionValidationError {
  valid: false
  error: string
  status: number
}

type SessionResult = SessionValidationResult | SessionValidationError

/**
 * Validates a session token from the request headers.
 * Returns the user ID if valid, or an error response if not.
 */
export async function validateSession(
  request: Request,
  db: D1Database
): Promise<SessionResult> {
  // Get session token from Authorization header
  const authHeader = request.headers.get('Authorization')
  const sessionToken = authHeader?.replace('Bearer ', '')

  if (!sessionToken) {
    return {
      valid: false,
      error: 'Session token required',
      status: 401,
    }
  }

  // Find session token
  const session = await db
    .prepare('SELECT * FROM session_tokens WHERE id = ?')
    .bind(sessionToken)
    .first<{ id: string; user_id: string; expires_at: number }>()

  if (!session) {
    return {
      valid: false,
      error: 'Invalid session token',
      status: 401,
    }
  }

  // Check if expired
  if (session.expires_at < Date.now()) {
    // Delete expired token
    await db.prepare('DELETE FROM session_tokens WHERE id = ?').bind(sessionToken).run()
    return {
      valid: false,
      error: 'Session expired',
      status: 401,
    }
  }

  return {
    valid: true,
    userId: session.user_id,
    sessionId: session.id,
  }
}

/**
 * Creates an error response with JSON body.
 */
export function errorResponse(error: string, status: number = 400): Response {
  return new Response(JSON.stringify({ error }), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

/**
 * Creates a success response with JSON body.
 */
export function jsonResponse<T>(data: T, status: number = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}
