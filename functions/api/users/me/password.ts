import { validateSession, errorResponse, jsonResponse } from '../../../lib/auth'
import { validateChangePasswordRequest } from '../../../../src/lib/schemas/auth'
import bcrypt from 'bcryptjs'

interface Env {
  DB: D1Database
}

interface UserRow {
  id: string
  password_hash: string | null
  oauth_provider: string | null
}

/**
 * PUT /api/users/me/password - Change user password
 *
 * Request body:
 * - old_password: Current password
 * - new_password: New password (min 8 characters)
 *
 * Note: OAuth users cannot change password through this endpoint
 */
export async function onRequestPut(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    // Validate session
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    const userId = session.userId

    // Parse and validate request body
    let body: unknown
    try {
      body = await context.request.json()
    } catch {
      return errorResponse('Invalid JSON body', 400)
    }

    let validatedData: { old_password: string; new_password: string }
    try {
      validatedData = validateChangePasswordRequest(body)
    } catch (e) {
      return errorResponse(e instanceof Error ? e.message : 'Invalid request data', 400)
    }

    const { old_password, new_password } = validatedData

    // Get user with password info
    const user = await DB.prepare(
      'SELECT id, password_hash, oauth_provider FROM users WHERE id = ?'
    )
      .bind(userId)
      .first<UserRow>()

    if (!user) {
      return errorResponse('User not found', 404)
    }

    // Check if user uses OAuth (no password)
    if (user.oauth_provider && !user.password_hash) {
      return errorResponse(
        'Cannot change password for OAuth accounts. Please manage your password through your OAuth provider.',
        400
      )
    }

    if (!user.password_hash) {
      return errorResponse('No password set for this account', 400)
    }

    // Verify old password
    const isValidPassword = await bcrypt.compare(old_password, user.password_hash)
    if (!isValidPassword) {
      return errorResponse('Current password is incorrect', 401)
    }

    // Hash new password
    const newPasswordHash = await bcrypt.hash(new_password, 10)

    // Update password
    await DB.prepare(
      'UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?'
    )
      .bind(newPasswordHash, Date.now(), userId)
      .run()

    // Optionally: Invalidate all other sessions for security
    // Keep current session active
    await DB.prepare(
      'DELETE FROM session_tokens WHERE user_id = ? AND id != ?'
    )
      .bind(userId, session.sessionId)
      .run()

    return jsonResponse({ message: 'Password changed successfully' })
  } catch (error) {
    console.error('Change password error:', error)
    return errorResponse('Internal server error', 500)
  }
}
