import bcrypt from 'bcryptjs'
import { and, eq, ne } from 'drizzle-orm'
import { createDb } from '../../../../shared/db/client'
import { sessionTokens, users } from '../../../../shared/db/schema'
import { validateChangePasswordRequest } from '../../../../src/lib/schemas/auth'
import { errorResponse, jsonResponse, validateSession } from '../../../lib/auth'

interface Env {
  DB: D1Database
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
    const db = createDb(DB)

    // Get user with password info
    const user = await db.query.users.findFirst({
      where: eq(users.id, userId),
      columns: {
        id: true,
        passwordHash: true,
        oauthProvider: true,
      },
    })

    if (!user) {
      return errorResponse('User not found', 404)
    }

    // Check if user uses OAuth (no password)
    if (user.oauthProvider && !user.passwordHash) {
      return errorResponse(
        'Cannot change password for OAuth accounts. Please manage your password through your OAuth provider.',
        400
      )
    }

    if (!user.passwordHash) {
      return errorResponse('No password set for this account', 400)
    }

    // Verify old password
    const isValidPassword = await bcrypt.compare(old_password, user.passwordHash)
    if (!isValidPassword) {
      return errorResponse('Current password is incorrect', 401)
    }

    // Hash new password
    const newPasswordHash = await bcrypt.hash(new_password, 10)

    // Update password
    await db
      .update(users)
      .set({
        passwordHash: newPasswordHash,
        updatedAt: Date.now(),
      })
      .where(eq(users.id, userId))

    // Optionally: Invalidate all other sessions for security
    // Keep current session active
    await db.delete(sessionTokens).where(and(eq(sessionTokens.userId, userId), ne(sessionTokens.id, session.sessionId)))

    return jsonResponse({ message: 'Password changed successfully' })
  } catch (error) {
    console.error('Change password error:', error)
    return errorResponse('Internal server error', 500)
  }
}
