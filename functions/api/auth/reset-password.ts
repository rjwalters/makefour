import { z } from 'zod'
import * as bcrypt from 'bcryptjs'
import { validateResetPasswordRequest, formatZodError } from '../../lib/schemas'
import { errorResponse, jsonResponse } from '../../lib/auth'
import { generateDEK, encryptDEK } from '../../lib/crypto'

interface Env {
  DB: D1Database
}

interface TokenRow {
  id: string
  user_id: string
  expires_at: number
  used: number
  created_at: number
}

interface UserRow {
  id: string
  email: string
  password_hash: string | null
  oauth_provider: string | null
}

/**
 * POST /api/auth/reset-password
 *
 * Reset password using a valid reset token.
 * - Token must exist and not be expired
 * - Token must not have been used before
 * - All existing sessions are invalidated
 * - A new DEK is generated (old encrypted data will be inaccessible)
 */
export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    // Parse and validate request body
    let body: unknown
    try {
      body = await context.request.json()
    } catch {
      return errorResponse('Invalid JSON body', 400)
    }

    let validatedData: { token: string; new_password: string }
    try {
      validatedData = validateResetPasswordRequest(body)
    } catch (error) {
      if (error instanceof z.ZodError) {
        return new Response(JSON.stringify(formatZodError(error)), {
          status: 400,
          headers: { 'Content-Type': 'application/json' },
        })
      }
      return errorResponse('Invalid request data', 400)
    }

    const { token, new_password } = validatedData

    // Find the reset token
    const resetToken = await DB.prepare(
      'SELECT * FROM password_reset_tokens WHERE id = ?'
    ).bind(token).first<TokenRow>()

    if (!resetToken) {
      return errorResponse('Invalid or expired reset link', 400)
    }

    // Check if token has been used
    if (resetToken.used === 1) {
      return errorResponse('This reset link has already been used', 400)
    }

    // Check if token has expired
    if (resetToken.expires_at < Date.now()) {
      // Clean up expired token
      await DB.prepare('DELETE FROM password_reset_tokens WHERE id = ?').bind(token).run()
      return errorResponse('This reset link has expired', 400)
    }

    // Get the user
    const user = await DB.prepare(
      'SELECT id, email, password_hash, oauth_provider FROM users WHERE id = ?'
    ).bind(resetToken.user_id).first<UserRow>()

    if (!user) {
      // User was deleted after requesting reset
      await DB.prepare('DELETE FROM password_reset_tokens WHERE id = ?').bind(token).run()
      return errorResponse('Account not found', 404)
    }

    // Prevent OAuth-only users from setting a password this way
    if (user.oauth_provider && !user.password_hash) {
      return errorResponse(
        'Cannot set password for OAuth accounts. Please manage your password through your OAuth provider.',
        400
      )
    }

    // Hash the new password
    const newPasswordHash = await bcrypt.hash(new_password, 10)
    const now = Date.now()

    // Generate new DEK and encrypt with new password
    // Note: This means any data encrypted with the old DEK will be inaccessible
    // This is a standard security trade-off for password reset flows
    const newDEK = await generateDEK()
    const encryptedDEK = await encryptDEK(newDEK, new_password)

    // Update user's password and DEK in a transaction
    // D1 doesn't support true transactions, so we'll do this in sequence
    await DB.prepare(
      'UPDATE users SET password_hash = ?, encrypted_dek = ?, updated_at = ? WHERE id = ?'
    ).bind(newPasswordHash, encryptedDEK, now, user.id).run()

    // Mark token as used
    await DB.prepare(
      'UPDATE password_reset_tokens SET used = 1 WHERE id = ?'
    ).bind(token).run()

    // Invalidate ALL existing sessions for security
    // This forces the user to log in with their new password
    await DB.prepare(
      'DELETE FROM session_tokens WHERE user_id = ?'
    ).bind(user.id).run()

    // Clean up any other unused reset tokens for this user
    await DB.prepare(
      'DELETE FROM password_reset_tokens WHERE user_id = ? AND used = 0'
    ).bind(user.id).run()

    return jsonResponse({
      message: 'Password reset successfully. Please log in with your new password.',
    })
  } catch (error) {
    console.error('Reset password error:', error)
    return errorResponse('Internal server error', 500)
  }
}
