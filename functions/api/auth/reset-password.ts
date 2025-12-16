import { z } from 'zod'
import * as bcrypt from 'bcryptjs'
import { validateResetPasswordRequest, formatZodError } from '../../lib/schemas'
import { errorResponse, jsonResponse } from '../../lib/auth'
import { generateDEK, encryptDEK } from '../../lib/crypto'
import { createDb } from '../../../shared/db/client'
import { users, passwordResetTokens, sessionTokens } from '../../../shared/db/schema'
import { eq, and } from 'drizzle-orm'

interface Env {
  DB: D1Database
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
  const db = createDb(DB)

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
    const resetToken = await db.query.passwordResetTokens.findFirst({
      where: eq(passwordResetTokens.id, token),
    })

    if (!resetToken) {
      return errorResponse('Invalid or expired reset link', 400)
    }

    // Check if token has been used
    if (resetToken.used === 1) {
      return errorResponse('This reset link has already been used', 400)
    }

    // Check if token has expired
    if (resetToken.expiresAt < Date.now()) {
      // Clean up expired token
      await db.delete(passwordResetTokens).where(eq(passwordResetTokens.id, token))
      return errorResponse('This reset link has expired', 400)
    }

    // Get the user
    const user = await db.query.users.findFirst({
      where: eq(users.id, resetToken.userId),
      columns: {
        id: true,
        email: true,
        passwordHash: true,
        oauthProvider: true,
      },
    })

    if (!user) {
      // User was deleted after requesting reset
      await db.delete(passwordResetTokens).where(eq(passwordResetTokens.id, token))
      return errorResponse('Account not found', 404)
    }

    // Prevent OAuth-only users from setting a password this way
    if (user.oauthProvider && !user.passwordHash) {
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
    await db.update(users)
      .set({
        passwordHash: newPasswordHash,
        encryptedDek: encryptedDEK,
        updatedAt: now,
      })
      .where(eq(users.id, user.id))

    // Mark token as used
    await db.update(passwordResetTokens)
      .set({ used: 1 })
      .where(eq(passwordResetTokens.id, token))

    // Invalidate ALL existing sessions for security
    // This forces the user to log in with their new password
    await db.delete(sessionTokens)
      .where(eq(sessionTokens.userId, user.id))

    // Clean up any other unused reset tokens for this user
    await db.delete(passwordResetTokens)
      .where(and(
        eq(passwordResetTokens.userId, user.id),
        eq(passwordResetTokens.used, 0)
      ))

    return jsonResponse({
      message: 'Password reset successfully. Please log in with your new password.',
    })
  } catch (error) {
    console.error('Reset password error:', error)
    return errorResponse('Internal server error', 500)
  }
}
