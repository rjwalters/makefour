import { z } from 'zod'
import { validateForgotPasswordRequest, formatZodError } from '../../lib/schemas'
import { errorResponse, jsonResponse } from '../../lib/auth'
import { sendEmail, generatePasswordResetEmail } from '../../lib/email'
import { createDb } from '../../../shared/db/client'
import { users, passwordResetTokens } from '../../../shared/db/schema'
import { eq, and } from 'drizzle-orm'
import { PASSWORD_RESET_TOKEN_EXPIRY_MS } from '../../lib/types'

interface Env {
  DB: D1Database
  EMAIL_PROVIDER?: string
  EMAIL_API_KEY?: string
  EMAIL_FROM?: string
  EMAIL_DOMAIN?: string
  APP_URL?: string
}

/**
 * POST /api/auth/forgot-password
 *
 * Request a password reset email.
 * For security, always returns success regardless of whether email exists.
 * This prevents email enumeration attacks.
 *
 * Rate limited to 3 requests per hour per email/IP (handled by middleware)
 */
export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB } = context.env
  const db = createDb(DB)
  const appUrl = context.env.APP_URL || 'http://localhost:5173'

  try {
    // Parse and validate request body
    let body: unknown
    try {
      body = await context.request.json()
    } catch {
      return errorResponse('Invalid JSON body', 400)
    }

    let validatedData: { email: string }
    try {
      validatedData = validateForgotPasswordRequest(body)
    } catch (error) {
      if (error instanceof z.ZodError) {
        return new Response(JSON.stringify(formatZodError(error)), {
          status: 400,
          headers: { 'Content-Type': 'application/json' },
        })
      }
      return errorResponse('Invalid request data', 400)
    }

    const { email } = validatedData
    const normalizedEmail = email.toLowerCase().trim()

    // Always return the same response to prevent email enumeration
    const successResponse = jsonResponse({
      message: 'If an account exists with this email, a password reset link has been sent.',
    })

    // Find user by email
    const user = await db.query.users.findFirst({
      where: eq(users.email, normalizedEmail),
      columns: {
        id: true,
        passwordHash: true,
        oauthProvider: true,
      },
    })

    // If user doesn't exist, return success anyway (security)
    if (!user) {
      return successResponse
    }

    // If user uses OAuth only (no password), they can't reset password
    if (user.oauthProvider && !user.passwordHash) {
      // Still return success to prevent enumeration
      // User will need to use their OAuth provider to manage their account
      console.log(`Password reset requested for OAuth-only user: ${normalizedEmail}`)
      return successResponse
    }

    // Delete any existing unused reset tokens for this user
    await db.delete(passwordResetTokens)
      .where(and(
        eq(passwordResetTokens.userId, user.id),
        eq(passwordResetTokens.used, 0)
      ))

    // Generate new token
    const tokenId = crypto.randomUUID()
    const now = Date.now()
    const expiresAt = now + PASSWORD_RESET_TOKEN_EXPIRY_MS

    // Insert reset token
    await db.insert(passwordResetTokens).values({
      id: tokenId,
      userId: user.id,
      expiresAt,
      used: 0,
      createdAt: now,
    })

    // Generate reset URL
    const resetUrl = `${appUrl}/reset-password?token=${tokenId}`

    // Generate email content
    const emailContent = generatePasswordResetEmail(resetUrl, 60)

    // Send email
    const emailSent = await sendEmail(
      {
        to: normalizedEmail,
        subject: emailContent.subject,
        html: emailContent.html,
        text: emailContent.text,
      },
      context.env
    )

    if (!emailSent) {
      console.error(`Failed to send password reset email to ${normalizedEmail}`)
      // Still return success to user - we don't want to reveal email delivery issues
    }

    return successResponse
  } catch (error) {
    console.error('Forgot password error:', error)
    return errorResponse('Internal server error', 500)
  }
}
