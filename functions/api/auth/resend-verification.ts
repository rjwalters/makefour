/**
 * POST /api/auth/resend-verification - Resend verification email
 *
 * Requires authentication. Rate limited to prevent abuse.
 */

import { validateSession, errorResponse, jsonResponse } from '../../lib/auth'
import {
  createVerificationToken,
  sendVerificationEmail,
  deleteUnusedTokens,
  isEmailVerified,
} from '../../lib/email'

interface Env {
  DB: D1Database
  RESEND_API_KEY?: string
  FROM_EMAIL?: string
  BASE_URL?: string
  RATE_LIMIT?: KVNamespace
}

// Rate limit: 3 resend requests per hour per user
const RATE_LIMIT_WINDOW_MS = 60 * 60 * 1000 // 1 hour
const RATE_LIMIT_MAX_REQUESTS = 3

export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB, RESEND_API_KEY, FROM_EMAIL, BASE_URL, RATE_LIMIT } = context.env

  try {
    // Validate session
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Check if email is already verified
    const alreadyVerified = await isEmailVerified(DB, session.userId)
    if (alreadyVerified) {
      return errorResponse('Email is already verified', 400)
    }

    // Check rate limit
    if (RATE_LIMIT) {
      const rateLimitKey = `resend-verification:${session.userId}`
      const currentCount = await RATE_LIMIT.get(rateLimitKey)
      const count = currentCount ? parseInt(currentCount, 10) : 0

      if (count >= RATE_LIMIT_MAX_REQUESTS) {
        return errorResponse(
          'Too many verification email requests. Please try again later.',
          429
        )
      }

      // Increment count with TTL
      await RATE_LIMIT.put(rateLimitKey, String(count + 1), {
        expirationTtl: Math.ceil(RATE_LIMIT_WINDOW_MS / 1000),
      })
    }

    // Check if RESEND_API_KEY is configured
    if (!RESEND_API_KEY) {
      return errorResponse('Email service not configured', 503)
    }

    // Get user email
    const user = await DB.prepare(`
      SELECT email FROM users WHERE id = ?
    `).bind(session.userId).first<{ email: string }>()

    if (!user) {
      return errorResponse('User not found', 404)
    }

    // Delete existing unused tokens for this user
    await deleteUnusedTokens(DB, session.userId)

    // Create new verification token
    const token = await createVerificationToken(DB, session.userId)

    // Send verification email
    const result = await sendVerificationEmail(
      {
        apiKey: RESEND_API_KEY,
        fromAddress: FROM_EMAIL,
        baseUrl: BASE_URL,
      },
      user.email,
      token
    )

    if (!result.success) {
      console.error('Failed to resend verification email:', result.error)
      return errorResponse('Failed to send verification email. Please try again.', 500)
    }

    return jsonResponse({
      success: true,
      message: 'Verification email sent',
    })
  } catch (error) {
    console.error('POST /api/auth/resend-verification error:', error)
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
