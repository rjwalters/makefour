/**
 * POST /api/auth/verify-email - Verify a user's email address
 *
 * Request body: { token: string }
 */

import { z } from 'zod'
import { validateVerificationToken, markEmailVerified } from '../../lib/email'
import { errorResponse, jsonResponse } from '../../lib/auth'
import { formatZodError } from '../../lib/schemas'

interface Env {
  DB: D1Database
}

const verifyEmailSchema = z.object({
  token: z.string().uuid('Invalid verification token format'),
})

export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    // Parse request body
    const body = await context.request.json().catch(() => ({}))
    const parseResult = verifyEmailSchema.safeParse(body)

    if (!parseResult.success) {
      const formattedError = formatZodError(parseResult.error)
      return errorResponse(formattedError.details || formattedError.error, 400)
    }

    const { token } = parseResult.data

    // Validate token
    const result = await validateVerificationToken(DB, token)

    if (!result.valid) {
      return errorResponse(result.error, 400)
    }

    // Mark email as verified
    await markEmailVerified(DB, result.userId)

    // Get user info for response
    const user = await DB.prepare(`
      SELECT id, email, email_verified, rating, games_played, wins, losses, draws,
             created_at, last_login, updated_at
      FROM users WHERE id = ?
    `).bind(result.userId).first<{
      id: string
      email: string
      email_verified: number
      rating: number
      games_played: number
      wins: number
      losses: number
      draws: number
      created_at: number
      last_login: number
      updated_at: number
    }>()

    if (!user) {
      return errorResponse('User not found', 404)
    }

    return jsonResponse({
      success: true,
      message: 'Email verified successfully',
      user: {
        id: user.id,
        email: user.email,
        email_verified: user.email_verified === 1,
        rating: user.rating,
        gamesPlayed: user.games_played,
        wins: user.wins,
        losses: user.losses,
        draws: user.draws,
        createdAt: user.created_at,
        lastLogin: user.last_login,
        updatedAt: user.updated_at,
      },
    })
  } catch (error) {
    console.error('POST /api/auth/verify-email error:', error)
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
