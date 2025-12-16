import { createDb } from '../../../shared/db/client'
import { sessionTokens, users } from '../../../shared/db/schema'
import { eq } from 'drizzle-orm'

interface Env {
  DB: D1Database
}

export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB } = context.env
  const db = createDb(DB)

  try {
    // Get session token from Authorization header or query param
    const authHeader = context.request.headers.get('Authorization')
    const url = new URL(context.request.url)
    const session_token = authHeader?.replace('Bearer ', '') || url.searchParams.get('session_token')

    if (!session_token) {
      return new Response(
        JSON.stringify({
          error: 'Session token required',
        }),
        {
          status: 401,
          headers: { 'Content-Type': 'application/json' },
        }
      )
    }

    // Find session token and check if it's expired
    const session = await db.query.sessionTokens.findFirst({
      where: eq(sessionTokens.id, session_token),
    })

    if (!session) {
      return new Response(
        JSON.stringify({
          error: 'Invalid session token',
        }),
        {
          status: 401,
          headers: { 'Content-Type': 'application/json' },
        }
      )
    }

    // Check if expired
    if (session.expiresAt < Date.now()) {
      // Delete expired token
      await db.delete(sessionTokens).where(eq(sessionTokens.id, session_token))

      return new Response(
        JSON.stringify({
          error: 'Session expired',
        }),
        {
          status: 401,
          headers: { 'Content-Type': 'application/json' },
        }
      )
    }

    // Get user with rating information
    const user = await db.query.users.findFirst({
      where: eq(users.id, session.userId),
      columns: {
        id: true,
        email: true,
        emailVerified: true,
        username: true,
        rating: true,
        gamesPlayed: true,
        wins: true,
        losses: true,
        draws: true,
        createdAt: true,
        lastLogin: true,
        updatedAt: true,
      },
    })

    if (!user) {
      return new Response(
        JSON.stringify({
          error: 'User not found',
        }),
        {
          status: 404,
          headers: { 'Content-Type': 'application/json' },
        }
      )
    }

    return new Response(
      JSON.stringify({
        user: {
          id: user.id,
          email: user.email,
          email_verified: user.emailVerified === 1,
          username: user.username,
          displayName: user.username || user.email.split('@')[0],
          rating: user.rating,
          gamesPlayed: user.gamesPlayed,
          wins: user.wins,
          losses: user.losses,
          draws: user.draws,
          createdAt: user.createdAt,
          lastLogin: user.lastLogin,
          updatedAt: user.updatedAt,
        },
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }
    )
  } catch (error) {
    console.error('Me endpoint error:', error)

    return new Response(
      JSON.stringify({
        error: 'Internal server error',
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    )
  }
}
