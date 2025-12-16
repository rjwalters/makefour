import { createDb } from '../../../shared/db/client'
import { sessionTokens } from '../../../shared/db/schema'
import { eq } from 'drizzle-orm'

interface Env {
  DB: D1Database
}

export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB } = context.env
  const db = createDb(DB)

  try {
    // Get session token from Authorization header
    const authHeader = context.request.headers.get('Authorization')
    const session_token = authHeader?.replace('Bearer ', '')

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

    // Delete session token
    await db.delete(sessionTokens).where(eq(sessionTokens.id, session_token))

    return new Response(
      JSON.stringify({
        message: 'Logged out successfully',
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }
    )
  } catch (error) {
    console.error('Logout error:', error)

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
