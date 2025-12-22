import { z } from 'zod'
import * as bcrypt from 'bcryptjs'
import { loginRequestSchema, formatZodError } from '../../lib/schemas'
import { createDb } from '../../../shared/db/client'
import { users, sessionTokens } from '../../../shared/db/schema'
import { eq } from 'drizzle-orm'
import { SESSION_DURATION_MS } from '../../lib/types'

interface Env {
  DB: D1Database
}

export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB } = context.env
  const db = createDb(DB)

  try {
    // Parse and validate request body
    const body = await context.request.json()
    const { email, password } = loginRequestSchema.parse(body)

    // Find user by email
    const user = await db.query.users.findFirst({
      where: eq(users.email, email),
    })

    if (!user || !user.passwordHash) {
      return new Response(
        JSON.stringify({
          error: 'Invalid credentials',
          details: 'Email or password is incorrect',
        }),
        {
          status: 401,
          headers: { 'Content-Type': 'application/json' },
        }
      )
    }

    // Verify password
    const password_match = await bcrypt.compare(password, user.passwordHash)

    if (!password_match) {
      return new Response(
        JSON.stringify({
          error: 'Invalid credentials',
          details: 'Email or password is incorrect',
        }),
        {
          status: 401,
          headers: { 'Content-Type': 'application/json' },
        }
      )
    }

    // Create session token
    const session_token_id = crypto.randomUUID()
    const now = Date.now()
    const expires_at = now + SESSION_DURATION_MS

    await db.insert(sessionTokens).values({
      id: session_token_id,
      userId: user.id,
      expiresAt: expires_at,
      createdAt: now,
    })

    // Update last login
    await db.update(users)
      .set({ lastLogin: now, updatedAt: now })
      .where(eq(users.id, user.id))

    // Return user data, session token, and encrypted DEK (excluding sensitive fields)
    const publicUser = {
      id: user.id,
      email: user.email,
      email_verified: user.emailVerified === 1,
      created_at: user.createdAt,
      last_login: now,
      updated_at: now,
    }

    return new Response(
      JSON.stringify({
        user: publicUser,
        session_token: session_token_id,
        encrypted_dek: user.encryptedDek, // Client needs this to decrypt DEK with password
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }
    )
  } catch (error) {
    console.error('Login error:', error)

    if (error instanceof z.ZodError) {
      return new Response(JSON.stringify(formatZodError(error)), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      })
    }

    return new Response(
      JSON.stringify({
        error: 'Internal server error',
        details: 'An unexpected error occurred',
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    )
  }
}
