import { z } from 'zod'
import * as bcrypt from 'bcryptjs'
import { loginRequestSchema, formatZodError } from '../../lib/schemas'

interface Env {
  DB: D1Database
}

const SESSION_DURATION_MS = 30 * 24 * 60 * 60 * 1000 // 30 days

export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    // Parse and validate request body
    const body = await context.request.json()
    const { email, password } = loginRequestSchema.parse(body)

    // Find user by email
    const user = await DB.prepare('SELECT * FROM users WHERE email = ?').bind(email).first()

    if (!user || !user.password_hash) {
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
    const password_match = await bcrypt.compare(password, user.password_hash as string)

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

    await DB.prepare(
      'INSERT INTO session_tokens (id, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)'
    )
      .bind(session_token_id, user.id, expires_at, now)
      .run()

    // Update last login
    await DB.prepare('UPDATE users SET last_login = ?, updated_at = ? WHERE id = ?')
      .bind(now, now, user.id)
      .run()

    // Return user data, session token, and encrypted DEK (excluding sensitive fields)
    const publicUser = {
      id: user.id,
      email: user.email,
      email_verified: user.email_verified === 1,
      created_at: user.created_at,
      last_login: now,
      updated_at: now,
    }

    return new Response(
      JSON.stringify({
        user: publicUser,
        session_token: session_token_id,
        encrypted_dek: user.encrypted_dek, // Client needs this to decrypt DEK with password
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
