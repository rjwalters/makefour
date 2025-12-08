import { z } from 'zod'
import * as bcrypt from 'bcryptjs'
import { registerRequestSchema, formatZodError } from '../../lib/schemas'
import { generateDEK, encryptDEK } from '../../lib/crypto'

interface Env {
  DB: D1Database
}

export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    // Parse and validate request body
    const body = await context.request.json()
    const { email, password } = registerRequestSchema.parse(body)

    // Check if user already exists
    const existingUser = await DB.prepare('SELECT id FROM users WHERE email = ?').bind(email).first()

    if (existingUser) {
      return new Response(
        JSON.stringify({
          error: 'User already exists',
          details: 'An account with this email address already exists',
        }),
        {
          status: 409,
          headers: { 'Content-Type': 'application/json' },
        }
      )
    }

    // Hash password
    const password_hash = await bcrypt.hash(password, 10)

    // Generate DEK (Data Encryption Key) for encrypting user data
    const dek = await generateDEK()

    // Encrypt DEK with password-derived KEK (Key Encryption Key)
    const encrypted_dek = await encryptDEK(dek, password)

    // Generate user ID
    const user_id = crypto.randomUUID()
    const now = Date.now()

    // For now, auto-verify email (Phase 1 - skip email confirmation)
    const email_verified = 1

    // Create user with encrypted_dek
    await DB.prepare(
      `INSERT INTO users
      (id, email, email_verified, password_hash, encrypted_dek, created_at, last_login, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
    )
      .bind(user_id, email, email_verified, password_hash, encrypted_dek, now, now, now)
      .run()

    // Fetch created user
    const user = await DB.prepare(
      'SELECT id, email, email_verified, created_at, last_login, updated_at FROM users WHERE id = ?'
    )
      .bind(user_id)
      .first()

    return new Response(
      JSON.stringify({
        user,
        message: 'Registration successful',
      }),
      {
        status: 201,
        headers: { 'Content-Type': 'application/json' },
      }
    )
  } catch (error) {
    console.error('Registration error:', error)

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
