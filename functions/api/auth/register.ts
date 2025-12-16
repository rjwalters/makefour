import { z } from 'zod'
import * as bcrypt from 'bcryptjs'
import { registerRequestSchema, formatZodError } from '../../lib/schemas'
import { generateDEK, encryptDEK } from '../../lib/crypto'
import { createVerificationToken, sendVerificationEmail } from '../../lib/email'
import { createDb } from '../../../shared/db/client'
import { users } from '../../../shared/db/schema'
import { eq, sql } from 'drizzle-orm'

interface Env {
  DB: D1Database
  RESEND_API_KEY?: string
  FROM_EMAIL?: string
  BASE_URL?: string
}

export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB, RESEND_API_KEY, FROM_EMAIL, BASE_URL } = context.env
  const db = createDb(DB)

  try {
    // Parse and validate request body
    const body = await context.request.json()
    const { email, password, username } = registerRequestSchema.parse(body)

    // Check if email already exists
    const existingEmail = await db.query.users.findFirst({
      where: eq(users.email, email),
      columns: { id: true },
    })

    if (existingEmail) {
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

    // Check if username is already taken (case-insensitive)
    const existingUsername = await db.query.users.findFirst({
      where: sql`LOWER(${users.username}) = LOWER(${username})`,
      columns: { id: true },
    })

    if (existingUsername) {
      return new Response(
        JSON.stringify({
          error: 'Username taken',
          details: 'This username is already taken',
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

    // Email starts unverified - user must click verification link
    const email_verified = 0

    // Create user with encrypted_dek and username
    await db.insert(users).values({
      id: user_id,
      email,
      emailVerified: email_verified,
      passwordHash: password_hash,
      encryptedDek: encrypted_dek,
      username,
      createdAt: now,
      lastLogin: now,
      updatedAt: now,
    })

    // Send verification email if RESEND_API_KEY is configured
    let verificationSent = false
    if (RESEND_API_KEY) {
      try {
        const token = await createVerificationToken(DB, user_id)
        const result = await sendVerificationEmail(
          {
            apiKey: RESEND_API_KEY,
            fromAddress: FROM_EMAIL,
            baseUrl: BASE_URL,
          },
          email,
          token
        )
        verificationSent = result.success
        if (!result.success) {
          console.error('Failed to send verification email:', result.error)
        }
      } catch (emailError) {
        console.error('Error sending verification email:', emailError)
        // Don't fail registration if email fails
      }
    } else {
      console.warn('RESEND_API_KEY not configured - skipping verification email')
    }

    // Fetch created user
    const user = await db.query.users.findFirst({
      where: eq(users.id, user_id),
      columns: {
        id: true,
        email: true,
        emailVerified: true,
        createdAt: true,
        lastLogin: true,
        updatedAt: true,
      },
    })

    return new Response(
      JSON.stringify({
        user: {
          id: user?.id,
          email: user?.email,
          email_verified: user?.emailVerified,
          created_at: user?.createdAt,
          last_login: user?.lastLogin,
          updated_at: user?.updatedAt,
        },
        message: 'Registration successful',
        verificationEmailSent: verificationSent,
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
