/**
 * Debug Login API - Creates or logs in as a debug test user
 *
 * POST /api/debug/login - Get a session token for a debug user
 *
 * This endpoint is for testing purposes only.
 * It creates a debug user if one doesn't exist, and returns a session token.
 */

import { createDb } from '../../../shared/db/client'
import { users, sessionTokens } from '../../../shared/db/schema'
import { eq } from 'drizzle-orm'
import * as bcrypt from 'bcryptjs'

interface Env {
  DB: D1Database
}

const DEBUG_USER_EMAIL = 'debug@makefour.test'
const DEBUG_USER_PASSWORD = 'DebugTest123!'
const DEBUG_USER_USERNAME = 'DebugTester'
const SESSION_DURATION_MS = 24 * 60 * 60 * 1000 // 24 hours

export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB } = context.env
  const db = createDb(DB)

  try {
    // Check if debug user exists
    let user = await db.query.users.findFirst({
      where: eq(users.email, DEBUG_USER_EMAIL),
    })

    const now = Date.now()

    // Create debug user if it doesn't exist
    if (!user) {
      const userId = crypto.randomUUID()
      const passwordHash = await bcrypt.hash(DEBUG_USER_PASSWORD, 10)

      await db.insert(users).values({
        id: userId,
        email: DEBUG_USER_EMAIL,
        username: DEBUG_USER_USERNAME,
        emailVerified: 1,
        passwordHash,
        rating: 1200,
        gamesPlayed: 0,
        wins: 0,
        losses: 0,
        draws: 0,
        preferences: '{}',
        createdAt: now,
        lastLogin: now,
        updatedAt: now,
      })

      user = await db.query.users.findFirst({
        where: eq(users.email, DEBUG_USER_EMAIL),
      })
    }

    if (!user) {
      return new Response(
        JSON.stringify({ error: 'Failed to create debug user' }),
        { status: 500, headers: { 'Content-Type': 'application/json' } }
      )
    }

    // Create session token
    const sessionTokenId = crypto.randomUUID()
    const expiresAt = now + SESSION_DURATION_MS

    await db.insert(sessionTokens).values({
      id: sessionTokenId,
      userId: user.id,
      expiresAt,
      createdAt: now,
    })

    // Update last login
    await db.update(users)
      .set({ lastLogin: now, updatedAt: now })
      .where(eq(users.id, user.id))

    return new Response(
      JSON.stringify({
        success: true,
        user: {
          id: user.id,
          email: user.email,
          username: user.username || DEBUG_USER_USERNAME,
        },
        session_token: sessionTokenId,
        message: 'Debug user logged in successfully',
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }
    )
  } catch (error) {
    console.error('Debug login error:', error)
    return new Response(
      JSON.stringify({
        error: 'Debug login failed',
        details: error instanceof Error ? error.message : String(error),
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    )
  }
}

export async function onRequestOptions() {
  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  })
}
