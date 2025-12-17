/**
 * Debug Game API - Create a test game for debugging
 *
 * POST /api/debug/game - Create a new debug game with a bot
 *
 * This endpoint is for testing purposes only.
 */

import { validateSession, errorResponse, jsonResponse } from '../../lib/auth'
import { createDb } from '../../../shared/db/client'
import { activeGames, botPersonas, users } from '../../../shared/db/schema'
import { eq } from 'drizzle-orm'

interface Env {
  DB: D1Database
}

export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB } = context.env
  const db = createDb(DB)

  try {
    // Validate session
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    const now = Date.now()
    const gameId = crypto.randomUUID()

    // Get user's rating
    const user = await db.query.users.findFirst({
      where: eq(users.id, session.userId),
      columns: { rating: true },
    })
    const playerRating = user?.rating || 1200

    // Get a random bot persona for the game
    const personas = await db.query.botPersonas.findMany({
      limit: 10,
    })

    const persona = personas.length > 0
      ? personas[Math.floor(Math.random() * personas.length)]
      : null

    // Create a new active game against a bot
    await db.insert(activeGames).values({
      id: gameId,
      player1Id: session.userId,
      player2Id: persona ? `bot_${persona.id}` : 'bot-opponent',
      moves: '[]',
      currentTurn: 1,
      status: 'active',
      mode: 'casual', // Debug games are casual (no ELO impact)
      winner: null,
      player1Rating: playerRating,
      player2Rating: 1200, // Bot default rating
      spectatable: 0, // Debug games not spectatable
      spectatorCount: 0,
      lastMoveAt: now,
      timeControlMs: null,
      player1TimeMs: null,
      player2TimeMs: null,
      turnStartedAt: null,
      isBotGame: 1,
      botDifficulty: null,
      botPersonaId: persona?.id || null,
      isBotVsBot: 0,
      bot1PersonaId: null,
      bot2PersonaId: null,
      moveDelayMs: null,
      nextMoveAt: null,
      createdAt: now,
      updatedAt: now,
    })

    return jsonResponse({
      success: true,
      gameId,
      botPersonaId: persona?.id || null,
      botName: persona?.name || 'Bot',
      message: 'Debug game created successfully',
    })
  } catch (error) {
    console.error('Debug game creation error:', error)
    return errorResponse(
      `Failed to create debug game: ${error instanceof Error ? error.message : String(error)}`,
      500
    )
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
