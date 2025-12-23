/**
 * Live games API endpoint
 *
 * GET /api/games/live - Get list of active spectatable games
 */

import { eq, and, gte, lte, desc, count, sql } from 'drizzle-orm'
import { alias } from 'drizzle-orm/sqlite-core'
import { createDb } from '../../../shared/db/client'
import { activeGames, users, botPersonas } from '../../../shared/db/schema'
import { jsonResponse } from '../../lib/auth'

interface Env {
  DB: D1Database
}

interface LiveGameRow {
  id: string
  player1_id: string
  player2_id: string
  moves: string
  current_turn: number
  status: string
  mode: string
  player1_rating: number
  player2_rating: number
  spectator_count: number
  created_at: number
  updated_at: number
  player1_email: string
  player2_email: string
  player1_username: string | null
  player2_username: string | null
  // Bot vs bot fields
  is_bot_vs_bot: number
  bot1_persona_id: string | null
  bot2_persona_id: string | null
  bot1_name: string | null
  bot2_name: string | null
  next_move_at: number | null
}

export interface LiveGame {
  id: string
  player1: {
    rating: number
    displayName: string
    isBot?: boolean
    personaId?: string
  }
  player2: {
    rating: number
    displayName: string
    isBot?: boolean
    personaId?: string
  }
  moveCount: number
  currentTurn: 1 | 2
  mode: 'ranked' | 'casual'
  spectatorCount: number
  createdAt: number
  updatedAt: number
  // Bot vs bot specific fields
  isBotVsBot?: boolean
  nextMoveAt?: number | null
}

/**
 * GET /api/games/live - Get list of active spectatable games
 *
 * Query parameters:
 * - limit: number (default 20, max 50)
 * - offset: number (default 0)
 * - minRating: number (filter by minimum average rating)
 * - maxRating: number (filter by maximum average rating)
 * - mode: 'ranked' | 'casual' (filter by game mode)
 */
export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB } = context.env
  const db = createDb(DB)
  const url = new URL(context.request.url)

  // Parse query parameters
  const limit = Math.min(Math.max(parseInt(url.searchParams.get('limit') || '20'), 1), 50)
  const offset = Math.max(parseInt(url.searchParams.get('offset') || '0'), 0)
  const minRating = url.searchParams.get('minRating')
  const maxRating = url.searchParams.get('maxRating')
  const mode = url.searchParams.get('mode')

  try {
    // Build query with optional filters
    const conditions = [eq(activeGames.status, 'active'), eq(activeGames.spectatable, 1)]

    if (minRating) {
      conditions.push(
        gte(sql`(${activeGames.player1Rating} + ${activeGames.player2Rating}) / 2`, parseInt(minRating))
      )
    }

    if (maxRating) {
      conditions.push(
        lte(sql`(${activeGames.player1Rating} + ${activeGames.player2Rating}) / 2`, parseInt(maxRating))
      )
    }

    if (mode && (mode === 'ranked' || mode === 'casual')) {
      conditions.push(eq(activeGames.mode, mode))
    }

    const whereCondition = and(...conditions)

    // Get total count for pagination
    const countResult = await db
      .select({ total: count() })
      .from(activeGames)
      .where(whereCondition)

    const total = countResult[0]?.total ?? 0

    // Get games with player info (including bot vs bot games)
    // Create proper SQL aliases for self-joins
    const u1 = alias(users, 'u1')
    const u2 = alias(users, 'u2')
    const bp1 = alias(botPersonas, 'bp1')
    const bp2 = alias(botPersonas, 'bp2')

    const games = await db
      .select({
        id: activeGames.id,
        player1_id: activeGames.player1Id,
        player2_id: activeGames.player2Id,
        moves: activeGames.moves,
        current_turn: activeGames.currentTurn,
        status: activeGames.status,
        mode: activeGames.mode,
        player1_rating: activeGames.player1Rating,
        player2_rating: activeGames.player2Rating,
        spectator_count: activeGames.spectatorCount,
        created_at: activeGames.createdAt,
        updated_at: activeGames.updatedAt,
        player1_email: sql<string>`${u1.email}`,
        player2_email: sql<string>`${u2.email}`,
        player1_username: sql<string | null>`${u1.username}`,
        player2_username: sql<string | null>`${u2.username}`,
        is_bot_vs_bot: activeGames.isBotVsBot,
        bot1_persona_id: activeGames.bot1PersonaId,
        bot2_persona_id: activeGames.bot2PersonaId,
        bot1_name: sql<string | null>`${bp1.name}`,
        bot2_name: sql<string | null>`${bp2.name}`,
        next_move_at: activeGames.nextMoveAt,
      })
      .from(activeGames)
      .innerJoin(u1, eq(activeGames.player1Id, u1.id))
      .innerJoin(u2, eq(activeGames.player2Id, u2.id))
      .leftJoin(bp1, eq(activeGames.bot1PersonaId, bp1.id))
      .leftJoin(bp2, eq(activeGames.bot2PersonaId, bp2.id))
      .where(whereCondition)
      .orderBy(desc(activeGames.spectatorCount), desc(activeGames.updatedAt))
      .limit(limit)
      .offset(offset)

    // Transform to response format (hide full email for privacy, show bot names for bot games)
    const liveGames: LiveGame[] = games.map((game) => {
      const moves = JSON.parse(game.moves) as number[]
      const isBotVsBot = game.is_bot_vs_bot === 1

      // For bot vs bot games, show bot names; otherwise prefer username, fall back to masked email
      const player1DisplayName = isBotVsBot && game.bot1_name
        ? game.bot1_name
        : game.player1_username || maskEmail(game.player1_email)
      const player2DisplayName = isBotVsBot && game.bot2_name
        ? game.bot2_name
        : game.player2_username || maskEmail(game.player2_email)

      return {
        id: game.id,
        player1: {
          rating: game.player1_rating,
          displayName: player1DisplayName,
          isBot: isBotVsBot,
          personaId: game.bot1_persona_id || undefined,
        },
        player2: {
          rating: game.player2_rating,
          displayName: player2DisplayName,
          isBot: isBotVsBot,
          personaId: game.bot2_persona_id || undefined,
        },
        moveCount: moves.length,
        currentTurn: game.current_turn as 1 | 2,
        mode: game.mode as 'ranked' | 'casual',
        spectatorCount: game.spectator_count,
        createdAt: game.created_at,
        updatedAt: game.updated_at,
        isBotVsBot,
        nextMoveAt: isBotVsBot ? game.next_move_at : undefined,
      }
    })

    return jsonResponse({
      games: liveGames,
      total,
      limit,
      offset,
    })
  } catch (error) {
    console.error('GET /api/games/live error:', error)
    return jsonResponse({ error: 'Internal server error' }, { status: 500 })
  }
}

/**
 * Mask email for display (e.g., "user@example.com" -> "us***@example.com")
 */
function maskEmail(email: string): string {
  const [local, domain] = email.split('@')
  if (!domain) return email
  if (local.length <= 2) return `${local}***@${domain}`
  return `${local.slice(0, 2)}***@${domain}`
}

/**
 * Handle OPTIONS for CORS preflight
 */
export async function onRequestOptions() {
  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  })
}
