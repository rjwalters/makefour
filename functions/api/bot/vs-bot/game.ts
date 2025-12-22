/**
 * Bot vs Bot Game API endpoint
 *
 * POST /api/bot/vs-bot/game - Create a new bot vs bot game
 * GET /api/bot/vs-bot/game - List active bot vs bot games
 */

import { jsonResponse } from '../../../lib/auth'
import { z } from 'zod'
import { DEFAULT_BOT_PERSONAS } from '../../../lib/botPersonas'
import { type BotPersonaRow, type UserRow } from '../../../lib/types'
import { createDb } from '../../../../shared/db/client'
import { users, activeGames, botPersonas } from '../../../../shared/db/schema'
import { eq, and, sql } from 'drizzle-orm'

interface Env {
  DB: D1Database
}

/**
 * Generate bot user ID from persona ID
 */
function getBotUserId(personaId: string): string {
  return `bot_${personaId}`
}

// Default time control for bot vs bot games: 2 minutes (faster than human games)
const DEFAULT_TIME_CONTROL_MS = 120000

// Default move delay: 2 seconds between moves for watchability
const DEFAULT_MOVE_DELAY_MS = 2000

// Maximum concurrent bot vs bot games
const MAX_CONCURRENT_BOT_GAMES = 5

// Schema for creating a bot vs bot game
const createGameSchema = z.object({
  bot1PersonaId: z.string(),
  bot2PersonaId: z.string(),
  moveDelayMs: z.number().int().min(500).max(10000).optional().default(DEFAULT_MOVE_DELAY_MS),
  timeControlMs: z.number().int().min(30000).max(600000).optional().default(DEFAULT_TIME_CONTROL_MS),
})

/**
 * POST /api/bot/vs-bot/game - Create a new bot vs bot game
 *
 * Creates a new game between two bot personas. The game runs asynchronously
 * with moves made at intervals defined by moveDelayMs.
 */
export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    const db = createDb(DB)
    const body = await context.request.json()
    const parseResult = createGameSchema.safeParse(body)

    if (!parseResult.success) {
      return jsonResponse({ error: parseResult.error.errors[0].message }, { status: 400 })
    }

    const { bot1PersonaId, bot2PersonaId, moveDelayMs, timeControlMs } = parseResult.data

    // Validate personas exist
    const [persona1, persona2] = await Promise.all([
      db.query.botPersonas.findFirst({
        where: and(eq(botPersonas.id, bot1PersonaId), eq(botPersonas.isActive, 1)),
        columns: {
          id: true,
          name: true,
          currentElo: true,
          aiEngine: true,
          aiConfig: true,
        },
      }),
      db.query.botPersonas.findFirst({
        where: and(eq(botPersonas.id, bot2PersonaId), eq(botPersonas.isActive, 1)),
        columns: {
          id: true,
          name: true,
          currentElo: true,
          aiEngine: true,
          aiConfig: true,
        },
      }),
    ])

    if (!persona1) {
      return jsonResponse({ error: `Bot persona '${bot1PersonaId}' not found` }, { status: 404 })
    }

    if (!persona2) {
      return jsonResponse({ error: `Bot persona '${bot2PersonaId}' not found` }, { status: 404 })
    }

    // Check concurrent game limit
    const activeGamesCount = await db.select({
      count: sql<number>`COUNT(*)`,
    })
    .from(activeGames)
    .where(and(eq(activeGames.isBotVsBot, 1), eq(activeGames.status, 'active')))
    .then(rows => rows[0])

    if ((activeGamesCount?.count ?? 0) >= MAX_CONCURRENT_BOT_GAMES) {
      return jsonResponse(
        { error: `Maximum concurrent bot games (${MAX_CONCURRENT_BOT_GAMES}) reached` },
        { status: 429 }
      )
    }

    // Get or create bot user IDs
    const bot1UserId = getBotUserId(bot1PersonaId)
    const bot2UserId = getBotUserId(bot2PersonaId)

    // Get bot ratings from users table (or fall back to persona)
    const [bot1User, bot2User] = await Promise.all([
      db.query.users.findFirst({
        where: and(eq(users.id, bot1UserId), eq(users.isBot, 1)),
        columns: {
          id: true,
          rating: true,
        },
      }),
      db.query.users.findFirst({
        where: and(eq(users.id, bot2UserId), eq(users.isBot, 1)),
        columns: {
          id: true,
          rating: true,
        },
      }),
    ])

    const bot1Rating = bot1User?.rating ?? persona1.currentElo
    const bot2Rating = bot2User?.rating ?? persona2.currentElo

    const now = Date.now()
    const gameId = crypto.randomUUID()

    // Create the game
    await db.insert(activeGames).values({
      id: gameId,
      player1Id: bot1UserId,
      player2Id: bot2UserId,
      moves: '[]',
      currentTurn: 1,
      status: 'active',
      mode: 'ranked',
      player1Rating: bot1Rating,
      player2Rating: bot2Rating,
      spectatable: 1,
      spectatorCount: 0,
      lastMoveAt: now,
      timeControlMs,
      player1TimeMs: timeControlMs,
      player2TimeMs: timeControlMs,
      turnStartedAt: now,
      isBotGame: 1,
      isBotVsBot: 1,
      bot1PersonaId,
      bot2PersonaId,
      moveDelayMs,
      nextMoveAt: now + moveDelayMs,
      createdAt: now,
      updatedAt: now,
    })

    return jsonResponse({
      gameId,
      bot1: {
        personaId: bot1PersonaId,
        name: persona1.name,
        rating: bot1Rating,
      },
      bot2: {
        personaId: bot2PersonaId,
        name: persona2.name,
        rating: bot2Rating,
      },
      moveDelayMs,
      timeControlMs,
      nextMoveAt: now + moveDelayMs,
    })
  } catch (error) {
    console.error('POST /api/bot/vs-bot/game error:', error)
    return jsonResponse({ error: 'Internal server error' }, { status: 500 })
  }
}

/**
 * GET /api/bot/vs-bot/game - List active bot vs bot games
 *
 * Query parameters:
 * - limit: number (default 10, max 50)
 * - offset: number (default 0)
 * - status: 'active' | 'completed' | 'all' (default 'active')
 */
export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB } = context.env
  const url = new URL(context.request.url)

  const limit = Math.min(Math.max(parseInt(url.searchParams.get('limit') || '10'), 1), 50)
  const offset = Math.max(parseInt(url.searchParams.get('offset') || '0'), 0)
  const status = url.searchParams.get('status') || 'active'

  try {
    const db = createDb(DB)

    const whereConditions = [eq(activeGames.isBotVsBot, 1)]
    if (status !== 'all') {
      whereConditions.push(eq(activeGames.status, status))
    }

    // Get total count
    const countResult = await db.select({
      total: sql<number>`COUNT(*)`,
    })
    .from(activeGames)
    .where(and(...whereConditions))
    .then(rows => rows[0])

    const total = countResult?.total ?? 0

    // Get games with persona info using a join
    const gamesData = await db.select({
      id: activeGames.id,
      player1Id: activeGames.player1Id,
      player2Id: activeGames.player2Id,
      moves: activeGames.moves,
      currentTurn: activeGames.currentTurn,
      status: activeGames.status,
      winner: activeGames.winner,
      player1Rating: activeGames.player1Rating,
      player2Rating: activeGames.player2Rating,
      spectatorCount: activeGames.spectatorCount,
      moveDelayMs: activeGames.moveDelayMs,
      nextMoveAt: activeGames.nextMoveAt,
      createdAt: activeGames.createdAt,
      updatedAt: activeGames.updatedAt,
      bot1PersonaId: activeGames.bot1PersonaId,
      bot2PersonaId: activeGames.bot2PersonaId,
      bot1Name: sql<string>`bp1.name`,
      bot2Name: sql<string>`bp2.name`,
    })
    .from(activeGames)
    .leftJoin(sql`bot_personas bp1`, sql`${activeGames.bot1PersonaId} = bp1.id`)
    .leftJoin(sql`bot_personas bp2`, sql`${activeGames.bot2PersonaId} = bp2.id`)
    .where(and(...whereConditions))
    .orderBy(sql`${activeGames.spectatorCount} DESC, ${activeGames.updatedAt} DESC`)
    .limit(limit)
    .offset(offset)

    const botGames = gamesData.map((game) => {
      const moves = JSON.parse(game.moves) as number[]
      return {
        id: game.id,
        bot1: {
          personaId: game.bot1PersonaId,
          name: game.bot1Name || 'Bot 1',
          rating: game.player1Rating,
        },
        bot2: {
          personaId: game.bot2PersonaId,
          name: game.bot2Name || 'Bot 2',
          rating: game.player2Rating,
        },
        moveCount: moves.length,
        currentTurn: game.currentTurn,
        status: game.status,
        winner: game.winner,
        spectatorCount: game.spectatorCount,
        nextMoveAt: game.nextMoveAt,
        createdAt: game.createdAt,
        updatedAt: game.updatedAt,
      }
    })

    return jsonResponse({
      games: botGames,
      total,
      limit,
      offset,
    })
  } catch (error) {
    console.error('GET /api/bot/vs-bot/game error:', error)
    return jsonResponse({ error: 'Internal server error' }, { status: 500 })
  }
}

export async function onRequestOptions() {
  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  })
}
