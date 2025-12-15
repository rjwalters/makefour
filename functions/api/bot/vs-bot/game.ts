/**
 * Bot vs Bot Game API endpoint
 *
 * POST /api/bot/vs-bot/game - Create a new bot vs bot game
 * GET /api/bot/vs-bot/game - List active bot vs bot games
 */

import { jsonResponse } from '../../../lib/auth'
import { z } from 'zod'
import { DEFAULT_BOT_PERSONAS } from '../../../lib/botPersonas'

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

interface BotPersonaRow {
  id: string
  name: string
  current_elo: number
  ai_engine: string
  ai_config: string
}

interface BotUserRow {
  id: string
  rating: number
}

/**
 * POST /api/bot/vs-bot/game - Create a new bot vs bot game
 *
 * Creates a new game between two bot personas. The game runs asynchronously
 * with moves made at intervals defined by moveDelayMs.
 */
export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    const body = await context.request.json()
    const parseResult = createGameSchema.safeParse(body)

    if (!parseResult.success) {
      return jsonResponse({ error: parseResult.error.errors[0].message }, { status: 400 })
    }

    const { bot1PersonaId, bot2PersonaId, moveDelayMs, timeControlMs } = parseResult.data

    // Validate personas exist
    const [persona1, persona2] = await Promise.all([
      DB.prepare(`
        SELECT id, name, current_elo, ai_engine, ai_config
        FROM bot_personas
        WHERE id = ? AND is_active = 1
      `)
        .bind(bot1PersonaId)
        .first<BotPersonaRow>(),
      DB.prepare(`
        SELECT id, name, current_elo, ai_engine, ai_config
        FROM bot_personas
        WHERE id = ? AND is_active = 1
      `)
        .bind(bot2PersonaId)
        .first<BotPersonaRow>(),
    ])

    if (!persona1) {
      return jsonResponse({ error: `Bot persona '${bot1PersonaId}' not found` }, { status: 404 })
    }

    if (!persona2) {
      return jsonResponse({ error: `Bot persona '${bot2PersonaId}' not found` }, { status: 404 })
    }

    // Check concurrent game limit
    const activeGamesCount = await DB.prepare(`
      SELECT COUNT(*) as count
      FROM active_games
      WHERE is_bot_vs_bot = 1 AND status = 'active'
    `).first<{ count: number }>()

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
      DB.prepare('SELECT id, rating FROM users WHERE id = ? AND is_bot = 1')
        .bind(bot1UserId)
        .first<BotUserRow>(),
      DB.prepare('SELECT id, rating FROM users WHERE id = ? AND is_bot = 1')
        .bind(bot2UserId)
        .first<BotUserRow>(),
    ])

    const bot1Rating = bot1User?.rating ?? persona1.current_elo
    const bot2Rating = bot2User?.rating ?? persona2.current_elo

    const now = Date.now()
    const gameId = crypto.randomUUID()

    // Create the game
    await DB.prepare(`
      INSERT INTO active_games (
        id, player1_id, player2_id, moves, current_turn, status, mode,
        player1_rating, player2_rating, spectatable, spectator_count,
        last_move_at, time_control_ms, player1_time_ms, player2_time_ms,
        turn_started_at, is_bot_game, is_bot_vs_bot,
        bot1_persona_id, bot2_persona_id, move_delay_ms, next_move_at,
        created_at, updated_at
      )
      VALUES (?, ?, ?, '[]', 1, 'active', 'ranked', ?, ?, 1, 0, ?, ?, ?, ?, ?, 1, 1, ?, ?, ?, ?, ?, ?)
    `).bind(
      gameId,
      bot1UserId,
      bot2UserId,
      bot1Rating,
      bot2Rating,
      now,
      timeControlMs,
      timeControlMs,
      timeControlMs,
      now, // turn_started_at
      bot1PersonaId,
      bot2PersonaId,
      moveDelayMs,
      now + moveDelayMs, // next_move_at - first move after delay
      now,
      now
    ).run()

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
    let whereClause = 'ag.is_bot_vs_bot = 1'
    const params: (string | number)[] = []

    if (status !== 'all') {
      whereClause += ' AND ag.status = ?'
      params.push(status)
    }

    // Get total count
    const countResult = await DB.prepare(`
      SELECT COUNT(*) as total
      FROM active_games ag
      WHERE ${whereClause}
    `)
      .bind(...params)
      .first<{ total: number }>()

    const total = countResult?.total ?? 0

    // Get games with persona info
    const games = await DB.prepare(`
      SELECT
        ag.id,
        ag.player1_id,
        ag.player2_id,
        ag.moves,
        ag.current_turn,
        ag.status,
        ag.winner,
        ag.player1_rating,
        ag.player2_rating,
        ag.spectator_count,
        ag.move_delay_ms,
        ag.next_move_at,
        ag.created_at,
        ag.updated_at,
        ag.bot1_persona_id,
        ag.bot2_persona_id,
        bp1.name as bot1_name,
        bp2.name as bot2_name
      FROM active_games ag
      LEFT JOIN bot_personas bp1 ON ag.bot1_persona_id = bp1.id
      LEFT JOIN bot_personas bp2 ON ag.bot2_persona_id = bp2.id
      WHERE ${whereClause}
      ORDER BY ag.spectator_count DESC, ag.updated_at DESC
      LIMIT ? OFFSET ?
    `)
      .bind(...params, limit, offset)
      .all()

    const botGames = games.results.map((game: any) => {
      const moves = JSON.parse(game.moves) as number[]
      return {
        id: game.id,
        bot1: {
          personaId: game.bot1_persona_id,
          name: game.bot1_name || 'Bot 1',
          rating: game.player1_rating,
        },
        bot2: {
          personaId: game.bot2_persona_id,
          name: game.bot2_name || 'Bot 2',
          rating: game.player2_rating,
        },
        moveCount: moves.length,
        currentTurn: game.current_turn,
        status: game.status,
        winner: game.winner,
        spectatorCount: game.spectator_count,
        nextMoveAt: game.next_move_at,
        createdAt: game.created_at,
        updatedAt: game.updated_at,
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
