/**
 * Bot vs Bot Game Tick API endpoint
 *
 * POST /api/bot/vs-bot/tick - Advance all ready bot vs bot games
 *
 * This endpoint is designed to be called by an external scheduler (cron job,
 * GitHub Actions, etc.) to advance bot vs bot games. It requires a secret
 * token for authentication to prevent abuse.
 *
 * Authentication:
 * - Set BOT_TICK_SECRET environment variable in Cloudflare
 * - Pass the secret in the Authorization header: "Bearer <secret>"
 */

import { jsonResponse } from '../../../lib/auth'
import {
  advanceGame,
  createRandomBotGames,
  findReadyGames,
  getActiveBotGameCount,
  MAX_GAMES_PER_TICK,
  TARGET_ACTIVE_GAMES,
  type AdvanceGameResult,
} from '../../../lib/botVsBotGame'

interface Env {
  DB: D1Database
  BOT_TICK_SECRET?: string
}

/**
 * POST /api/bot/vs-bot/tick - Advance all ready bot vs bot games
 *
 * Requires authentication via BOT_TICK_SECRET environment variable.
 *
 * Query parameters:
 * - limit: number (default 10, max 10) - max games to process
 * - createNew: boolean (default false) - whether to create new games if under threshold
 */
export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB, BOT_TICK_SECRET } = context.env
  const url = new URL(context.request.url)

  // Authenticate the request
  if (BOT_TICK_SECRET) {
    const authHeader = context.request.headers.get('Authorization')
    const expectedAuth = `Bearer ${BOT_TICK_SECRET}`

    if (!authHeader || authHeader !== expectedAuth) {
      return jsonResponse({ error: 'Unauthorized' }, { status: 401 })
    }
  } else {
    // If no secret is configured, log a warning but allow in development
    console.warn('BOT_TICK_SECRET not configured - tick endpoint is unprotected')
  }

  const limit = Math.min(
    Math.max(parseInt(url.searchParams.get('limit') || '10'), 1),
    MAX_GAMES_PER_TICK
  )
  const createNew = url.searchParams.get('createNew') === 'true'

  const now = Date.now()
  const results: AdvanceGameResult[] = []

  try {
    // Find active bot vs bot games that are ready for next move
    const readyGames = await findReadyGames(DB, now, limit)

    // Process each game
    for (const game of readyGames) {
      try {
        const result = await advanceGame(DB, game, now)
        results.push(result)
      } catch (error) {
        console.error(`Error advancing game ${game.id}:`, error)
        results.push({
          gameId: game.id,
          status: 'error',
          error: error instanceof Error ? error.message : 'Unknown error',
        })
      }
    }

    // Optionally create new games if under threshold
    let newGames: string[] = []
    if (createNew) {
      const currentActive = await getActiveBotGameCount(DB)

      if (currentActive < TARGET_ACTIVE_GAMES) {
        const gamesToCreate = Math.min(TARGET_ACTIVE_GAMES - currentActive, 2)
        newGames = await createRandomBotGames(DB, gamesToCreate, now)
      }
    }

    return jsonResponse({
      processed: results.length,
      results,
      newGamesCreated: newGames,
      timestamp: now,
    })
  } catch (error) {
    console.error('POST /api/bot/vs-bot/tick error:', error)
    return jsonResponse({ error: 'Internal server error' }, { status: 500 })
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
