/**
 * Bot Game API endpoint
 *
 * POST /api/bot/game - Create a new ranked bot game
 */

import { validateSession, errorResponse, jsonResponse } from '../../lib/auth'
import { z } from 'zod'

interface Env {
  DB: D1Database
}

// Bot user ID (special reserved ID for bot opponent)
const BOT_USER_ID = 'bot-opponent'

// Legacy bot ratings by difficulty (for backward compatibility)
const BOT_RATINGS: Record<string, number> = {
  beginner: 800,
  intermediate: 1200,
  expert: 1600,
  perfect: 2000,
}

// Default time control: 5 minutes
const DEFAULT_TIME_CONTROL_MS = 300000

// Schema for creating a game - supports both personaId (new) and difficulty (legacy)
const createGameSchema = z.object({
  personaId: z.string().optional(),
  difficulty: z.enum(['beginner', 'intermediate', 'expert', 'perfect']).optional(),
  playerColor: z.union([z.literal(1), z.literal(2)]).optional().default(1),
}).refine(
  (data) => data.personaId || data.difficulty,
  { message: 'Either personaId or difficulty is required' }
)

// Bot persona row from database
interface BotPersonaRow {
  id: string
  name: string
  current_elo: number
  ai_config: string
}

/**
 * POST /api/bot/game - Create a new ranked bot game
 */
export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Parse request body
    const body = await context.request.json()
    const parseResult = createGameSchema.safeParse(body)

    if (!parseResult.success) {
      return errorResponse(parseResult.error.errors[0].message, 400)
    }

    const { personaId, difficulty, playerColor } = parseResult.data

    // Resolve bot rating and configuration
    let botRating: number
    let botPersonaId: string | null = null
    let aiConfig: { searchDepth: number; errorRate: number; timeMultiplier?: number } | null = null
    let effectiveDifficulty: string | null = difficulty || null

    if (personaId) {
      // Look up the persona
      const persona = await DB.prepare(`
        SELECT id, name, current_elo, ai_config
        FROM bot_personas
        WHERE id = ? AND is_active = 1
      `)
        .bind(personaId)
        .first<BotPersonaRow>()

      if (!persona) {
        return errorResponse('Bot persona not found', 404)
      }

      botRating = persona.current_elo
      botPersonaId = persona.id
      aiConfig = JSON.parse(persona.ai_config)
      // Map persona to difficulty based on rating for backward compatibility
      if (botRating < 900) effectiveDifficulty = 'beginner'
      else if (botRating < 1300) effectiveDifficulty = 'intermediate'
      else if (botRating < 1700) effectiveDifficulty = 'expert'
      else effectiveDifficulty = 'perfect'
    } else if (difficulty) {
      botRating = BOT_RATINGS[difficulty]
    } else {
      return errorResponse('Either personaId or difficulty is required', 400)
    }

    // Check if user already has an active game
    const existingGame = await DB.prepare(`
      SELECT id FROM active_games
      WHERE (player1_id = ? OR player2_id = ?)
      AND status = 'active'
    `)
      .bind(session.userId, session.userId)
      .first()

    if (existingGame) {
      return errorResponse('You already have an active game', 400)
    }

    // Get user's current rating
    const user = await DB.prepare(`
      SELECT rating FROM users WHERE id = ?
    `)
      .bind(session.userId)
      .first<{ rating: number }>()

    if (!user) {
      return errorResponse('User not found', 404)
    }

    const now = Date.now()
    const gameId = crypto.randomUUID()

    // Determine player positions
    // If playerColor is 1, user is player1 (red, goes first)
    // If playerColor is 2, user is player2 (yellow, goes second)
    const player1Id = playerColor === 1 ? session.userId : BOT_USER_ID
    const player2Id = playerColor === 1 ? BOT_USER_ID : session.userId
    const player1Rating = playerColor === 1 ? user.rating : botRating
    const player2Rating = playerColor === 1 ? botRating : user.rating
    const botPlayer = playerColor === 1 ? 2 : 1

    // Create the game
    await DB.prepare(`
      INSERT INTO active_games (
        id, player1_id, player2_id, moves, current_turn, status, mode,
        player1_rating, player2_rating, spectatable, spectator_count,
        last_move_at, time_control_ms, player1_time_ms, player2_time_ms,
        turn_started_at, is_bot_game, bot_difficulty, bot_persona_id, created_at, updated_at
      )
      VALUES (?, ?, ?, '[]', 1, 'active', 'ranked', ?, ?, 0, 0, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
    `).bind(
      gameId,
      player1Id,
      player2Id,
      player1Rating,
      player2Rating,
      now,
      DEFAULT_TIME_CONTROL_MS,
      DEFAULT_TIME_CONTROL_MS,
      DEFAULT_TIME_CONTROL_MS,
      now, // turn_started_at
      effectiveDifficulty,
      botPersonaId,
      now,
      now
    ).run()

    // If bot goes first (player is yellow), make bot's first move
    if (botPlayer === 1) {
      // Import bot module and make move
      const { suggestMove, suggestMoveWithConfig, calculateTimeBudget, measureMoveTime } = await import('../../lib/bot')

      const timeBudget = calculateTimeBudget(DEFAULT_TIME_CONTROL_MS, 0, effectiveDifficulty || 'intermediate')
      const emptyBoard = Array.from({ length: 6 }, () => Array(7).fill(null))

      let botMove: number
      let elapsedMs: number

      if (aiConfig) {
        // Use persona's AI config
        const result = measureMoveTime(() =>
          suggestMoveWithConfig(emptyBoard, 1, aiConfig, timeBudget)
        )
        botMove = result.result
        elapsedMs = result.elapsedMs
      } else {
        // Use legacy difficulty-based AI
        const result = measureMoveTime(() =>
          suggestMove(emptyBoard, 1, effectiveDifficulty || 'intermediate', timeBudget)
        )
        botMove = result.result
        elapsedMs = result.elapsedMs
      }

      // Update game with bot's first move
      const newPlayer1Time = DEFAULT_TIME_CONTROL_MS - elapsedMs

      await DB.prepare(`
        UPDATE active_games
        SET moves = ?, current_turn = 2, last_move_at = ?,
            player1_time_ms = ?, turn_started_at = ?, updated_at = ?
        WHERE id = ?
      `).bind(
        JSON.stringify([botMove]),
        now,
        newPlayer1Time,
        now,
        now,
        gameId
      ).run()

      return jsonResponse({
        gameId,
        playerNumber: 2,
        difficulty: effectiveDifficulty,
        personaId: botPersonaId,
        botRating,
        botMovedFirst: true,
        botMove,
      })
    }

    return jsonResponse({
      gameId,
      playerNumber: 1,
      difficulty: effectiveDifficulty,
      personaId: botPersonaId,
      botRating,
      botMovedFirst: false,
    })
  } catch (error) {
    console.error('POST /api/bot/game error:', error)
    return errorResponse('Internal server error', 500)
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
