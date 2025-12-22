/**
 * Bot Game API endpoint
 *
 * POST /api/bot/game - Create a new ranked bot game
 */

import { validateSession, errorResponse, jsonResponse } from '../../lib/auth'
import { z } from 'zod'
import type { EngineType } from '../../lib/ai-engine'
import { createDb } from '../../../shared/db/client'
import { users, activeGames, botPersonas } from '../../../shared/db/schema'
import { eq, and, or } from 'drizzle-orm'

interface Env {
  DB: D1Database
}

/**
 * Generate bot user ID from persona ID
 * Format: bot_<persona_id>
 */
function getBotUserId(personaId: string): string {
  return `bot_${personaId}`
}

// Legacy bot user ID for backward compatibility (used when personaId not specified)
const LEGACY_BOT_USER_ID = 'bot-opponent'

// Legacy bot ratings by difficulty (for backward compatibility)
const BOT_RATINGS: Record<string, number> = {
  beginner: 800,
  intermediate: 1200,
  expert: 1600,
  perfect: 2000,
}

// Default persona mappings for legacy difficulty levels
const LEGACY_DIFFICULTY_PERSONAS: Record<string, string> = {
  beginner: 'rookie',
  intermediate: 'nova',
  expert: 'scholar',
  perfect: 'oracle',
}

// Default time control: 5 minutes
const DEFAULT_TIME_CONTROL_MS = 300000

// Supported engine types for validation
const ENGINE_TYPES = ['minimax', 'heuristic', 'mcts', 'neural', 'hybrid'] as const

// Schema for creating a game - supports both personaId (new) and difficulty (legacy)
const createGameSchema = z.object({
  personaId: z.string().optional(),
  difficulty: z.enum(['beginner', 'intermediate', 'expert', 'perfect']).optional(),
  playerColor: z.union([z.literal(1), z.literal(2)]).optional().default(1),
  // Optional engine selection - defaults to 'minimax' if not specified
  engine: z.enum(ENGINE_TYPES).optional().default('minimax'),
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

    const { personaId, difficulty, playerColor, engine } = parseResult.data

    // Resolve bot rating and configuration
    let botRating: number
    let botPersonaId: string | null = null
    let botUserId: string
    let effectiveDifficulty: string | null = difficulty || null
    let neuralConfig: { modelId: string; temperature: number } | null = null

    const db = createDb(DB)

    if (personaId) {
      // Look up the persona
      const persona = await db.query.botPersonas.findFirst({
        where: and(eq(botPersonas.id, personaId), eq(botPersonas.isActive, 1)),
        columns: {
          id: true,
          name: true,
          currentElo: true,
          aiConfig: true,
          aiEngine: true,
        },
      })

      if (!persona) {
        return errorResponse('Bot persona not found', 404)
      }

      botPersonaId = persona.id
      botUserId = getBotUserId(persona.id)

      // Check if this is a neural bot and extract neural config for client-side inference
      if (persona.aiEngine === 'neural') {
        try {
          const aiConfig = JSON.parse(persona.aiConfig)
          if (aiConfig.neuralModelId) {
            neuralConfig = {
              modelId: aiConfig.neuralModelId,
              temperature: aiConfig.neuralTemperature ?? 0,
            }
          }
        } catch {
          // Ignore JSON parse errors - will fall back to server-side inference
        }
      }

      // Get the bot's current rating from users table
      const botUser = await db.query.users.findFirst({
        where: and(eq(users.id, botUserId), eq(users.isBot, 1)),
        columns: {
          rating: true,
        },
      })

      // Use bot user's rating if exists, otherwise fall back to persona's current_elo
      botRating = botUser?.rating ?? persona.currentElo

      // Map persona to difficulty based on rating for backward compatibility
      if (botRating < 900) effectiveDifficulty = 'beginner'
      else if (botRating < 1300) effectiveDifficulty = 'intermediate'
      else if (botRating < 1700) effectiveDifficulty = 'expert'
      else effectiveDifficulty = 'perfect'
    } else if (difficulty) {
      // Legacy mode: map difficulty to a default persona
      botPersonaId = LEGACY_DIFFICULTY_PERSONAS[difficulty] || null
      if (botPersonaId) {
        botUserId = getBotUserId(botPersonaId)
        // Try to get rating from bot user, fall back to legacy rating
        const botUser = await db.query.users.findFirst({
          where: and(eq(users.id, botUserId), eq(users.isBot, 1)),
          columns: {
            rating: true,
          },
        })
        botRating = botUser?.rating ?? BOT_RATINGS[difficulty]
      } else {
        // Fallback if no persona mapping exists
        botUserId = LEGACY_BOT_USER_ID
        botRating = BOT_RATINGS[difficulty]
      }
    } else {
      return errorResponse('Either personaId or difficulty is required', 400)
    }

    // Check if user already has an active game
    const existingGame = await db.query.activeGames.findFirst({
      where: and(
        or(eq(activeGames.player1Id, session.userId), eq(activeGames.player2Id, session.userId)),
        eq(activeGames.status, 'active')
      ),
      columns: {
        id: true,
      },
    })

    if (existingGame) {
      return errorResponse('You already have an active game', 400)
    }

    // Get user's current rating
    const user = await db.query.users.findFirst({
      where: eq(users.id, session.userId),
      columns: {
        rating: true,
      },
    })

    if (!user) {
      return errorResponse('User not found', 404)
    }

    const now = Date.now()
    const gameId = crypto.randomUUID()

    // Determine player positions
    // If playerColor is 1, user is player1 (red, goes first)
    // If playerColor is 2, user is player2 (yellow, goes second)
    const player1Id = playerColor === 1 ? session.userId : botUserId
    const player2Id = playerColor === 1 ? botUserId : session.userId
    const player1Rating = playerColor === 1 ? user.rating : botRating
    const player2Rating = playerColor === 1 ? botRating : user.rating
    const botPlayer = playerColor === 1 ? 2 : 1

    // Create the game
    await db.insert(activeGames).values({
      id: gameId,
      player1Id,
      player2Id,
      moves: '[]',
      currentTurn: 1,
      status: 'active',
      mode: 'ranked',
      player1Rating,
      player2Rating,
      spectatable: 0,
      spectatorCount: 0,
      lastMoveAt: now,
      timeControlMs: DEFAULT_TIME_CONTROL_MS,
      player1TimeMs: DEFAULT_TIME_CONTROL_MS,
      player2TimeMs: DEFAULT_TIME_CONTROL_MS,
      turnStartedAt: now,
      isBotGame: 1,
      botDifficulty: effectiveDifficulty,
      bot1PersonaId: botPersonaId,
      createdAt: now,
      updatedAt: now,
    })

    // If bot goes first (player is yellow), make bot's first move
    if (botPlayer === 1) {
      // Use engine-based move suggestion
      const { suggestMoveWithEngine, calculateTimeBudget } = await import('../../lib/bot')

      const timeBudget = calculateTimeBudget(DEFAULT_TIME_CONTROL_MS, 0, (effectiveDifficulty || 'intermediate') as 'beginner' | 'intermediate' | 'expert' | 'perfect')
      const startTime = Date.now()
      const moveResult = await suggestMoveWithEngine(
        Array.from({ length: 6 }, () => Array(7).fill(null)), // Empty board
        1, // Bot is player 1
        { difficulty: (effectiveDifficulty || 'intermediate') as 'beginner' | 'intermediate' | 'expert' | 'perfect', engine: engine as EngineType },
        timeBudget
      )
      const elapsedMs = Date.now() - startTime

      // Update game with bot's first move
      const newPlayer1Time = DEFAULT_TIME_CONTROL_MS - elapsedMs

      await db.update(activeGames)
        .set({
          moves: JSON.stringify([moveResult.column]),
          currentTurn: 2,
          lastMoveAt: now,
          player1TimeMs: newPlayer1Time,
          turnStartedAt: now,
          updatedAt: now,
        })
        .where(eq(activeGames.id, gameId))

      return jsonResponse({
        gameId,
        playerNumber: 2,
        difficulty: effectiveDifficulty,
        personaId: botPersonaId,
        engine,
        botRating,
        botMovedFirst: true,
        botMove: moveResult.column,
        searchInfo: moveResult.searchInfo,
        neuralConfig, // For client-side inference
      })
    }

    return jsonResponse({
      gameId,
      playerNumber: 1,
      difficulty: effectiveDifficulty,
      personaId: botPersonaId,
      engine,
      botRating,
      botMovedFirst: false,
      neuralConfig, // For client-side inference
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
