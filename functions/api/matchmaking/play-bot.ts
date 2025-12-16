/**
 * POST /api/matchmaking/play-bot - Skip human matchmaking and play a bot immediately
 *
 * Finds an appropriate bot persona based on the user's rating and creates a ranked game.
 */

import { validateSession, errorResponse, jsonResponse } from '../../lib/auth'

interface Env {
  DB: D1Database
}

interface QueueEntry {
  id: string
  user_id: string
  rating: number
  mode: string
}

interface UserRow {
  id: string
  rating: number
}

interface BotPersona {
  id: string
  name: string
  current_elo: number
}

interface BotUser {
  id: string
  bot_persona_id: string
}

// Default time control: 5 minutes per player
const DEFAULT_TIME_CONTROL_MS = 300000

export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Check if user is in queue
    const queueEntry = await DB.prepare(`
      SELECT id, user_id, rating, mode
      FROM matchmaking_queue
      WHERE user_id = ?
    `)
      .bind(session.userId)
      .first<QueueEntry>()

    // Get user's current rating
    const user = await DB.prepare(`SELECT id, rating FROM users WHERE id = ?`)
      .bind(session.userId)
      .first<UserRow>()

    if (!user) {
      return errorResponse('User not found', 404)
    }

    const userRating = queueEntry?.rating ?? user.rating

    // Find an appropriate bot persona near the user's rating
    // Get the bot with rating closest to user, preferring slightly lower rated
    const botPersona = await DB.prepare(`
      SELECT id, name, current_elo
      FROM bot_personas
      WHERE is_active = 1
      ORDER BY ABS(current_elo - ?) ASC
      LIMIT 1
    `)
      .bind(userRating)
      .first<BotPersona>()

    if (!botPersona) {
      return errorResponse('No bots available', 503)
    }

    // Get the bot user for this persona
    const botUser = await DB.prepare(`
      SELECT id, bot_persona_id
      FROM users
      WHERE is_bot = 1 AND bot_persona_id = ?
    `)
      .bind(botPersona.id)
      .first<BotUser>()

    if (!botUser) {
      return errorResponse('Bot user not found', 503)
    }

    // Create the game
    const gameId = crypto.randomUUID()
    const now = Date.now()

    // Randomly assign player 1 and player 2
    const userIsPlayer1 = Math.random() < 0.5
    const player1Id = userIsPlayer1 ? session.userId : botUser.id
    const player2Id = userIsPlayer1 ? botUser.id : session.userId
    const player1Rating = userIsPlayer1 ? userRating : botPersona.current_elo
    const player2Rating = userIsPlayer1 ? botPersona.current_elo : userRating

    try {
      // Create game and remove from queue in a batch
      const statements = [
        DB.prepare(`
          INSERT INTO active_games (
            id, player1_id, player2_id, moves, current_turn, status, mode,
            player1_rating, player2_rating, spectatable, spectator_count,
            last_move_at, time_control_ms, player1_time_ms, player2_time_ms,
            turn_started_at, is_bot_game, bot_difficulty, created_at, updated_at
          )
          VALUES (?, ?, ?, '[]', 1, 'active', 'ranked', ?, ?, 1, 0, ?, ?, ?, ?, ?, 1, 'persona', ?, ?)
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
          now,
          now,
          now
        ),
      ]

      // Remove from queue if they were in it
      if (queueEntry) {
        statements.push(
          DB.prepare(`DELETE FROM matchmaking_queue WHERE user_id = ?`).bind(session.userId)
        )
      }

      await DB.batch(statements)

      return jsonResponse({
        status: 'matched',
        gameId,
        playerNumber: userIsPlayer1 ? 1 : 2,
        opponent: {
          name: botPersona.name,
          rating: botPersona.current_elo,
          isBot: true,
        },
        mode: 'ranked',
      })
    } catch (dbError) {
      console.error('Bot game creation failed:', dbError)
      return errorResponse('Failed to create game', 500)
    }
  } catch (error) {
    console.error('POST /api/matchmaking/play-bot error:', error)
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
