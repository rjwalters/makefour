/**
 * Bot vs Bot Game Generation API endpoint
 *
 * POST /api/bot/vs-bot/generate - Generate a complete bot vs bot game on demand
 *
 * This endpoint instantly plays a full game between two random bots and returns
 * the complete move sequence for client-side playback. The game result is stored
 * and ELO ratings are updated immediately.
 *
 * No authentication required - this is a public endpoint for spectators.
 */

import { jsonResponse } from '../../../lib/auth'
import { generateBotGame } from '../../../lib/botVsBotGame'

interface Env {
  DB: D1Database
}

/**
 * POST /api/bot/vs-bot/generate - Generate a new bot vs bot game
 *
 * Returns the complete game data including:
 * - Full move sequence
 * - Winner
 * - Bot info (names, ratings, engines)
 * - Playback timing
 */
export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  if (!DB) {
    console.error('POST /api/bot/vs-bot/generate error: Database not configured')
    return jsonResponse(
      { error: 'Database not configured' },
      { status: 503 }
    )
  }

  try {
    const game = await generateBotGame(DB)

    return jsonResponse({
      success: true,
      game,
    })
  } catch (error) {
    console.error('POST /api/bot/vs-bot/generate error:', error)

    if (error instanceof Error) {
      if (error.message.includes('Not enough active bot personas')) {
        return jsonResponse(
          { error: 'No bots available for matchmaking. The database may not have any active bot personas configured.' },
          { status: 503 }
        )
      }
      // Return the actual error message for debugging
      return jsonResponse(
        { error: `Failed to generate game: ${error.message}` },
        { status: 500 }
      )
    }

    return jsonResponse(
      { error: 'Failed to generate game' },
      { status: 500 }
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
