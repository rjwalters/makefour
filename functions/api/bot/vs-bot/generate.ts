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

    const errorMessage = error instanceof Error ? error.message : String(error)
    const errorStack = error instanceof Error ? error.stack : undefined

    // Log detailed error info
    console.error('Error details:', {
      message: errorMessage,
      stack: errorStack,
      type: error?.constructor?.name,
    })

    if (errorMessage.includes('Not enough active bot personas')) {
      return jsonResponse(
        { error: 'No bots available. Please ensure bot personas are configured in the database.' },
        { status: 503 }
      )
    }

    // Return the actual error message for debugging (in production, consider sanitizing)
    return jsonResponse(
      { error: `Game generation failed: ${errorMessage}` },
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
