/**
 * POST /api/matchmaking/play-bot - Skip human matchmaking and play a bot immediately
 *
 * Finds an appropriate bot persona based on the user's rating and creates a ranked game.
 */

import { validateSession, errorResponse, jsonResponse } from '../../lib/auth'
import { createDb } from '../../../shared/db/client'
import { users, matchmakingQueue, activeGames, botPersonas } from '../../../shared/db/schema'
import { eq, and, or, sql } from 'drizzle-orm'
import { createGameState, makeMove } from '../../lib/game'
import { suggestMoveWithEngine, type BotPersonaConfig, type AIConfig } from '../../lib/bot'
import { DEFAULT_TIME_CONTROL_MS } from '../../lib/types'

interface Env {
  DB: D1Database
}

export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    const db = createDb(DB)

    // First check if user is already in an active game
    const activeGame = await db.query.activeGames.findFirst({
      where: and(
        or(
          eq(activeGames.player1Id, session.userId),
          eq(activeGames.player2Id, session.userId)
        ),
        eq(activeGames.status, 'active')
      ),
      columns: { id: true, player1Id: true, player2Id: true }
    })

    if (activeGame) {
      // User already has an active game - return it instead of creating new one
      return jsonResponse({
        status: 'matched',
        gameId: activeGame.id,
        playerNumber: activeGame.player1Id === session.userId ? 1 : 2,
      })
    }

    // Check if user is in queue
    const queueEntry = await db.query.matchmakingQueue.findFirst({
      where: eq(matchmakingQueue.userId, session.userId),
      columns: { id: true, rating: true, mode: true }
    })

    // Get user's current rating
    const user = await db.query.users.findFirst({
      where: eq(users.id, session.userId),
      columns: { id: true, rating: true }
    })

    if (!user) {
      return errorResponse('User not found', 404)
    }

    const userRating = queueEntry?.rating ?? user.rating

    // Find an appropriate bot persona near the user's rating
    // Get the bot with rating closest to user, preferring slightly lower rated
    const botPersona = await db
      .select()
      .from(botPersonas)
      .where(eq(botPersonas.isActive, 1))
      .orderBy(sql`ABS(${botPersonas.currentElo} - ${userRating}) ASC`)
      .limit(1)
      .then(rows => rows[0])

    if (!botPersona) {
      console.error('No active bot personas found in database')
      return errorResponse('No bots available', 503)
    }

    // Get the bot user for this persona
    const botUser = await db.query.users.findFirst({
      where: and(
        eq(users.isBot, 1),
        eq(users.botPersonaId, botPersona.id)
      ),
      columns: { id: true, botPersonaId: true }
    })

    if (!botUser) {
      console.error(`No bot user found for persona ${botPersona.id} (${botPersona.name})`)
      return errorResponse('Bot user not found', 503)
    }

    // Create the game
    const gameId = crypto.randomUUID()
    const now = Date.now()

    // Randomly assign player 1 and player 2
    const userIsPlayer1 = Math.random() < 0.5
    const player1Id = userIsPlayer1 ? session.userId : botUser.id
    const player2Id = userIsPlayer1 ? botUser.id : session.userId
    const player1Rating = userIsPlayer1 ? userRating : botPersona.currentElo
    const player2Rating = userIsPlayer1 ? botPersona.currentElo : userRating

    // If bot is player 1, make the bot's first move
    let initialMoves: number[] = []
    let currentTurn = 1
    let player1TimeMs = DEFAULT_TIME_CONTROL_MS

    if (!userIsPlayer1) {
      // Bot is player 1, make its first move
      const gameState = createGameState()
      const botConfig: BotPersonaConfig = {
        difficulty: 'intermediate', // Will use persona config
        engine: 'minimax',
      }

      // Parse persona AI config if available
      let aiConfig: AIConfig | undefined
      try {
        aiConfig = JSON.parse(botPersona.aiConfig)
      } catch {
        // Use defaults
      }

      const moveResult = await suggestMoveWithEngine(
        gameState.board,
        1, // Bot is player 1
        botConfig,
        5000 // 5 second time budget for first move
      )

      const afterBotMove = makeMove(gameState, moveResult.column)
      if (afterBotMove) {
        initialMoves = [moveResult.column]
        currentTurn = 2 // Now human's turn
        // Deduct a small amount of time for the bot's move (simulated)
        player1TimeMs = DEFAULT_TIME_CONTROL_MS - 1000
      }
    }

    try {
      // Create game and remove from queue in a batch
      const statements = [
        db.insert(activeGames).values({
          id: gameId,
          player1Id,
          player2Id,
          moves: JSON.stringify(initialMoves),
          currentTurn,
          status: 'active',
          mode: 'ranked',
          player1Rating,
          player2Rating,
          spectatable: 1,
          spectatorCount: 0,
          lastMoveAt: now,
          timeControlMs: DEFAULT_TIME_CONTROL_MS,
          player1TimeMs,
          player2TimeMs: DEFAULT_TIME_CONTROL_MS,
          turnStartedAt: now,
          isBotGame: 1,
          botPersonaId: botPersona.id,
          createdAt: now,
          updatedAt: now,
        }),
      ]

      // Remove from queue if they were in it
      if (queueEntry) {
        statements.push(
          db.delete(matchmakingQueue).where(eq(matchmakingQueue.userId, session.userId))
        )
      }

      await db.batch(statements as any)

      return jsonResponse({
        status: 'matched',
        gameId,
        playerNumber: userIsPlayer1 ? 1 : 2,
        opponent: {
          name: botPersona.name,
          rating: botPersona.currentElo,
          isBot: true,
        },
        mode: 'ranked',
      })
    } catch (dbError) {
      const errorMessage = dbError instanceof Error ? dbError.message : String(dbError)
      console.error('Bot game creation failed:', {
        error: errorMessage,
        stack: dbError instanceof Error ? dbError.stack : undefined,
        gameId,
        player1Id,
        player2Id,
        botPersonaId: botPersona.id,
        botUserId: botUser.id,
      })
      return errorResponse(`Failed to create game: ${errorMessage}`, 500)
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
