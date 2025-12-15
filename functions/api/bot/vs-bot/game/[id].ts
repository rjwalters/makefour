/**
 * Bot vs Bot Game API endpoint for specific games
 *
 * GET /api/bot/vs-bot/game/:id - Get game state (for spectating)
 * POST /api/bot/vs-bot/game/:id - Advance the game (make next bot move)
 */

import { jsonResponse } from '../../../../lib/auth'
import {
  replayMoves,
  makeMove,
  createGameState,
  type Board,
  type Player,
} from '../../../../lib/game'
import { calculateNewRating, type GameOutcome } from '../../../../lib/elo'
import {
  suggestMoveWithEngine,
  calculateTimeBudget,
  type DifficultyLevel,
  type BotPersonaConfig,
  type AIConfig,
} from '../../../../lib/bot'
import type { EngineType } from '../../../../lib/ai-engine'
import {
  getRandomReaction,
  shouldBotSpeak,
  type ChatPersonality,
  type ReactionType,
} from '../../../../lib/botPersonas'

interface Env {
  DB: D1Database
}

interface ActiveGameRow {
  id: string
  player1_id: string
  player2_id: string
  moves: string
  current_turn: number
  status: string
  mode: string
  winner: string | null
  player1_rating: number
  player2_rating: number
  last_move_at: number
  time_control_ms: number | null
  player1_time_ms: number | null
  player2_time_ms: number | null
  turn_started_at: number | null
  is_bot_game: number
  is_bot_vs_bot: number
  bot1_persona_id: string | null
  bot2_persona_id: string | null
  move_delay_ms: number | null
  next_move_at: number | null
  spectator_count: number
  created_at: number
  updated_at: number
}

interface BotPersonaRow {
  id: string
  name: string
  current_elo: number
  ai_engine: string
  ai_config: string
  chat_personality: string
}

/**
 * GET /api/bot/vs-bot/game/:id - Get game state for spectating
 */
export async function onRequestGet(context: EventContext<Env, any, { id: string }>) {
  const { DB } = context.env
  const gameId = context.params.id

  try {
    const game = await DB.prepare(`
      SELECT
        ag.id, ag.player1_id, ag.player2_id, ag.moves, ag.current_turn,
        ag.status, ag.mode, ag.winner, ag.player1_rating, ag.player2_rating,
        ag.spectator_count, ag.move_delay_ms, ag.next_move_at,
        ag.time_control_ms, ag.player1_time_ms, ag.player2_time_ms,
        ag.last_move_at, ag.created_at, ag.updated_at,
        ag.bot1_persona_id, ag.bot2_persona_id,
        bp1.name as bot1_name,
        bp2.name as bot2_name
      FROM active_games ag
      LEFT JOIN bot_personas bp1 ON ag.bot1_persona_id = bp1.id
      LEFT JOIN bot_personas bp2 ON ag.bot2_persona_id = bp2.id
      WHERE ag.id = ? AND ag.is_bot_vs_bot = 1
    `)
      .bind(gameId)
      .first<ActiveGameRow & { bot1_name: string | null; bot2_name: string | null }>()

    if (!game) {
      return jsonResponse({ error: 'Game not found' }, { status: 404 })
    }

    const moves = JSON.parse(game.moves) as number[]
    const gameState = moves.length > 0 ? replayMoves(moves) : createGameState()

    return jsonResponse({
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
      currentTurn: game.current_turn,
      moves,
      board: gameState?.board ?? null,
      status: game.status,
      winner: game.winner,
      spectatorCount: game.spectator_count,
      moveDelayMs: game.move_delay_ms,
      nextMoveAt: game.next_move_at,
      timeControlMs: game.time_control_ms,
      player1TimeMs: game.player1_time_ms,
      player2TimeMs: game.player2_time_ms,
      lastMoveAt: game.last_move_at,
      createdAt: game.created_at,
      updatedAt: game.updated_at,
    })
  } catch (error) {
    console.error('GET /api/bot/vs-bot/game/:id error:', error)
    return jsonResponse({ error: 'Internal server error' }, { status: 500 })
  }
}

/**
 * POST /api/bot/vs-bot/game/:id - Advance the game (make next bot move)
 *
 * This endpoint is called to make the next move in a bot vs bot game.
 * It can be triggered by a scheduled worker or manually.
 *
 * Returns the new game state after the move is made.
 */
export async function onRequestPost(context: EventContext<Env, any, { id: string }>) {
  const { DB } = context.env
  const gameId = context.params.id

  try {
    // Get the game
    const game = await DB.prepare(`
      SELECT
        ag.id, ag.player1_id, ag.player2_id, ag.moves, ag.current_turn,
        ag.status, ag.mode, ag.winner, ag.player1_rating, ag.player2_rating,
        ag.spectator_count, ag.move_delay_ms, ag.next_move_at,
        ag.time_control_ms, ag.player1_time_ms, ag.player2_time_ms,
        ag.turn_started_at, ag.last_move_at, ag.created_at, ag.updated_at,
        ag.is_bot_game, ag.is_bot_vs_bot,
        ag.bot1_persona_id, ag.bot2_persona_id
      FROM active_games ag
      WHERE ag.id = ? AND ag.is_bot_vs_bot = 1
    `)
      .bind(gameId)
      .first<ActiveGameRow>()

    if (!game) {
      return jsonResponse({ error: 'Game not found' }, { status: 404 })
    }

    if (game.status !== 'active') {
      return jsonResponse({ error: 'Game is not active', status: game.status }, { status: 400 })
    }

    const now = Date.now()

    // Check if it's time for the next move (with some tolerance)
    if (game.next_move_at && now < game.next_move_at - 100) {
      return jsonResponse({
        error: 'Too early for next move',
        nextMoveAt: game.next_move_at,
        waitMs: game.next_move_at - now,
      }, { status: 425 }) // 425 Too Early
    }

    // Get the current bot's persona
    const currentBotPersonaId = game.current_turn === 1
      ? game.bot1_persona_id
      : game.bot2_persona_id

    if (!currentBotPersonaId) {
      return jsonResponse({ error: 'Missing bot persona configuration' }, { status: 500 })
    }

    const persona = await DB.prepare(`
      SELECT id, name, current_elo, ai_engine, ai_config, chat_personality
      FROM bot_personas
      WHERE id = ?
    `)
      .bind(currentBotPersonaId)
      .first<BotPersonaRow>()

    if (!persona) {
      return jsonResponse({ error: 'Bot persona not found' }, { status: 500 })
    }

    const aiConfig = JSON.parse(persona.ai_config) as AIConfig
    const chatPersonality = JSON.parse(persona.chat_personality) as ChatPersonality

    // Reconstruct the current board state
    const moves = JSON.parse(game.moves) as number[]
    const currentState = moves.length > 0 ? replayMoves(moves) : createGameState()

    if (!currentState) {
      return jsonResponse({ error: 'Invalid game state' }, { status: 500 })
    }

    // Calculate time budget for the bot
    const botTimeRemaining = game.current_turn === 1 ? game.player1_time_ms : game.player2_time_ms
    let player1Time = game.player1_time_ms
    let player2Time = game.player2_time_ms

    // Map persona rating to difficulty level
    const difficulty: DifficultyLevel =
      persona.current_elo < 900 ? 'beginner' :
      persona.current_elo < 1300 ? 'intermediate' :
      persona.current_elo < 1700 ? 'expert' : 'perfect'

    const timeBudget = calculateTimeBudget(botTimeRemaining ?? 60000, moves.length, difficulty)
    const startTime = Date.now()

    // Get bot's move using the engine
    const botPersonaConfig: BotPersonaConfig = {
      difficulty,
      engine: (persona.ai_engine || 'minimax') as EngineType,
    }

    const moveResult = await suggestMoveWithEngine(
      currentState.board,
      game.current_turn as Player,
      botPersonaConfig,
      timeBudget
    )

    const elapsedMs = Date.now() - startTime

    // Deduct time from current player
    if (game.current_turn === 1 && player1Time !== null) {
      player1Time = Math.max(0, player1Time - elapsedMs)
    } else if (game.current_turn === 2 && player2Time !== null) {
      player2Time = Math.max(0, player2Time - elapsedMs)
    }

    // Apply the move
    const afterMove = makeMove(currentState, moveResult.column)
    if (!afterMove) {
      return jsonResponse({ error: 'Invalid move generated by bot' }, { status: 500 })
    }

    const newMoves = [...moves, moveResult.column]
    let newStatus = 'active'
    let winner: string | null = null
    let nextTurn = game.current_turn === 1 ? 2 : 1

    // Check for game over
    if (afterMove.winner !== null) {
      newStatus = 'completed'
      winner = afterMove.winner === 'draw' ? 'draw' : String(afterMove.winner)
    }

    // Calculate next move time
    const nextMoveAt = newStatus === 'active' ? now + (game.move_delay_ms ?? 2000) : null

    // Update the game
    await DB.prepare(`
      UPDATE active_games
      SET moves = ?, current_turn = ?, status = ?, winner = ?,
          last_move_at = ?, updated_at = ?,
          player1_time_ms = ?, player2_time_ms = ?,
          turn_started_at = ?, next_move_at = ?
      WHERE id = ?
    `).bind(
      JSON.stringify(newMoves),
      nextTurn,
      newStatus,
      winner,
      now,
      now,
      player1Time,
      player2Time,
      newStatus === 'active' ? now : game.turn_started_at,
      nextMoveAt,
      gameId
    ).run()

    // Generate bot chat message if appropriate
    let chatMessage: string | null = null
    if (shouldBotSpeak(chatPersonality.chattiness)) {
      let reactionType: ReactionType = 'gameStart'

      if (newMoves.length === 1) {
        reactionType = 'gameStart'
      } else if (newStatus === 'completed') {
        if (winner === 'draw') {
          reactionType = 'draw'
        } else if (winner === String(game.current_turn)) {
          reactionType = 'gameWon'
        } else {
          reactionType = 'gameLost'
        }
      } else {
        // Check if we're winning or losing based on move evaluation
        const evalScore = moveResult.score ?? 0
        if (evalScore > 500) {
          reactionType = 'botWinning'
        } else if (evalScore < -500) {
          reactionType = 'botLosing'
        }
      }

      chatMessage = getRandomReaction(chatPersonality, reactionType)

      // Insert chat message if we have one
      if (chatMessage) {
        const messageId = crypto.randomUUID()
        const botUserId = game.current_turn === 1 ? game.player1_id : game.player2_id

        await DB.prepare(`
          INSERT INTO game_messages (id, game_id, sender_id, sender_type, content, created_at)
          VALUES (?, ?, ?, 'bot', ?, ?)
        `).bind(messageId, gameId, botUserId, chatMessage, now).run()
      }
    }

    // Update ratings if game is complete
    if (newStatus === 'completed') {
      await updateBotRatings(DB, game, winner, now)
    }

    return jsonResponse({
      success: true,
      move: moveResult.column,
      moves: newMoves,
      board: afterMove.board,
      currentTurn: nextTurn,
      status: newStatus,
      winner,
      nextMoveAt,
      player1TimeMs: player1Time,
      player2TimeMs: player2Time,
      chatMessage,
      searchInfo: moveResult.searchInfo,
    })
  } catch (error) {
    console.error('POST /api/bot/vs-bot/game/:id error:', error)
    return jsonResponse({ error: 'Internal server error' }, { status: 500 })
  }
}

/**
 * Update both bots' ratings after a bot vs bot game
 */
async function updateBotRatings(
  DB: D1Database,
  game: ActiveGameRow,
  winner: string | null,
  now: number
) {
  const moves = JSON.parse(game.moves) as number[]

  // Determine outcomes for each bot
  let bot1Outcome: GameOutcome
  let bot2Outcome: GameOutcome

  if (winner === 'draw') {
    bot1Outcome = 'draw'
    bot2Outcome = 'draw'
  } else if (winner === '1') {
    bot1Outcome = 'win'
    bot2Outcome = 'loss'
  } else {
    bot1Outcome = 'loss'
    bot2Outcome = 'win'
  }

  // Get bot users' current stats
  const [bot1User, bot2User] = await Promise.all([
    DB.prepare('SELECT rating, games_played FROM users WHERE id = ? AND is_bot = 1')
      .bind(game.player1_id)
      .first<{ rating: number; games_played: number }>(),
    DB.prepare('SELECT rating, games_played FROM users WHERE id = ? AND is_bot = 1')
      .bind(game.player2_id)
      .first<{ rating: number; games_played: number }>(),
  ])

  const statements: D1PreparedStatement[] = []

  // Update bot 1
  if (bot1User) {
    const bot1Result = calculateNewRating(
      game.player1_rating,
      game.player2_rating,
      bot1Outcome,
      bot1User.games_played
    )

    const bot1GameId = crypto.randomUUID()
    const bot1RatingHistoryId = crypto.randomUUID()

    // Game record for bot 1
    statements.push(
      DB.prepare(`
        INSERT INTO games (id, user_id, outcome, moves, move_count, rating_change,
                          opponent_type, player_number, created_at)
        VALUES (?, ?, ?, ?, ?, ?, 'ai', 1, ?)
      `).bind(
        bot1GameId,
        game.player1_id,
        bot1Outcome,
        JSON.stringify(moves),
        moves.length,
        bot1Result.ratingChange,
        now
      )
    )

    // Update bot 1 user stats
    statements.push(
      DB.prepare(`
        UPDATE users SET
          rating = ?,
          games_played = games_played + 1,
          wins = wins + ?,
          losses = losses + ?,
          draws = draws + ?,
          updated_at = ?
        WHERE id = ? AND is_bot = 1
      `).bind(
        bot1Result.newRating,
        bot1Outcome === 'win' ? 1 : 0,
        bot1Outcome === 'loss' ? 1 : 0,
        bot1Outcome === 'draw' ? 1 : 0,
        now,
        game.player1_id
      )
    )

    // Rating history for bot 1
    statements.push(
      DB.prepare(`
        INSERT INTO rating_history (id, user_id, game_id, rating_before, rating_after, rating_change, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
      `).bind(
        bot1RatingHistoryId,
        game.player1_id,
        bot1GameId,
        game.player1_rating,
        bot1Result.newRating,
        bot1Result.ratingChange,
        now
      )
    )

    // Update bot_personas table for bot 1
    if (game.bot1_persona_id) {
      statements.push(
        DB.prepare(`
          UPDATE bot_personas SET
            current_elo = ?,
            games_played = games_played + 1,
            wins = wins + ?,
            losses = losses + ?,
            draws = draws + ?,
            updated_at = ?
          WHERE id = ?
        `).bind(
          bot1Result.newRating,
          bot1Outcome === 'win' ? 1 : 0,
          bot1Outcome === 'loss' ? 1 : 0,
          bot1Outcome === 'draw' ? 1 : 0,
          now,
          game.bot1_persona_id
        )
      )
    }
  }

  // Update bot 2
  if (bot2User) {
    const bot2Result = calculateNewRating(
      game.player2_rating,
      game.player1_rating,
      bot2Outcome,
      bot2User.games_played
    )

    const bot2GameId = crypto.randomUUID()
    const bot2RatingHistoryId = crypto.randomUUID()

    // Game record for bot 2
    statements.push(
      DB.prepare(`
        INSERT INTO games (id, user_id, outcome, moves, move_count, rating_change,
                          opponent_type, player_number, created_at)
        VALUES (?, ?, ?, ?, ?, ?, 'ai', 2, ?)
      `).bind(
        bot2GameId,
        game.player2_id,
        bot2Outcome,
        JSON.stringify(moves),
        moves.length,
        bot2Result.ratingChange,
        now
      )
    )

    // Update bot 2 user stats
    statements.push(
      DB.prepare(`
        UPDATE users SET
          rating = ?,
          games_played = games_played + 1,
          wins = wins + ?,
          losses = losses + ?,
          draws = draws + ?,
          updated_at = ?
        WHERE id = ? AND is_bot = 1
      `).bind(
        bot2Result.newRating,
        bot2Outcome === 'win' ? 1 : 0,
        bot2Outcome === 'loss' ? 1 : 0,
        bot2Outcome === 'draw' ? 1 : 0,
        now,
        game.player2_id
      )
    )

    // Rating history for bot 2
    statements.push(
      DB.prepare(`
        INSERT INTO rating_history (id, user_id, game_id, rating_before, rating_after, rating_change, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
      `).bind(
        bot2RatingHistoryId,
        game.player2_id,
        bot2GameId,
        game.player2_rating,
        bot2Result.newRating,
        bot2Result.ratingChange,
        now
      )
    )

    // Update bot_personas table for bot 2
    if (game.bot2_persona_id) {
      statements.push(
        DB.prepare(`
          UPDATE bot_personas SET
            current_elo = ?,
            games_played = games_played + 1,
            wins = wins + ?,
            losses = losses + ?,
            draws = draws + ?,
            updated_at = ?
          WHERE id = ?
        `).bind(
          bot2Result.newRating,
          bot2Outcome === 'win' ? 1 : 0,
          bot2Outcome === 'loss' ? 1 : 0,
          bot2Outcome === 'draw' ? 1 : 0,
          now,
          game.bot2_persona_id
        )
      )
    }
  }

  if (statements.length > 0) {
    await DB.batch(statements)
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
