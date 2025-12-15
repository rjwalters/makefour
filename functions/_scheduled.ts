/**
 * Scheduled handler for bot vs bot games
 *
 * This function is triggered by a cron schedule (every minute) to:
 * 1. Advance active bot vs bot games that are ready for their next move
 * 2. Optionally create new bot vs bot games if below threshold
 *
 * Bot games have a move_delay_ms that controls pacing, so even with
 * 1-minute cron intervals, the games advance at a watchable pace.
 */

import {
  replayMoves,
  makeMove,
  createGameState,
  type Player,
} from './lib/game'
import { calculateNewRating, type GameOutcome } from './lib/elo'
import {
  suggestMoveWithEngine,
  calculateTimeBudget,
  type DifficultyLevel,
  type BotPersonaConfig,
  type AIConfig,
} from './lib/bot'
import type { EngineType } from './lib/ai-engine'
import {
  getRandomReaction,
  shouldBotSpeak,
  type ChatPersonality,
  type ReactionType,
} from './lib/botPersonas'

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
  winner: string | null
  player1_rating: number
  player2_rating: number
  player1_time_ms: number | null
  player2_time_ms: number | null
  turn_started_at: number | null
  bot1_persona_id: string | null
  bot2_persona_id: string | null
  move_delay_ms: number | null
  next_move_at: number | null
}

interface BotPersonaRow {
  id: string
  name: string
  current_elo: number
  ai_engine: string
  ai_config: string
  chat_personality: string
}

// Maximum games to process per scheduled run
const MAX_GAMES_PER_RUN = 20

// Target number of active bot games
const TARGET_ACTIVE_GAMES = 3

/**
 * Scheduled event handler
 */
export async function scheduled(
  event: ScheduledEvent,
  env: Env,
  ctx: ExecutionContext
): Promise<void> {
  const { DB } = env
  const now = Date.now()

  console.log(`[Scheduled] Bot game tick starting at ${new Date(now).toISOString()}`)

  try {
    // Find active bot vs bot games that are ready for next move
    const readyGames = await DB.prepare(`
      SELECT
        id, player1_id, player2_id, moves, current_turn,
        status, winner, player1_rating, player2_rating,
        player1_time_ms, player2_time_ms, turn_started_at,
        bot1_persona_id, bot2_persona_id,
        move_delay_ms, next_move_at
      FROM active_games
      WHERE is_bot_vs_bot = 1
        AND status = 'active'
        AND (next_move_at IS NULL OR next_move_at <= ?)
      ORDER BY next_move_at ASC
      LIMIT ?
    `)
      .bind(now, MAX_GAMES_PER_RUN)
      .all<ActiveGameRow>()

    let processedCount = 0
    let completedCount = 0
    let errorCount = 0

    // Process each ready game
    for (const game of readyGames.results) {
      try {
        const completed = await advanceGame(DB, game, now)
        processedCount++
        if (completed) completedCount++
      } catch (error) {
        console.error(`[Scheduled] Error advancing game ${game.id}:`, error)
        errorCount++
      }
    }

    // Check if we need to create new games
    const activeCount = await DB.prepare(`
      SELECT COUNT(*) as count
      FROM active_games
      WHERE is_bot_vs_bot = 1 AND status = 'active'
    `).first<{ count: number }>()

    const currentActive = activeCount?.count ?? 0

    if (currentActive < TARGET_ACTIVE_GAMES) {
      const gamesToCreate = Math.min(TARGET_ACTIVE_GAMES - currentActive, 2)
      const createdIds = await createRandomBotGames(DB, gamesToCreate, now)
      console.log(`[Scheduled] Created ${createdIds.length} new bot games`)
    }

    console.log(`[Scheduled] Processed ${processedCount} games, ${completedCount} completed, ${errorCount} errors`)
  } catch (error) {
    console.error('[Scheduled] Bot game tick error:', error)
  }
}

/**
 * Advance a single bot vs bot game
 */
async function advanceGame(
  DB: D1Database,
  game: ActiveGameRow,
  now: number
): Promise<boolean> {
  // Get the current bot's persona
  const currentBotPersonaId = game.current_turn === 1
    ? game.bot1_persona_id
    : game.bot2_persona_id

  if (!currentBotPersonaId) {
    throw new Error('Missing bot persona configuration')
  }

  const persona = await DB.prepare(`
    SELECT id, name, current_elo, ai_engine, ai_config, chat_personality
    FROM bot_personas
    WHERE id = ?
  `)
    .bind(currentBotPersonaId)
    .first<BotPersonaRow>()

  if (!persona) {
    throw new Error('Bot persona not found')
  }

  const aiConfig = JSON.parse(persona.ai_config) as AIConfig
  const chatPersonality = parseChatPersonality(persona.chat_personality)

  // Reconstruct the current board state
  const moves = JSON.parse(game.moves) as number[]
  const currentState = moves.length > 0 ? replayMoves(moves) : createGameState()

  if (!currentState) {
    throw new Error('Invalid game state')
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
    throw new Error('Invalid move generated by bot')
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
    game.id
  ).run()

  // Generate bot chat message if appropriate
  if (chatPersonality && shouldBotSpeak(chatPersonality.chattiness)) {
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
      const evalScore = moveResult.score ?? 0
      if (evalScore > 500) {
        reactionType = 'botWinning'
      } else if (evalScore < -500) {
        reactionType = 'botLosing'
      }
    }

    const chatMessage = getRandomReaction(chatPersonality, reactionType)

    if (chatMessage) {
      const messageId = crypto.randomUUID()
      const botUserId = game.current_turn === 1 ? game.player1_id : game.player2_id

      await DB.prepare(`
        INSERT INTO game_messages (id, game_id, sender_id, sender_type, content, created_at)
        VALUES (?, ?, ?, 'bot', ?, ?)
      `).bind(messageId, game.id, botUserId, chatMessage, now).run()
    }
  }

  // Update ratings if game is complete
  if (newStatus === 'completed') {
    await updateBotRatings(DB, game, winner, now)
    return true
  }

  return false
}

/**
 * Parse chat personality, handling both old and new formats
 */
function parseChatPersonality(json: string): ChatPersonality | null {
  try {
    const parsed = JSON.parse(json)
    if (parsed.reactions && typeof parsed.chattiness === 'number') {
      return parsed as ChatPersonality
    }
    return null
  } catch {
    return null
  }
}

/**
 * Create random bot vs bot games for entertainment
 */
async function createRandomBotGames(
  DB: D1Database,
  count: number,
  now: number
): Promise<string[]> {
  const personas = await DB.prepare(`
    SELECT id, name, current_elo
    FROM bot_personas
    WHERE is_active = 1
    ORDER BY RANDOM()
    LIMIT ?
  `).bind(count * 2 + 4).all<{ id: string; name: string; current_elo: number }>()

  if (personas.results.length < 2) {
    return []
  }

  const createdGameIds: string[] = []
  const usedPersonas = new Set<string>()

  for (let i = 0; i < count && usedPersonas.size < personas.results.length - 1; i++) {
    const availablePersonas = personas.results.filter(p => !usedPersonas.has(p.id))
    if (availablePersonas.length < 2) break

    availablePersonas.sort((a, b) => a.current_elo - b.current_elo)
    const idx = Math.floor(Math.random() * (availablePersonas.length - 1))
    const bot1 = availablePersonas[idx]
    const bot2 = availablePersonas[idx + 1]

    usedPersonas.add(bot1.id)
    usedPersonas.add(bot2.id)

    const bot1UserId = `bot_${bot1.id}`
    const bot2UserId = `bot_${bot2.id}`

    const [bot1User, bot2User] = await Promise.all([
      DB.prepare('SELECT rating FROM users WHERE id = ? AND is_bot = 1')
        .bind(bot1UserId)
        .first<{ rating: number }>(),
      DB.prepare('SELECT rating FROM users WHERE id = ? AND is_bot = 1')
        .bind(bot2UserId)
        .first<{ rating: number }>(),
    ])

    const bot1Rating = bot1User?.rating ?? bot1.current_elo
    const bot2Rating = bot2User?.rating ?? bot2.current_elo

    const gameId = crypto.randomUUID()
    const moveDelayMs = 2000 + Math.floor(Math.random() * 1000)

    await DB.prepare(`
      INSERT INTO active_games (
        id, player1_id, player2_id, moves, current_turn, status, mode,
        player1_rating, player2_rating, spectatable, spectator_count,
        last_move_at, time_control_ms, player1_time_ms, player2_time_ms,
        turn_started_at, is_bot_game, is_bot_vs_bot,
        bot1_persona_id, bot2_persona_id, move_delay_ms, next_move_at,
        created_at, updated_at
      )
      VALUES (?, ?, ?, '[]', 1, 'active', 'ranked', ?, ?, 1, 0, ?, 120000, 120000, 120000, ?, 1, 1, ?, ?, ?, ?, ?, ?)
    `).bind(
      gameId,
      bot1UserId,
      bot2UserId,
      bot1Rating,
      bot2Rating,
      now,
      now,
      bot1.id,
      bot2.id,
      moveDelayMs,
      now + moveDelayMs,
      now,
      now
    ).run()

    createdGameIds.push(gameId)
  }

  return createdGameIds
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

  const [bot1User, bot2User] = await Promise.all([
    DB.prepare('SELECT rating, games_played FROM users WHERE id = ? AND is_bot = 1')
      .bind(game.player1_id)
      .first<{ rating: number; games_played: number }>(),
    DB.prepare('SELECT rating, games_played FROM users WHERE id = ? AND is_bot = 1')
      .bind(game.player2_id)
      .first<{ rating: number; games_played: number }>(),
  ])

  const statements: D1PreparedStatement[] = []

  if (bot1User) {
    const bot1Result = calculateNewRating(
      game.player1_rating,
      game.player2_rating,
      bot1Outcome,
      bot1User.games_played
    )

    const bot1GameId = crypto.randomUUID()
    const bot1RatingHistoryId = crypto.randomUUID()

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
      ),
      DB.prepare(`
        UPDATE users SET
          rating = ?, games_played = games_played + 1,
          wins = wins + ?, losses = losses + ?, draws = draws + ?,
          updated_at = ?
        WHERE id = ? AND is_bot = 1
      `).bind(
        bot1Result.newRating,
        bot1Outcome === 'win' ? 1 : 0,
        bot1Outcome === 'loss' ? 1 : 0,
        bot1Outcome === 'draw' ? 1 : 0,
        now,
        game.player1_id
      ),
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

    if (game.bot1_persona_id) {
      statements.push(
        DB.prepare(`
          UPDATE bot_personas SET
            current_elo = ?, games_played = games_played + 1,
            wins = wins + ?, losses = losses + ?, draws = draws + ?,
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

  if (bot2User) {
    const bot2Result = calculateNewRating(
      game.player2_rating,
      game.player1_rating,
      bot2Outcome,
      bot2User.games_played
    )

    const bot2GameId = crypto.randomUUID()
    const bot2RatingHistoryId = crypto.randomUUID()

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
      ),
      DB.prepare(`
        UPDATE users SET
          rating = ?, games_played = games_played + 1,
          wins = wins + ?, losses = losses + ?, draws = draws + ?,
          updated_at = ?
        WHERE id = ? AND is_bot = 1
      `).bind(
        bot2Result.newRating,
        bot2Outcome === 'win' ? 1 : 0,
        bot2Outcome === 'loss' ? 1 : 0,
        bot2Outcome === 'draw' ? 1 : 0,
        now,
        game.player2_id
      ),
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

    if (game.bot2_persona_id) {
      statements.push(
        DB.prepare(`
          UPDATE bot_personas SET
            current_elo = ?, games_played = games_played + 1,
            wins = wins + ?, losses = losses + ?, draws = draws + ?,
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

export default {
  scheduled,
}
