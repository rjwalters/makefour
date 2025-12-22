/**
 * Bot Reaction API endpoint for proactive game-aware responses
 *
 * POST /api/match/:id/bot-reaction - Request a bot reaction after player's move
 *
 * This endpoint analyzes the current game position and may generate
 * a proactive comment from the bot about notable moves.
 * Uses persona chat_personality for canned responses and LLM generation.
 */

import { validateSession, errorResponse, jsonResponse } from '../../../lib/auth'
import type { ChatPersonality, ReactionType } from '../../../lib/botPersonas'
import { getRandomReaction, shouldBotSpeak } from '../../../lib/botPersonas'
import {
  type ActiveGameRow,
  type BotPersonaRow,
  safeParseMoves,
  ROWS,
  COLUMNS,
  WIN_LENGTH,
} from '../../../lib/types'
import { createDb } from '../../../../shared/db/client'
import { activeGames, gameMessages, botPersonas } from '../../../../shared/db/schema'
import { eq, and, count } from 'drizzle-orm'

interface Env {
  DB: D1Database
  AI: Ai
}

type Board = (1 | 2 | null)[][]
type Player = 1 | 2

/**
 * POST /api/match/:id/bot-reaction - Request bot reaction after player move
 */
export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB, AI } = context.env
  const db = createDb(DB)
  const gameId = context.params.id as string

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Get the game
    const game = await db.query.activeGames.findFirst({
      where: eq(activeGames.id, gameId),
      columns: {
        id: true,
        player1Id: true,
        player2Id: true,
        moves: true,
        currentTurn: true,
        status: true,
        winner: true,
        isBotGame: true,
        botDifficulty: true,
        botPersonaId: true,
      },
    })

    if (!game) {
      return errorResponse('Game not found', 404)
    }

    // Verify this is a bot game
    if (game.isBotGame !== 1) {
      return jsonResponse({ message: null, reason: 'not_bot_game' })
    }

    // Verify user is the player (not the bot)
    if (game.player1Id !== session.userId) {
      return errorResponse('You are not the player in this game', 403)
    }

    const moves = safeParseMoves(game.moves)

    // Need at least one move to react to
    if (moves.length === 0) {
      return jsonResponse({ message: null, reason: 'no_moves' })
    }

    // Analyze the position
    const board = replayMoves(moves)
    if (!board) {
      return jsonResponse({ message: null, reason: 'invalid_game_state' })
    }

    // Load bot persona personality if available
    let personality: ChatPersonality | null = null
    if (game.botPersonaId) {
      const persona = await db.query.botPersonas.findFirst({
        where: eq(botPersonas.id, game.botPersonaId),
        columns: {
          id: true,
          name: true,
          chatPersonality: true,
        },
      })

      if (persona) {
        try {
          personality = JSON.parse(persona.chatPersonality) as ChatPersonality
        } catch {
          // Fall back to default personality
        }
      }
    }

    const context_analysis = analyzePosition(board, moves)
    const decision = await shouldCommentWithPersonality(DB, context_analysis, gameId, moves.length, personality)

    if (!decision.shouldComment) {
      return jsonResponse({ message: null, reason: decision.reason })
    }

    // Try to use a canned response first (faster), fall back to LLM
    let botMessage: string | null = null

    // Map analysis to reaction type for canned responses
    const reactionType = getReactionTypeForAnalysis(context_analysis)
    if (personality && reactionType) {
      // Use chattiness to decide whether to use canned vs LLM
      // For more interesting situations, prefer LLM
      const useCanned = context_analysis.moveQuality === 'neutral' ||
        (Math.random() > 0.3) // 70% chance to use canned for faster response

      if (useCanned) {
        botMessage = getRandomReaction(personality, reactionType)
      }
    }

    // Fall back to LLM if no canned response
    if (!botMessage) {
      // Convert to ActiveGameRow format for compatibility
      const gameRow: ActiveGameRow = {
        id: game.id,
        player1_id: game.player1Id,
        player2_id: game.player2Id,
        moves: game.moves,
        current_turn: game.currentTurn,
        status: game.status,
        winner: game.winner,
        is_bot_game: game.isBotGame,
        bot_difficulty: game.botDifficulty,
        bot_persona_id: game.botPersonaId,
      }
      botMessage = await generateReactionMessage(AI, context_analysis, gameRow, personality)
    }

    if (!botMessage) {
      return jsonResponse({ message: null, reason: 'generation_failed' })
    }

    // Store the bot message
    const messageId = crypto.randomUUID()
    const now = Date.now()

    await db.insert(gameMessages).values({
      id: messageId,
      gameId,
      senderId: 'bot',
      senderType: 'bot',
      content: botMessage,
      createdAt: now,
    })

    return jsonResponse({
      message: botMessage,
      messageId,
      reason: decision.reason,
    })
  } catch (error) {
    console.error('POST /api/match/:id/bot-reaction error:', error)
    return errorResponse('Internal server error', 500)
  }
}

// ============================================================================
// GAME ANALYSIS HELPERS (simplified versions for server-side use)
// ============================================================================

interface PositionAnalysis {
  moveQuality: 'brilliant' | 'good' | 'neutral' | 'mistake' | 'blunder'
  playerThreats: number // count of player's winning threats
  botThreats: number // count of bot's winning threats
  isPlayerWinning: boolean
  isBotWinning: boolean
  gamePhase: 'opening' | 'midgame' | 'endgame'
  gameTension: 'calm' | 'building' | 'critical'
  lastColumn: number
  moveCount: number
  gameEnded: boolean
  winner: Player | 'draw' | null
  centerControl: { player: number; bot: number }
}

/**
 * Replay moves to reconstruct board state
 */
function replayMoves(moves: number[]): Board | null {
  const board: Board = Array.from({ length: ROWS }, () =>
    Array.from({ length: COLUMNS }, () => null)
  )

  for (let i = 0; i < moves.length; i++) {
    const column = moves[i]
    const player: Player = (i % 2 === 0) ? 1 : 2

    // Find the lowest empty row in this column
    let placed = false
    for (let row = ROWS - 1; row >= 0; row--) {
      if (board[row][column] === null) {
        board[row][column] = player
        placed = true
        break
      }
    }

    if (!placed) return null // Invalid move
  }

  return board
}

/**
 * Check if there's a winner
 */
function checkWinner(board: Board): Player | 'draw' | null {
  // Check horizontal
  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col <= COLUMNS - WIN_LENGTH; col++) {
      const cell = board[row][col]
      if (cell !== null &&
          cell === board[row][col + 1] &&
          cell === board[row][col + 2] &&
          cell === board[row][col + 3]) {
        return cell
      }
    }
  }

  // Check vertical
  for (let col = 0; col < COLUMNS; col++) {
    for (let row = 0; row <= ROWS - WIN_LENGTH; row++) {
      const cell = board[row][col]
      if (cell !== null &&
          cell === board[row + 1][col] &&
          cell === board[row + 2][col] &&
          cell === board[row + 3][col]) {
        return cell
      }
    }
  }

  // Check diagonal (down-right)
  for (let row = 0; row <= ROWS - WIN_LENGTH; row++) {
    for (let col = 0; col <= COLUMNS - WIN_LENGTH; col++) {
      const cell = board[row][col]
      if (cell !== null &&
          cell === board[row + 1][col + 1] &&
          cell === board[row + 2][col + 2] &&
          cell === board[row + 3][col + 3]) {
        return cell
      }
    }
  }

  // Check diagonal (down-left)
  for (let row = 0; row <= ROWS - WIN_LENGTH; row++) {
    for (let col = WIN_LENGTH - 1; col < COLUMNS; col++) {
      const cell = board[row][col]
      if (cell !== null &&
          cell === board[row + 1][col - 1] &&
          cell === board[row + 2][col - 2] &&
          cell === board[row + 3][col - 3]) {
        return cell
      }
    }
  }

  // Check for draw
  const isFull = board[0].every(cell => cell !== null)
  if (isFull) return 'draw'

  return null
}

/**
 * Count winning threats (3-in-a-row with empty space that can be played)
 */
function countThreats(board: Board, player: Player): number {
  let threats = 0

  // For each possible winning line, check if player has 3 and there's a playable empty
  const directions = [
    { dr: 0, dc: 1 },  // horizontal
    { dr: 1, dc: 0 },  // vertical
    { dr: 1, dc: 1 },  // diagonal down-right
    { dr: 1, dc: -1 }, // diagonal down-left
  ]

  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLUMNS; col++) {
      for (const { dr, dc } of directions) {
        const endRow = row + (WIN_LENGTH - 1) * dr
        const endCol = col + (WIN_LENGTH - 1) * dc

        if (endRow < 0 || endRow >= ROWS || endCol < 0 || endCol >= COLUMNS) continue

        let playerCount = 0
        let emptyPos: { row: number; col: number } | null = null

        for (let i = 0; i < WIN_LENGTH; i++) {
          const r = row + i * dr
          const c = col + i * dc
          const cell = board[r][c]

          if (cell === player) {
            playerCount++
          } else if (cell === null) {
            if (emptyPos === null) {
              emptyPos = { row: r, col: c }
            } else {
              emptyPos = null // More than one empty, not a threat
              break
            }
          } else {
            // Opponent piece, not a threat
            emptyPos = null
            break
          }
        }

        if (playerCount === 3 && emptyPos !== null) {
          // Check if the empty position is playable (has support below or is bottom row)
          if (emptyPos.row === ROWS - 1 || board[emptyPos.row + 1][emptyPos.col] !== null) {
            threats++
          }
        }
      }
    }
  }

  return threats
}

/**
 * Estimate move quality based on simple heuristics
 */
function estimateMoveQuality(
  boardBefore: Board | null,
  boardAfter: Board,
  lastMove: number,
  player: Player
): 'brilliant' | 'good' | 'neutral' | 'mistake' | 'blunder' {
  const winner = checkWinner(boardAfter)

  // Winning move is always brilliant
  if (winner === player) {
    return 'brilliant'
  }

  // Check if opponent (bot, player 2) has a winning threat we didn't block
  const botThreats = countThreats(boardAfter, 2)

  // If we gave the opponent a winning threat, that's bad
  if (botThreats > 0) {
    // Check if there was a threat before that we should have blocked
    if (boardBefore) {
      const threatsBefore = countThreats(boardBefore, 2)
      if (threatsBefore > 0) {
        // We had a threat to block but didn't
        return 'blunder'
      }
    }
    // We created a situation where opponent has a threat
    return 'mistake'
  }

  // Check if we created a threat
  const playerThreats = countThreats(boardAfter, player)
  if (playerThreats > 0) {
    return 'good'
  }

  // Center column moves are generally better in early game
  if (lastMove === 3 && boardAfter[0].filter(c => c !== null).length < 10) {
    return 'good'
  }

  return 'neutral'
}

/**
 * Analyze the current position
 */
function analyzePosition(board: Board, moves: number[]): PositionAnalysis {
  const moveCount = moves.length
  const lastColumn = moves[moves.length - 1]
  const winner = checkWinner(board)
  const gameEnded = winner !== null

  // Get board before the last move for comparison
  const boardBefore = moves.length > 1 ? replayMoves(moves.slice(0, -1)) : null

  // The last move was made by player 1 if moveCount is odd, player 2 if even
  const lastMovePlayer: Player = moveCount % 2 === 1 ? 1 : 2
  const moveQuality = estimateMoveQuality(boardBefore, board, lastColumn, lastMovePlayer)

  const playerThreats = countThreats(board, 1)
  const botThreats = countThreats(board, 2)

  // Simple position evaluation
  const isPlayerWinning = playerThreats > botThreats + 1
  const isBotWinning = botThreats > playerThreats + 1

  // Game phase
  let gamePhase: 'opening' | 'midgame' | 'endgame'
  if (moveCount < 6) {
    gamePhase = 'opening'
  } else if (moveCount < 20) {
    gamePhase = 'midgame'
  } else {
    gamePhase = 'endgame'
  }

  // Game tension
  let gameTension: 'calm' | 'building' | 'critical'
  if (playerThreats > 0 || botThreats > 0) {
    gameTension = 'critical'
  } else if (moveCount > 15 || (playerThreats + botThreats > 0)) {
    gameTension = 'building'
  } else {
    gameTension = 'calm'
  }

  // Center control
  const centerCols = [2, 3, 4]
  let playerCenter = 0
  let botCenter = 0
  for (let row = 0; row < ROWS; row++) {
    for (const col of centerCols) {
      if (board[row][col] === 1) playerCenter++
      else if (board[row][col] === 2) botCenter++
    }
  }

  return {
    moveQuality,
    playerThreats,
    botThreats,
    isPlayerWinning,
    isBotWinning,
    gamePhase,
    gameTension,
    lastColumn,
    moveCount,
    gameEnded,
    winner,
    centerControl: { player: playerCenter, bot: botCenter },
  }
}

/**
 * Map position analysis to reaction type for canned responses
 */
function getReactionTypeForAnalysis(analysis: PositionAnalysis): ReactionType | null {
  // Game end states
  if (analysis.gameEnded) {
    if (analysis.winner === 'draw') return 'draw'
    if (analysis.winner === 1) return 'gameLost' // Player won, bot lost
    return 'gameWon' // Bot won
  }

  // Move quality reactions
  if (analysis.moveQuality === 'brilliant' || analysis.moveQuality === 'good') {
    return 'playerGoodMove'
  }
  if (analysis.moveQuality === 'blunder' || analysis.moveQuality === 'mistake') {
    return 'playerBlunder'
  }

  // Position-based reactions
  if (analysis.isBotWinning || analysis.botThreats > analysis.playerThreats) {
    return 'botWinning'
  }
  if (analysis.isPlayerWinning || analysis.playerThreats > analysis.botThreats) {
    return 'botLosing'
  }

  return null
}

/**
 * Decide whether bot should comment, using persona chattiness.
 * Uses database to track bot message count for rate limiting.
 */
async function shouldCommentWithPersonality(
  DB: D1Database,
  analysis: PositionAnalysis,
  gameId: string,
  moveCount: number,
  personality: ChatPersonality | null
): Promise<{ shouldComment: boolean; reason: string }> {
  // Query database for bot reaction message count
  const db = createDb(DB)
  const messageCountResult = await db
    .select({ count: count() })
    .from(gameMessages)
    .where(and(eq(gameMessages.gameId, gameId), eq(gameMessages.senderType, 'bot')))

  const botMessageCount = messageCountResult[0]?.count ?? 0

  // Get chattiness from persona (default 0.5)
  const chattiness = personality?.chattiness ?? 0.5

  // Adjust rate limit based on chattiness
  // Higher chattiness = more frequent comments allowed
  // chattiness 0.2 = ~1 comment per 5 moves, chattiness 0.8 = ~1 comment per 1.5 moves
  const movesPerComment = Math.max(1, Math.floor(3 / (chattiness + 0.1)))
  const maxAllowedMessages = Math.floor(moveCount / movesPerComment)
  const rateLimited = botMessageCount >= maxAllowedMessages

  // Game ending always gets a comment (bypass rate limit)
  if (analysis.gameEnded) {
    return { shouldComment: true, reason: 'game_ended' }
  }

  // Blunders always get a comment (unless rate limited)
  if (analysis.moveQuality === 'blunder') {
    if (rateLimited) {
      return { shouldComment: false, reason: 'rate_limited' }
    }
    return { shouldComment: true, reason: 'blunder' }
  }

  // Brilliant moves get a comment
  if (analysis.moveQuality === 'brilliant') {
    if (rateLimited) {
      return { shouldComment: false, reason: 'rate_limited' }
    }
    return { shouldComment: true, reason: 'brilliant_move' }
  }

  // Player created a winning threat
  if (analysis.playerThreats > 0) {
    if (rateLimited) {
      return { shouldComment: false, reason: 'rate_limited' }
    }
    // Use chattiness to determine if we comment
    if (!shouldBotSpeak(chattiness)) {
      return { shouldComment: false, reason: 'chattiness_skip' }
    }
    return { shouldComment: true, reason: 'player_threat' }
  }

  // Bot has a winning threat
  if (analysis.botThreats > 0) {
    if (rateLimited) {
      return { shouldComment: false, reason: 'rate_limited' }
    }
    if (!shouldBotSpeak(chattiness)) {
      return { shouldComment: false, reason: 'chattiness_skip' }
    }
    return { shouldComment: true, reason: 'bot_threat' }
  }

  // Mistakes - use chattiness to determine comment probability
  if (analysis.moveQuality === 'mistake') {
    if (rateLimited || !shouldBotSpeak(chattiness * 0.6)) {
      return { shouldComment: false, reason: 'random_skip' }
    }
    return { shouldComment: true, reason: 'mistake' }
  }

  // Good moves in tense situations
  if (analysis.moveQuality === 'good' && analysis.gameTension !== 'calm') {
    if (rateLimited || !shouldBotSpeak(chattiness * 0.6)) {
      return { shouldComment: false, reason: 'random_skip' }
    }
    return { shouldComment: true, reason: 'good_move' }
  }

  // Endgame - use lower chattiness factor
  if (analysis.gamePhase === 'endgame') {
    if (rateLimited || !shouldBotSpeak(chattiness * 0.3)) {
      return { shouldComment: false, reason: 'random_skip' }
    }
    return { shouldComment: true, reason: 'endgame_tension' }
  }

  return { shouldComment: false, reason: 'not_notable' }
}

/**
 * Default personality for backward compatibility
 */
const DEFAULT_REACTION_PERSONALITY: ChatPersonality = {
  name: 'Bot',
  systemPrompt: `You are a friendly Connect 4 opponent providing brief, natural commentary during the game. Be competitive but friendly.`,
  reactions: {
    gameStart: ["Let's play!"],
    playerGoodMove: ["Nice move!"],
    playerBlunder: ["Interesting choice..."],
    botWinning: ["Looking good for me!"],
    botLosing: ["You're doing well!"],
    gameWon: ["GG!"],
    gameLost: ["Good game!"],
    draw: ["A draw!"],
  },
  chattiness: 0.5,
  useEmoji: false,
  maxLength: 100,
  temperature: 0.7,
}

/**
 * Generate reaction message using LLM with persona personality
 */
async function generateReactionMessage(
  AI: Ai,
  analysis: PositionAnalysis,
  game: ActiveGameRow,
  personality: ChatPersonality | null
): Promise<string | null> {
  try {
    const pers = personality || DEFAULT_REACTION_PERSONALITY

    // Build context description
    let situationDesc = ''

    if (analysis.gameEnded) {
      if (analysis.winner === 'draw') {
        situationDesc = 'The game just ended in a draw.'
      } else if (analysis.winner === 1) {
        situationDesc = 'The player just won the game!'
      } else {
        situationDesc = 'You (the bot) just won the game!'
      }
    } else {
      situationDesc = `Move ${analysis.moveCount}, ${analysis.gamePhase} phase. `

      if (analysis.moveQuality === 'blunder') {
        situationDesc += 'The player just made a significant mistake. '
      } else if (analysis.moveQuality === 'brilliant') {
        situationDesc += 'The player just made a great move. '
      } else if (analysis.moveQuality === 'mistake') {
        situationDesc += 'The player made a questionable move. '
      } else if (analysis.moveQuality === 'good') {
        situationDesc += 'The player made a solid move. '
      }

      if (analysis.playerThreats > 0) {
        situationDesc += `The player has ${analysis.playerThreats} winning threat(s). `
      }
      if (analysis.botThreats > 0) {
        situationDesc += `You have ${analysis.botThreats} winning threat(s). `
      }

      situationDesc += `Game tension: ${analysis.gameTension}.`
    }

    // Build emoji rule based on personality
    const emojiRule = pers.useEmoji
      ? '- You may use 1-2 emojis if it fits your personality'
      : '- Do NOT use emojis'

    // Get example reactions from personality for guidance
    const reactionType = getReactionTypeForAnalysis(analysis)
    const exampleReactions = reactionType ? pers.reactions[reactionType] : []
    const exampleText = exampleReactions.length > 0
      ? `\nExamples of responses in your voice:\n${exampleReactions.map(r => `- "${r}"`).join('\n')}`
      : ''

    const systemPrompt = `${pers.systemPrompt}

SITUATION: ${situationDesc}

Generate ONE brief comment (1 sentence max, under ${pers.maxLength} characters) reacting to the situation. Stay in character and be:
- Natural and conversational
- Consistent with your personality
- Never condescending or mean
- Never reveal strategic advice
${emojiRule}
${exampleText}

Respond with ONLY the comment text.`

    const response = await AI.run('@cf/meta/llama-3.1-8b-instruct', {
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: 'Generate a brief reaction.' }
      ],
      max_tokens: Math.min(50, Math.ceil(pers.maxLength / 3)),
      temperature: pers.temperature,
    })

    if (response && typeof response === 'object' && 'response' in response) {
      const text = (response as { response: string }).response
      // Clean up and limit length
      return text.trim().replace(/^["']|["']$/g, '').slice(0, pers.maxLength)
    }

    return null
  } catch (error) {
    console.error('Bot reaction generation error:', error)
    // Return a random canned reaction as fallback
    const pers = personality || DEFAULT_REACTION_PERSONALITY
    const reactionType = getReactionTypeForAnalysis(analysis)
    if (reactionType) {
      return getRandomReaction(pers, reactionType)
    }
    return null
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
