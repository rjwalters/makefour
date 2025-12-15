/**
 * Bot Reaction API endpoint for proactive game-aware responses
 *
 * POST /api/match/:id/bot-reaction - Request a bot reaction after player's move
 *
 * This endpoint analyzes the current game position and may generate
 * a proactive comment from the bot about notable moves.
 */

import { validateSession, errorResponse, jsonResponse } from '../../../lib/auth'

interface Env {
  DB: D1Database
  AI: Ai
}

interface ActiveGameRow {
  id: string
  player1_id: string
  player2_id: string
  moves: string
  current_turn: number
  status: string
  winner: string | null
  is_bot_game: number
  bot_difficulty: string | null
}

interface GameMessageRow {
  id: string
  game_id: string
  created_at: number
}

// Board dimensions
const ROWS = 6
const COLUMNS = 7
const WIN_LENGTH = 4

type Board = (1 | 2 | null)[][]
type Player = 1 | 2

// Rate limiting: track last proactive comment per game
const reactionRateLimits = new Map<string, { lastMoveCount: number; commentCount: number }>()

/**
 * POST /api/match/:id/bot-reaction - Request bot reaction after player move
 */
export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB, AI } = context.env
  const gameId = context.params.id as string

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Get the game
    const game = await DB.prepare(`
      SELECT id, player1_id, player2_id, moves, current_turn, status, winner,
             is_bot_game, bot_difficulty
      FROM active_games
      WHERE id = ?
    `)
      .bind(gameId)
      .first<ActiveGameRow>()

    if (!game) {
      return errorResponse('Game not found', 404)
    }

    // Verify this is a bot game
    if (game.is_bot_game !== 1) {
      return jsonResponse({ message: null, reason: 'not_bot_game' })
    }

    // Verify user is the player (not the bot)
    if (game.player1_id !== session.userId) {
      return errorResponse('You are not the player in this game', 403)
    }

    const moves = JSON.parse(game.moves) as number[]

    // Need at least one move to react to
    if (moves.length === 0) {
      return jsonResponse({ message: null, reason: 'no_moves' })
    }

    // Analyze the position
    const board = replayMoves(moves)
    if (!board) {
      return jsonResponse({ message: null, reason: 'invalid_game_state' })
    }

    const context_analysis = analyzePosition(board, moves)
    const decision = shouldComment(context_analysis, gameId, moves.length)

    if (!decision.shouldComment) {
      return jsonResponse({ message: null, reason: decision.reason })
    }

    // Generate bot message using LLM
    const botMessage = await generateReactionMessage(AI, context_analysis, game)

    if (!botMessage) {
      return jsonResponse({ message: null, reason: 'generation_failed' })
    }

    // Store the bot message
    const messageId = crypto.randomUUID()
    const now = Date.now()

    await DB.prepare(`
      INSERT INTO game_messages (id, game_id, sender_id, sender_type, content, created_at)
      VALUES (?, ?, 'bot', 'bot', ?, ?)
    `).bind(messageId, gameId, botMessage, now).run()

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
 * Decide whether bot should comment
 */
function shouldComment(
  analysis: PositionAnalysis,
  gameId: string,
  moveCount: number
): { shouldComment: boolean; reason: string } {
  // Initialize rate limit state if needed
  if (!reactionRateLimits.has(gameId)) {
    reactionRateLimits.set(gameId, { lastMoveCount: 0, commentCount: 0 })
  }
  const rateState = reactionRateLimits.get(gameId)!

  // Rate limit: max 1 comment per 3 moves
  const movesSinceLastComment = moveCount - rateState.lastMoveCount
  const rateLimited = movesSinceLastComment < 3 && rateState.commentCount > 0

  // Game ending always gets a comment
  if (analysis.gameEnded) {
    updateRateLimit(gameId, moveCount)
    return { shouldComment: true, reason: 'game_ended' }
  }

  // Blunders always get a comment (unless rate limited)
  if (analysis.moveQuality === 'blunder') {
    if (rateLimited) {
      return { shouldComment: false, reason: 'rate_limited' }
    }
    updateRateLimit(gameId, moveCount)
    return { shouldComment: true, reason: 'blunder' }
  }

  // Brilliant moves get a comment
  if (analysis.moveQuality === 'brilliant') {
    if (rateLimited) {
      return { shouldComment: false, reason: 'rate_limited' }
    }
    updateRateLimit(gameId, moveCount)
    return { shouldComment: true, reason: 'brilliant_move' }
  }

  // Player created a winning threat
  if (analysis.playerThreats > 0) {
    if (rateLimited) {
      return { shouldComment: false, reason: 'rate_limited' }
    }
    updateRateLimit(gameId, moveCount)
    return { shouldComment: true, reason: 'player_threat' }
  }

  // Bot has a winning threat
  if (analysis.botThreats > 0) {
    if (rateLimited) {
      return { shouldComment: false, reason: 'rate_limited' }
    }
    updateRateLimit(gameId, moveCount)
    return { shouldComment: true, reason: 'bot_threat' }
  }

  // Mistakes get 30% chance comment
  if (analysis.moveQuality === 'mistake') {
    if (rateLimited || Math.random() > 0.3) {
      return { shouldComment: false, reason: 'random_skip' }
    }
    updateRateLimit(gameId, moveCount)
    return { shouldComment: true, reason: 'mistake' }
  }

  // Good moves in tense situations get 30% chance
  if (analysis.moveQuality === 'good' && analysis.gameTension !== 'calm') {
    if (rateLimited || Math.random() > 0.3) {
      return { shouldComment: false, reason: 'random_skip' }
    }
    updateRateLimit(gameId, moveCount)
    return { shouldComment: true, reason: 'good_move' }
  }

  // Endgame gets 10% chance
  if (analysis.gamePhase === 'endgame') {
    if (rateLimited || Math.random() > 0.1) {
      return { shouldComment: false, reason: 'random_skip' }
    }
    updateRateLimit(gameId, moveCount)
    return { shouldComment: true, reason: 'endgame_tension' }
  }

  return { shouldComment: false, reason: 'not_notable' }
}

function updateRateLimit(gameId: string, moveCount: number): void {
  const state = reactionRateLimits.get(gameId)
  if (state) {
    state.lastMoveCount = moveCount
    state.commentCount++
  }
}

/**
 * Generate reaction message using LLM
 */
async function generateReactionMessage(
  AI: Ai,
  analysis: PositionAnalysis,
  game: ActiveGameRow
): Promise<string | null> {
  try {
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

    const systemPrompt = `You are a Connect 4 bot opponent providing brief, natural commentary during the game.

SITUATION: ${situationDesc}

Generate ONE brief comment (1 sentence max) reacting to the situation. Be:
- Natural and conversational
- Competitive but friendly
- Never condescending or mean
- Never reveal strategic advice

Examples of good responses:
${analysis.gameEnded && analysis.winner === 1 ? '- "Good game! You got me this time."' : ''}
${analysis.gameEnded && analysis.winner === 2 ? '- "GG! That was a close one."' : ''}
${analysis.gameEnded && analysis.winner === 'draw' ? '- "A draw! That was intense."' : ''}
${analysis.moveQuality === 'blunder' && !analysis.gameEnded ? '- "Hmm, interesting choice..."' : ''}
${analysis.moveQuality === 'brilliant' && !analysis.gameEnded ? '- "Nice move!"' : ''}
${analysis.playerThreats > 0 && !analysis.gameEnded ? '- "Uh oh, I need to watch that."' : ''}
${analysis.botThreats > 0 && !analysis.gameEnded ? '- "Things are looking good for me..."' : ''}

Respond with ONLY the comment text.`

    const response = await AI.run('@cf/meta/llama-3.1-8b-instruct', {
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: 'Generate a brief reaction.' }
      ],
      max_tokens: 50,
      temperature: 0.8,
    })

    if (response && typeof response === 'object' && 'response' in response) {
      const text = (response as { response: string }).response
      // Clean up and limit length
      return text.trim().replace(/^["']|["']$/g, '').slice(0, 150)
    }

    return null
  } catch (error) {
    console.error('Bot reaction generation error:', error)
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
