/**
 * Bot Reaction Module
 *
 * Analyzes game positions and generates context for proactive bot chat responses.
 * Used by the bot-reaction API endpoint to make the bot comment on notable moves.
 */

import {
  type Board,
  type Player,
  getStateAtMove,
  checkWinner,
  ROWS,
} from '../game/makefour'
import { evaluatePosition, analyzeThreats, type Threat, EVAL_WEIGHTS } from './coach'
import { type MoveQuality, analyzeSingleMove } from './moveQuality'

/**
 * Game phase based on move count.
 */
export type GamePhase = 'opening' | 'midgame' | 'endgame'

/**
 * Game tension level based on position evaluation and threats.
 */
export type GameTension = 'calm' | 'building' | 'critical'

/**
 * Context about a move for LLM-based bot reactions.
 */
export interface MoveContext {
  /** Quality classification of the move */
  moveQuality: MoveQuality
  /** Position evaluation (positive = player ahead, negative = bot ahead) */
  positionEval: number
  /** Normalized position description */
  positionDescription: 'player_winning' | 'bot_winning' | 'even'
  /** Threats the player created with this move */
  playerThreats: Threat[]
  /** Threats the bot currently has */
  botThreats: Threat[]
  /** Whether the player is in a winning position */
  isPlayerWinning: boolean
  /** Whether this is a critical moment (close game or decisive move) */
  isCriticalMoment: boolean
  /** Current game phase */
  gamePhase: GamePhase
  /** Current game tension level */
  gameTension: GameTension
  /** The column that was played */
  column: number
  /** Total move count */
  moveCount: number
  /** Whether the game just ended */
  gameEnded: boolean
  /** Winner if game ended (1 = player, 2 = bot, 'draw', null = ongoing) */
  winner: Player | 'draw' | null
}

/**
 * Result of decision logic for whether bot should comment.
 */
export interface ReactionDecision {
  shouldComment: boolean
  priority: 'high' | 'medium' | 'low'
  reason: string
}

/**
 * Determines the game phase based on move count.
 */
export function getGamePhase(moveCount: number): GamePhase {
  if (moveCount < 6) return 'opening'
  if (moveCount < 20) return 'midgame'
  return 'endgame'
}

/**
 * Determines game tension based on evaluation and threats.
 */
export function getGameTension(
  positionEval: number,
  playerThreats: Threat[],
  botThreats: Threat[]
): GameTension {
  const hasWinningThreats =
    playerThreats.some((t) => t.type === 'win') || botThreats.some((t) => t.type === 'win')

  // Critical if either side has immediate winning threats
  if (hasWinningThreats) return 'critical'

  // Critical if position is very one-sided
  if (Math.abs(positionEval) >= EVAL_WEIGHTS.WIN) return 'critical'

  // Building tension if there are multiple threats
  const totalThreats = playerThreats.length + botThreats.length
  if (totalThreats >= 2) return 'building'

  // Building if position is notably unequal
  if (Math.abs(positionEval) > 200) return 'building'

  return 'calm'
}

/**
 * Counts how many pieces are in the center columns (2, 3, 4).
 */
export function countCenterControl(board: Board, player: Player): number {
  let count = 0
  const centerCols = [2, 3, 4]
  for (let row = 0; row < ROWS; row++) {
    for (const col of centerCols) {
      if (board[row][col] === player) count++
    }
  }
  return count
}

/**
 * Analyzes a move and builds context for bot reaction generation.
 *
 * @param moves - All moves in the game so far (including the latest)
 * @param playerNumber - Which player is the human (1 or 2)
 * @returns MoveContext with analysis results, or null if analysis fails
 */
export async function analyzeMoveForReaction(
  moves: number[],
  playerNumber: 1 | 2
): Promise<MoveContext | null> {
  if (moves.length === 0) return null

  const latestMove = moves[moves.length - 1]
  const movesBefore = moves.slice(0, -1)

  // Get board state BEFORE the move
  const stateBefore = getStateAtMove(moves, moves.length - 1)
  if (!stateBefore) return null

  // Get board state AFTER the move
  const stateAfter = getStateAtMove(moves, moves.length)
  if (!stateAfter) return null

  // Analyze move quality
  const moveAnalysis = await analyzeSingleMove(
    stateBefore.board,
    stateBefore.currentPlayer,
    latestMove,
    moves.length - 1,
    movesBefore
  )

  // Get threats after the move
  const playerAsNumber: Player = playerNumber
  const botAsNumber: Player = playerNumber === 1 ? 2 : 1
  const threatAnalysis = analyzeThreats(stateAfter.board, stateAfter.currentPlayer)

  // Separate threats by who they benefit
  const playerThreats = threatAnalysis.threats.filter((t) => t.player === playerAsNumber)
  const botThreats = threatAnalysis.threats.filter((t) => t.player === botAsNumber)

  // Evaluate position from player's perspective
  const positionEval = evaluatePosition(stateAfter.board, playerAsNumber)

  // Determine position description
  let positionDescription: 'player_winning' | 'bot_winning' | 'even'
  if (positionEval > 100) {
    positionDescription = 'player_winning'
  } else if (positionEval < -100) {
    positionDescription = 'bot_winning'
  } else {
    positionDescription = 'even'
  }

  // Check if game ended
  const winner = checkWinner(stateAfter.board)
  const gameEnded = winner !== null

  // Determine if this is a critical moment
  const gamePhase = getGamePhase(moves.length)
  const gameTension = getGameTension(positionEval, playerThreats, botThreats)
  const isCriticalMoment =
    gameTension === 'critical' ||
    (gameTension === 'building' && gamePhase === 'endgame') ||
    gameEnded

  return {
    moveQuality: moveAnalysis.quality,
    positionEval,
    positionDescription,
    playerThreats,
    botThreats,
    isPlayerWinning: positionEval >= EVAL_WEIGHTS.WIN,
    isCriticalMoment,
    gamePhase,
    gameTension,
    column: latestMove,
    moveCount: moves.length,
    gameEnded,
    winner,
  }
}

/**
 * Rate limiting state for bot reactions.
 * Key: gameId, Value: { lastCommentMove: number, consecutiveComments: number }
 */
const reactionRateLimits = new Map<string, { lastCommentMove: number; commentCount: number }>()

/**
 * Clears rate limit state for a game (e.g., when game ends).
 */
export function clearReactionRateLimit(gameId: string): void {
  reactionRateLimits.delete(gameId)
}

/**
 * Decides whether the bot should comment on the current move.
 * Implements rate limiting to prevent spam.
 *
 * Decision factors:
 * - Always comment: Brilliant moves, blunders, game-ending moves
 * - Sometimes comment (30% chance): Good/bad moves, new threats
 * - Rarely comment (10% chance): Neutral moves
 * - Rate limit: Max 1 proactive message per 3 moves
 *
 * @param context - The move context from analyzeMoveForReaction
 * @param gameId - The game ID for rate limiting
 * @returns Decision object with shouldComment, priority, and reason
 */
export function shouldBotComment(context: MoveContext, gameId: string): ReactionDecision {
  // Initialize rate limit state if needed
  if (!reactionRateLimits.has(gameId)) {
    reactionRateLimits.set(gameId, { lastCommentMove: 0, commentCount: 0 })
  }
  const rateState = reactionRateLimits.get(gameId)!

  // Check rate limit: max 1 comment per 3 moves
  const movesSinceLastComment = context.moveCount - rateState.lastCommentMove
  const rateLimited = movesSinceLastComment < 3 && rateState.commentCount > 0

  // Game-ending moves always get a comment
  if (context.gameEnded) {
    return {
      shouldComment: true,
      priority: 'high',
      reason: context.winner === 'draw' ? 'game_draw' : 'game_won',
    }
  }

  // Blunders always get a comment (unless rate limited)
  if (context.moveQuality === 'blunder') {
    if (rateLimited) {
      return { shouldComment: false, priority: 'high', reason: 'rate_limited' }
    }
    updateRateLimit(gameId, context.moveCount)
    return { shouldComment: true, priority: 'high', reason: 'blunder' }
  }

  // Optimal/brilliant moves get a comment (unless rate limited)
  if (context.moveQuality === 'optimal' && context.isCriticalMoment) {
    if (rateLimited) {
      return { shouldComment: false, priority: 'high', reason: 'rate_limited' }
    }
    updateRateLimit(gameId, context.moveCount)
    return { shouldComment: true, priority: 'high', reason: 'brilliant_move' }
  }

  // Player created a winning threat
  if (context.playerThreats.some((t) => t.type === 'win')) {
    if (rateLimited) {
      return { shouldComment: false, priority: 'medium', reason: 'rate_limited' }
    }
    updateRateLimit(gameId, context.moveCount)
    return { shouldComment: true, priority: 'medium', reason: 'player_threat' }
  }

  // Bot has a winning threat that player didn't block
  if (context.botThreats.some((t) => t.type === 'win')) {
    if (rateLimited) {
      return { shouldComment: false, priority: 'medium', reason: 'rate_limited' }
    }
    updateRateLimit(gameId, context.moveCount)
    return { shouldComment: true, priority: 'medium', reason: 'bot_threat' }
  }

  // Mistakes get 30% chance comment
  if (context.moveQuality === 'mistake') {
    if (rateLimited || Math.random() > 0.3) {
      return { shouldComment: false, priority: 'medium', reason: 'random_skip' }
    }
    updateRateLimit(gameId, context.moveCount)
    return { shouldComment: true, priority: 'medium', reason: 'mistake' }
  }

  // Good moves in tense situations get 30% chance
  if (context.moveQuality === 'good' && context.gameTension !== 'calm') {
    if (rateLimited || Math.random() > 0.3) {
      return { shouldComment: false, priority: 'low', reason: 'random_skip' }
    }
    updateRateLimit(gameId, context.moveCount)
    return { shouldComment: true, priority: 'low', reason: 'good_move' }
  }

  // Neutral moves get 10% chance in endgame
  if (context.gamePhase === 'endgame' && Math.random() < 0.1) {
    if (rateLimited) {
      return { shouldComment: false, priority: 'low', reason: 'rate_limited' }
    }
    updateRateLimit(gameId, context.moveCount)
    return { shouldComment: true, priority: 'low', reason: 'endgame_tension' }
  }

  return { shouldComment: false, priority: 'low', reason: 'not_notable' }
}

/**
 * Updates rate limit state after a comment.
 */
function updateRateLimit(gameId: string, moveCount: number): void {
  const state = reactionRateLimits.get(gameId)
  if (state) {
    state.lastCommentMove = moveCount
    state.commentCount++
  }
}

/**
 * Builds a system prompt for the LLM with game analysis context.
 */
export function buildReactionSystemPrompt(context: MoveContext): string {
  const qualityDescriptions: Record<MoveQuality, string> = {
    optimal: 'brilliant/optimal',
    good: 'good',
    neutral: 'neutral/okay',
    mistake: 'a mistake',
    blunder: 'a blunder',
  }

  const threatDescription = (threats: Threat[]): string => {
    if (threats.length === 0) return 'no immediate threats'
    const winThreats = threats.filter((t) => t.type === 'win')
    if (winThreats.length > 0) {
      return `winning threat${winThreats.length > 1 ? 's' : ''} (can win in column ${winThreats.map((t) => t.column + 1).join(', ')})`
    }
    return `${threats.length} potential threat${threats.length > 1 ? 's' : ''}`
  }

  let gameState = ''
  if (context.gameEnded) {
    if (context.winner === 'draw') {
      gameState = 'The game just ended in a draw!'
    } else if (context.winner === 1) {
      gameState = 'The player just won the game!'
    } else {
      gameState = 'You (the bot) just won the game!'
    }
  } else {
    gameState = `Move ${context.moveCount}, ${context.gamePhase} phase.`
  }

  return `You are a Connect 4 bot opponent providing brief, natural commentary during the game.

CURRENT GAME ANALYSIS:
- ${gameState}
- Move quality: The player's last move was ${qualityDescriptions[context.moveQuality]}
- Position evaluation: ${context.positionDescription === 'player_winning' ? 'Player is ahead' : context.positionDescription === 'bot_winning' ? 'Bot is ahead' : 'Position is even'}
- Player's threats: ${threatDescription(context.playerThreats)}
- Bot's threats: ${threatDescription(context.botThreats)}
- Game tension: ${context.gameTension}

YOUR TASK:
Generate ONE brief, natural comment (1 sentence max) reacting to the player's move. Match the situation:
${context.gameEnded ? '- Game just ended - be a good sport!' : ''}
${context.moveQuality === 'blunder' ? "- Player made a mistake - be subtle, don't lecture" : ''}
${context.moveQuality === 'optimal' ? '- Great move - acknowledge it genuinely' : ''}
${context.playerThreats.some((t) => t.type === 'win') ? '- Player created a winning threat - show concern!' : ''}
${context.botThreats.some((t) => t.type === 'win') ? "- Bot has a winning threat - be confident but not cocky" : ''}

RULES:
- Maximum 1-2 sentences, keep it SHORT
- Be conversational and natural
- NEVER reveal optimal moves or give strategic hints
- NEVER be condescending or lecture the player
- Match the bot's competitive but friendly personality
- No emojis (or max 1 if really fitting)

Respond ONLY with the comment text, nothing else.`
}

/**
 * Builds example responses for different scenarios (for few-shot prompting if needed).
 */
export const REACTION_EXAMPLES: Record<string, string[]> = {
  blunder: [
    "Hmm, interesting choice there...",
    "Wait, are you sure about that one?",
    "I like where this is going.",
    "Oh? That's... bold.",
  ],
  brilliant_move: [
    "Okay, that was actually really good.",
    "Nice! Didn't see that coming.",
    "Well played.",
    "Ooh, clever.",
  ],
  player_threat: [
    "Uh oh, that's a problem for me.",
    "I see what you're doing there...",
    "That's dangerous.",
    "I need to watch that.",
  ],
  bot_threat: [
    "I think I've got something here...",
    "Things are looking up for me.",
    "This is getting interesting.",
  ],
  game_won_player: [
    "Good game! You got me.",
    "Well played, I didn't see that coming.",
    "GG! Rematch?",
  ],
  game_won_bot: [
    "Got you! Good game though.",
    "That was close! Want to play again?",
    "GG! You put up a good fight.",
  ],
  game_draw: [
    "A draw! That was intense.",
    "Tied up! Neither of us could break through.",
    "Good defensive game from both of us.",
  ],
  good_move: [
    "Solid move.",
    "Not bad!",
    "Good thinking.",
  ],
  endgame_tension: [
    "This is getting tense...",
    "Down to the wire now.",
    "Every move counts here.",
  ],
}
