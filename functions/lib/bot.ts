/**
 * Server-side Bot AI for MakeFour
 *
 * Provides a unified interface for bot move selection with pluggable AI engines.
 * Supports multiple engine types: minimax (default), neural, MCTS, etc.
 *
 * For backward compatibility, the legacy minimax implementation is preserved
 * and used when no engine is specified.
 */

import {
  type Board,
  type Player,
  applyMove,
  checkWinner,
  ROWS,
  COLUMNS,
  WIN_LENGTH,
} from './game'
import type { EngineConfig, MoveResult, AIEngine, EngineType } from './ai-engine'
import { engineRegistry, DEFAULT_ENGINE_CONFIGS } from './ai-engine'
// Import engines to register them
import './engines'

// ============================================================================
// TYPES
// ============================================================================

export type DifficultyLevel = 'beginner' | 'intermediate' | 'expert' | 'perfect'

interface DifficultyConfig {
  searchDepth: number
  errorRate: number
}

/**
 * Extended configuration for bot persona with engine selection.
 */
export interface BotPersonaConfig {
  difficulty: DifficultyLevel
  engine?: EngineType
  customEngineParams?: Record<string, unknown>
}

const DIFFICULTY_CONFIGS: Record<DifficultyLevel, DifficultyConfig> = {
  beginner: { searchDepth: 2, errorRate: 0.3 },
  intermediate: { searchDepth: 4, errorRate: 0.1 },
  expert: { searchDepth: 8, errorRate: 0.02 },
  perfect: { searchDepth: 42, errorRate: 0 },
}

// ============================================================================
// POSITION EVALUATION
// ============================================================================

const EVAL_WEIGHTS = {
  WIN: 100000,
  THREE_IN_ROW: 100,
  TWO_IN_ROW: 10,
  CENTER_CONTROL: 3,
}

/**
 * Evaluates a window of 4 cells for scoring potential.
 */
function evaluateWindow(window: (Player | null)[], player: Player): number {
  const opponent: Player = player === 1 ? 2 : 1
  const playerCount = window.filter((c) => c === player).length
  const opponentCount = window.filter((c) => c === opponent).length
  const emptyCount = window.filter((c) => c === null).length

  if (opponentCount > 0 && playerCount > 0) return 0

  if (playerCount === 4) return EVAL_WEIGHTS.WIN
  if (playerCount === 3 && emptyCount === 1) return EVAL_WEIGHTS.THREE_IN_ROW
  if (playerCount === 2 && emptyCount === 2) return EVAL_WEIGHTS.TWO_IN_ROW

  if (opponentCount === 4) return -EVAL_WEIGHTS.WIN
  if (opponentCount === 3 && emptyCount === 1) return -EVAL_WEIGHTS.THREE_IN_ROW
  if (opponentCount === 2 && emptyCount === 2) return -EVAL_WEIGHTS.TWO_IN_ROW

  return 0
}

/**
 * Evaluates the board position from the perspective of the given player.
 */
function evaluatePosition(board: Board, player: Player): number {
  let score = 0

  // Center column control
  const centerCol = Math.floor(COLUMNS / 2)
  for (let row = 0; row < ROWS; row++) {
    if (board[row][centerCol] === player) {
      score += EVAL_WEIGHTS.CENTER_CONTROL
    } else if (board[row][centerCol] !== null) {
      score -= EVAL_WEIGHTS.CENTER_CONTROL
    }
  }

  // Horizontal windows
  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col <= COLUMNS - WIN_LENGTH; col++) {
      const window = [
        board[row][col],
        board[row][col + 1],
        board[row][col + 2],
        board[row][col + 3],
      ]
      score += evaluateWindow(window, player)
    }
  }

  // Vertical windows
  for (let col = 0; col < COLUMNS; col++) {
    for (let row = 0; row <= ROWS - WIN_LENGTH; row++) {
      const window = [
        board[row][col],
        board[row + 1][col],
        board[row + 2][col],
        board[row + 3][col],
      ]
      score += evaluateWindow(window, player)
    }
  }

  // Diagonal windows (down-right)
  for (let row = 0; row <= ROWS - WIN_LENGTH; row++) {
    for (let col = 0; col <= COLUMNS - WIN_LENGTH; col++) {
      const window = [
        board[row][col],
        board[row + 1][col + 1],
        board[row + 2][col + 2],
        board[row + 3][col + 3],
      ]
      score += evaluateWindow(window, player)
    }
  }

  // Diagonal windows (down-left)
  for (let row = 0; row <= ROWS - WIN_LENGTH; row++) {
    for (let col = WIN_LENGTH - 1; col < COLUMNS; col++) {
      const window = [
        board[row][col],
        board[row + 1][col - 1],
        board[row + 2][col - 2],
        board[row + 3][col - 3],
      ]
      score += evaluateWindow(window, player)
    }
  }

  return score
}

// ============================================================================
// MOVE GENERATION
// ============================================================================

/**
 * Returns valid column indices for the current board.
 */
function getValidMoves(board: Board): number[] {
  const moves: number[] = []
  for (let col = 0; col < COLUMNS; col++) {
    if (board[0][col] === null) {
      moves.push(col)
    }
  }
  return moves
}

/**
 * Orders moves with center columns first for better alpha-beta pruning.
 */
function orderMoves(moves: number[]): number[] {
  const centerCol = Math.floor(COLUMNS / 2)
  return [...moves].sort((a, b) => Math.abs(a - centerCol) - Math.abs(b - centerCol))
}

// ============================================================================
// MINIMAX WITH ALPHA-BETA PRUNING
// ============================================================================

interface MinimaxResult {
  score: number
  move: number | null
}

/**
 * Minimax search with alpha-beta pruning and time limit.
 */
function minimax(
  board: Board,
  depth: number,
  alpha: number,
  beta: number,
  maximizingPlayer: boolean,
  player: Player,
  currentPlayer: Player,
  deadline: number
): MinimaxResult {
  // Check time limit
  if (Date.now() > deadline) {
    return { score: evaluatePosition(board, player), move: null }
  }

  // Check terminal states
  const winner = checkWinner(board)
  if (winner !== null) {
    if (winner === 'draw') return { score: 0, move: null }
    const winScore = EVAL_WEIGHTS.WIN + depth * 100
    return {
      score: winner === player ? winScore : -winScore,
      move: null,
    }
  }

  const validMoves = getValidMoves(board)
  if (validMoves.length === 0) return { score: 0, move: null }

  // Depth limit reached
  if (depth === 0) {
    return { score: evaluatePosition(board, player), move: null }
  }

  const orderedMoves = orderMoves(validMoves)
  const nextPlayer: Player = currentPlayer === 1 ? 2 : 1

  if (maximizingPlayer) {
    let maxScore = -Infinity
    let bestMove = orderedMoves[0]

    for (const move of orderedMoves) {
      const result = applyMove(board, move, currentPlayer)
      if (!result.success || !result.board) continue

      const { score } = minimax(
        result.board,
        depth - 1,
        alpha,
        beta,
        false,
        player,
        nextPlayer,
        deadline
      )

      if (score > maxScore) {
        maxScore = score
        bestMove = move
      }

      alpha = Math.max(alpha, score)
      if (beta <= alpha) break
    }

    return { score: maxScore, move: bestMove }
  } else {
    let minScore = Infinity
    let bestMove = orderedMoves[0]

    for (const move of orderedMoves) {
      const result = applyMove(board, move, currentPlayer)
      if (!result.success || !result.board) continue

      const { score } = minimax(
        result.board,
        depth - 1,
        alpha,
        beta,
        true,
        player,
        nextPlayer,
        deadline
      )

      if (score < minScore) {
        minScore = score
        bestMove = move
      }

      beta = Math.min(beta, score)
      if (beta <= alpha) break
    }

    return { score: minScore, move: bestMove }
  }
}

// ============================================================================
// TIME-BUDGETED MOVE SUGGESTION
// ============================================================================

/**
 * Calculate how much time the bot should spend on this move.
 *
 * @param timeRemainingMs - Bot's remaining time in milliseconds
 * @param moveCount - Total moves played so far
 * @param difficulty - Bot difficulty level
 * @returns Time budget in milliseconds
 */
export function calculateTimeBudget(
  timeRemainingMs: number,
  moveCount: number,
  difficulty: DifficultyLevel
): number {
  // Estimate moves remaining (Connect Four rarely goes past 30 moves per player)
  const movesRemaining = Math.max(42 - moveCount, 8)
  const movesPerPlayer = Math.ceil(movesRemaining / 2)

  // Base allocation: divide remaining time by expected moves
  const baseAllocation = timeRemainingMs / movesPerPlayer

  // Scale by difficulty (harder bots use more time)
  const difficultyMultiplier: Record<DifficultyLevel, number> = {
    beginner: 0.1,
    intermediate: 0.3,
    expert: 0.6,
    perfect: 0.9,
  }

  const budget = baseAllocation * difficultyMultiplier[difficulty]

  // Clamp to reasonable bounds: 50ms - 5000ms
  return Math.max(50, Math.min(5000, budget))
}

/**
 * Suggest the best move using iterative deepening with time limit.
 *
 * @param board - Current board state
 * @param currentPlayer - Player to move (the bot)
 * @param difficulty - Bot difficulty level
 * @param timeBudgetMs - Time budget in milliseconds
 * @returns Best move column (0-6)
 */
export function suggestMove(
  board: Board,
  currentPlayer: Player,
  difficulty: DifficultyLevel,
  timeBudgetMs: number
): number {
  const config = DIFFICULTY_CONFIGS[difficulty]
  const validMoves = getValidMoves(board)

  if (validMoves.length === 0) {
    throw new Error('No valid moves available')
  }

  // If only one move, return it immediately
  if (validMoves.length === 1) {
    return validMoves[0]
  }

  const deadline = Date.now() + timeBudgetMs * 0.9 // Leave 10% buffer
  let bestMove = validMoves[Math.floor(validMoves.length / 2)] // Start with center-ish
  let bestScore = -Infinity

  // Iterative deepening: search progressively deeper until time runs out
  for (let depth = 1; depth <= config.searchDepth; depth++) {
    if (Date.now() > deadline) break

    const result = minimax(
      board,
      depth,
      -Infinity,
      Infinity,
      true,
      currentPlayer,
      currentPlayer,
      deadline
    )

    if (result.move !== null) {
      bestMove = result.move
      bestScore = result.score
    }

    // Stop early if we found a forced win
    if (Math.abs(bestScore) >= EVAL_WEIGHTS.WIN) break
  }

  // Introduce random errors based on difficulty
  if (config.errorRate > 0 && Math.random() < config.errorRate) {
    const randomIndex = Math.floor(Math.random() * validMoves.length)
    return validMoves[randomIndex]
  }

  return bestMove
}

/**
 * Get the time elapsed during the bot's turn for deduction.
 */
export function measureMoveTime<T>(fn: () => T): { result: T; elapsedMs: number } {
  const start = Date.now()
  const result = fn()
  const elapsedMs = Date.now() - start
  return { result, elapsedMs }
}

// ============================================================================
// ENGINE-BASED MOVE SUGGESTION
// ============================================================================

/**
 * Convert difficulty level to engine configuration.
 */
export function difficultyToEngineConfig(
  difficulty: DifficultyLevel,
  engineType: EngineType = 'minimax',
  customParams?: Record<string, unknown>
): EngineConfig {
  const diffConfig = DIFFICULTY_CONFIGS[difficulty]
  const defaultConfig = DEFAULT_ENGINE_CONFIGS[engineType]

  return {
    searchDepth: diffConfig.searchDepth,
    errorRate: diffConfig.errorRate,
    ...defaultConfig,
    customParams,
  }
}

/**
 * Suggest a move using the specified AI engine.
 *
 * This is the preferred API for new code. It supports pluggable engines
 * and returns detailed move information including confidence and search stats.
 *
 * @param board - Current board state
 * @param currentPlayer - Player to move (the bot)
 * @param config - Bot persona configuration with engine selection
 * @param timeBudgetMs - Time budget in milliseconds
 * @returns Promise resolving to move result with metadata
 */
export async function suggestMoveWithEngine(
  board: Board,
  currentPlayer: Player,
  config: BotPersonaConfig,
  timeBudgetMs: number
): Promise<MoveResult> {
  const engineType = config.engine ?? 'minimax'
  const engine = engineRegistry.getWithFallback(engineType)
  const engineConfig = difficultyToEngineConfig(
    config.difficulty,
    engineType,
    config.customEngineParams
  )

  return engine.selectMove(board, currentPlayer, engineConfig, timeBudgetMs)
}

/**
 * Get the engine registry for managing AI engines.
 */
export function getEngineRegistry() {
  return engineRegistry
}

/**
 * List available AI engines.
 */
export function listEngines() {
  return engineRegistry.list()
}

// Re-export types for convenience
export type { EngineConfig, MoveResult, AIEngine, EngineType }
