/**
 * Neural Network Engine
 *
 * ONNX-based inference for pattern-learned play.
 * Supports three modes:
 * - Pure policy network (sample from probability distribution)
 * - Value network + 1-ply search
 * - Hybrid (policy suggests candidates, minimax evaluates)
 *
 * When no model is loaded, uses pattern-based heuristics to simulate
 * neural-like behavior for testing and fallback.
 */

import type { AIEngine, EngineConfig, MoveResult, SearchInfo } from '../ai-engine'
import {
  type Board,
  type Player,
  applyMove,
  checkWinner,
  ROWS,
  COLUMNS,
} from '../game'

// ============================================================================
// NEURAL ENGINE CONFIGURATION
// ============================================================================

/**
 * Configuration specific to the neural engine.
 * Passed via EngineConfig.customParams.
 */
export interface NeuralConfig {
  /** Path to ONNX model file (null for simulated mode) */
  modelPath: string | null
  /** Temperature for sampling from policy distribution (0 = greedy, higher = more random) */
  temperature: number
  /** Whether to use hybrid search combining policy with minimax */
  useHybridSearch: boolean
  /** Depth to search in hybrid mode */
  hybridDepth: number
}

export const DEFAULT_NEURAL_CONFIG: NeuralConfig = {
  modelPath: null, // No model = simulated mode
  temperature: 0.5,
  useHybridSearch: true,
  hybridDepth: 3,
}

// ============================================================================
// MODEL INFERENCE ABSTRACTION
// ============================================================================

/**
 * Output from the neural network model.
 */
interface ModelOutput {
  /** Policy: probability distribution over columns (length 7) */
  policy: number[]
  /** Value: position evaluation from -1 (losing) to 1 (winning) */
  value?: number
}

/**
 * Abstract interface for model inference.
 * Allows swapping between ONNX runtime and simulated inference.
 */
interface ModelInference {
  isLoaded(): boolean
  predict(boardInput: Float32Array): Promise<ModelOutput>
}

// ============================================================================
// SIMULATED NEURAL INFERENCE
// ============================================================================

/**
 * Pattern-based inference that simulates neural network behavior.
 * Uses heuristics to generate policy-like probability distributions.
 * This allows testing the neural engine architecture without a trained model.
 */
class SimulatedInference implements ModelInference {
  isLoaded(): boolean {
    // Always "loaded" since it's simulated
    return true
  }

  async predict(boardInput: Float32Array): Promise<ModelOutput> {
    // Decode board from input tensor
    const board = this.decodeBoardFromInput(boardInput)

    // Generate policy using pattern recognition heuristics
    const policy = this.generatePolicy(board)

    // Generate value estimate
    const value = this.estimateValue(board)

    return { policy, value }
  }

  private decodeBoardFromInput(input: Float32Array): Board {
    // Input is one-hot encoded: [6*7*2] = [row][col][player]
    // First 42 values: player 1 positions
    // Next 42 values: player 2 positions
    const board: Board = Array.from({ length: ROWS }, () =>
      Array(COLUMNS).fill(null)
    )

    for (let row = 0; row < ROWS; row++) {
      for (let col = 0; col < COLUMNS; col++) {
        const idx = row * COLUMNS + col
        if (input[idx] === 1) {
          board[row][col] = 1
        } else if (input[idx + 42] === 1) {
          board[row][col] = 2
        }
      }
    }

    return board
  }

  private generatePolicy(board: Board): number[] {
    const weights: number[] = Array(COLUMNS).fill(0)
    const validMoves = this.getValidMoves(board)

    if (validMoves.length === 0) {
      return Array(COLUMNS).fill(1 / COLUMNS)
    }

    // Base weights for valid moves
    for (const col of validMoves) {
      weights[col] = 1
    }

    // Pattern recognition: center preference
    const centerCol = Math.floor(COLUMNS / 2)
    for (const col of validMoves) {
      const distFromCenter = Math.abs(col - centerCol)
      weights[col] += (3 - distFromCenter) * 0.3
    }

    // Pattern: immediate winning moves
    const player = this.inferCurrentPlayer(board)
    for (const col of validMoves) {
      const result = applyMove(board, col, player)
      if (result.success && result.board) {
        const winner = checkWinner(result.board)
        if (winner === player) {
          weights[col] += 10 // Strong preference for winning moves
        }
      }
    }

    // Pattern: blocking opponent wins
    const opponent: Player = player === 1 ? 2 : 1
    for (const col of validMoves) {
      const result = applyMove(board, col, opponent)
      if (result.success && result.board) {
        const winner = checkWinner(result.board)
        if (winner === opponent) {
          weights[col] += 8 // High preference for blocking
        }
      }
    }

    // Pattern: threat creation (3-in-a-row with open end)
    for (const col of validMoves) {
      const threatScore = this.evaluateThreats(board, col, player)
      weights[col] += threatScore * 0.5
    }

    // Pattern: avoid giving opponent winning setup
    for (const col of validMoves) {
      const result = applyMove(board, col, player)
      if (result.success && result.board) {
        // Check if this creates a winning opportunity for opponent
        for (const opCol of this.getValidMoves(result.board)) {
          const opResult = applyMove(result.board, opCol, opponent)
          if (opResult.success && opResult.board) {
            const winner = checkWinner(opResult.board)
            if (winner === opponent) {
              weights[col] -= 3 // Penalize moves that enable opponent win
            }
          }
        }
      }
    }

    // Normalize to probability distribution
    const sum = weights.reduce((a, b) => Math.max(0, a) + Math.max(0, b), 0)
    if (sum === 0) {
      // All weights non-positive, return uniform over valid moves
      const policy = Array(COLUMNS).fill(0)
      for (const col of validMoves) {
        policy[col] = 1 / validMoves.length
      }
      return policy
    }

    return weights.map((w) => Math.max(0, w) / sum)
  }

  private estimateValue(board: Board): number {
    const player = this.inferCurrentPlayer(board)
    const opponent: Player = player === 1 ? 2 : 1
    const validMoves = this.getValidMoves(board)

    if (validMoves.length === 0) return 0

    // Check immediate game end
    for (const col of validMoves) {
      const result = applyMove(board, col, player)
      if (result.success && result.board) {
        if (checkWinner(result.board) === player) return 0.9
      }
    }

    for (const col of validMoves) {
      const result = applyMove(board, col, opponent)
      if (result.success && result.board) {
        if (checkWinner(result.board) === opponent) return -0.7
      }
    }

    // Simple piece count and position evaluation
    let score = 0
    const centerCol = Math.floor(COLUMNS / 2)

    for (let row = 0; row < ROWS; row++) {
      for (let col = 0; col < COLUMNS; col++) {
        if (board[row][col] === player) {
          score += 0.02 + (3 - Math.abs(col - centerCol)) * 0.01
        } else if (board[row][col] === opponent) {
          score -= 0.02 + (3 - Math.abs(col - centerCol)) * 0.01
        }
      }
    }

    return Math.max(-1, Math.min(1, score))
  }

  private evaluateThreats(board: Board, col: number, player: Player): number {
    const result = applyMove(board, col, player)
    if (!result.success || !result.board) return 0

    let threats = 0
    const newBoard = result.board

    // Check for 3-in-a-row patterns
    const directions = [
      [0, 1], // horizontal
      [1, 0], // vertical
      [1, 1], // diagonal down-right
      [1, -1], // diagonal down-left
    ]

    for (let row = 0; row < ROWS; row++) {
      for (let startCol = 0; startCol < COLUMNS; startCol++) {
        for (const [dr, dc] of directions) {
          let playerCount = 0
          let emptyCount = 0
          let valid = true

          for (let i = 0; i < 4; i++) {
            const r = row + i * dr
            const c = startCol + i * dc

            if (r < 0 || r >= ROWS || c < 0 || c >= COLUMNS) {
              valid = false
              break
            }

            const cell = newBoard[r][c]
            if (cell === player) playerCount++
            else if (cell === null) emptyCount++
            else {
              valid = false
              break
            }
          }

          if (valid && playerCount === 3 && emptyCount === 1) {
            threats++
          }
        }
      }
    }

    return threats
  }

  private getValidMoves(board: Board): number[] {
    const moves: number[] = []
    for (let col = 0; col < COLUMNS; col++) {
      if (board[0][col] === null) moves.push(col)
    }
    return moves
  }

  private inferCurrentPlayer(board: Board): Player {
    let p1Count = 0
    let p2Count = 0

    for (let row = 0; row < ROWS; row++) {
      for (let col = 0; col < COLUMNS; col++) {
        if (board[row][col] === 1) p1Count++
        else if (board[row][col] === 2) p2Count++
      }
    }

    return p1Count <= p2Count ? 1 : 2
  }
}

// ============================================================================
// ONNX MODEL INFERENCE (PLACEHOLDER)
// ============================================================================

/**
 * ONNX-based model inference using WASM runtime.
 * This is a placeholder for actual ONNX integration.
 *
 * For Cloudflare Workers, consider using:
 * - workers-wonnx: https://github.com/cloudflare/workers-wonnx
 * - onnxruntime-web with WASM backend
 */
class ONNXInference implements ModelInference {
  private loaded = false
  private modelPath: string

  constructor(modelPath: string) {
    this.modelPath = modelPath
  }

  isLoaded(): boolean {
    return this.loaded
  }

  async load(): Promise<void> {
    // TODO: Implement ONNX model loading
    // 1. Fetch model file from modelPath
    // 2. Initialize ONNX runtime session
    // 3. Validate input/output shapes

    console.warn(`ONNX model loading not yet implemented: ${this.modelPath}`)
    this.loaded = false
  }

  async predict(boardInput: Float32Array): Promise<ModelOutput> {
    if (!this.loaded) {
      throw new Error('ONNX model not loaded')
    }

    // TODO: Implement ONNX inference
    // 1. Create input tensor from boardInput
    // 2. Run inference session
    // 3. Extract policy and value outputs
    // 4. Return ModelOutput

    // Placeholder - should never reach here if loaded check works
    return {
      policy: Array(COLUMNS).fill(1 / COLUMNS),
      value: 0,
    }
  }
}

// ============================================================================
// BOARD ENCODING
// ============================================================================

/**
 * Encodes board state as a float tensor for neural network input.
 * Uses one-hot encoding: [6*7*2] where first 42 = player 1, next 42 = player 2.
 */
function encodeBoardForModel(board: Board, player: Player): Float32Array {
  const input = new Float32Array(ROWS * COLUMNS * 2)

  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLUMNS; col++) {
      const idx = row * COLUMNS + col
      const cell = board[row][col]

      if (cell === player) {
        input[idx] = 1 // Current player's pieces in first channel
      } else if (cell !== null) {
        input[idx + 42] = 1 // Opponent's pieces in second channel
      }
    }
  }

  return input
}

// ============================================================================
// MINIMAX HELPERS (for hybrid mode)
// ============================================================================

const EVAL_WEIGHTS = {
  WIN: 100000,
  THREE_IN_ROW: 100,
  TWO_IN_ROW: 10,
  CENTER_CONTROL: 3,
}

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
    for (let col = 0; col <= COLUMNS - 4; col++) {
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
    for (let row = 0; row <= ROWS - 4; row++) {
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
  for (let row = 0; row <= ROWS - 4; row++) {
    for (let col = 0; col <= COLUMNS - 4; col++) {
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
  for (let row = 0; row <= ROWS - 4; row++) {
    for (let col = 3; col < COLUMNS; col++) {
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

interface MinimaxResult {
  score: number
  move: number | null
}

function minimaxSearch(
  board: Board,
  depth: number,
  alpha: number,
  beta: number,
  maximizing: boolean,
  player: Player,
  currentPlayer: Player,
  deadline: number
): MinimaxResult {
  if (Date.now() > deadline) {
    return { score: evaluatePosition(board, player), move: null }
  }

  const winner = checkWinner(board)
  if (winner !== null) {
    if (winner === 'draw') return { score: 0, move: null }
    return {
      score: winner === player ? EVAL_WEIGHTS.WIN + depth * 100 : -EVAL_WEIGHTS.WIN - depth * 100,
      move: null,
    }
  }

  const validMoves: number[] = []
  for (let col = 0; col < COLUMNS; col++) {
    if (board[0][col] === null) validMoves.push(col)
  }

  if (validMoves.length === 0 || depth === 0) {
    return { score: evaluatePosition(board, player), move: null }
  }

  // Order by center preference
  const centerCol = Math.floor(COLUMNS / 2)
  validMoves.sort((a, b) => Math.abs(a - centerCol) - Math.abs(b - centerCol))

  const nextPlayer: Player = currentPlayer === 1 ? 2 : 1

  if (maximizing) {
    let maxScore = -Infinity
    let bestMove = validMoves[0]

    for (const move of validMoves) {
      const result = applyMove(board, move, currentPlayer)
      if (!result.success || !result.board) continue

      const searchResult = minimaxSearch(
        result.board,
        depth - 1,
        alpha,
        beta,
        false,
        player,
        nextPlayer,
        deadline
      )

      if (searchResult.score > maxScore) {
        maxScore = searchResult.score
        bestMove = move
      }

      alpha = Math.max(alpha, searchResult.score)
      if (beta <= alpha) break
    }

    return { score: maxScore, move: bestMove }
  } else {
    let minScore = Infinity
    let bestMove = validMoves[0]

    for (const move of validMoves) {
      const result = applyMove(board, move, currentPlayer)
      if (!result.success || !result.board) continue

      const searchResult = minimaxSearch(
        result.board,
        depth - 1,
        alpha,
        beta,
        true,
        player,
        nextPlayer,
        deadline
      )

      if (searchResult.score < minScore) {
        minScore = searchResult.score
        bestMove = move
      }

      beta = Math.min(beta, searchResult.score)
      if (beta <= alpha) break
    }

    return { score: minScore, move: bestMove }
  }
}

// ============================================================================
// NEURAL ENGINE
// ============================================================================

/**
 * Neural network engine for Connect Four.
 *
 * Plays by "feel" using learned patterns rather than explicit search.
 * Supports pure policy mode, value network + 1-ply, and hybrid approaches.
 */
export class NeuralEngine implements AIEngine {
  readonly name = 'neural'
  readonly description = 'Neural network inference with optional hybrid search'

  private inference: ModelInference
  private onnxInference: ONNXInference | null = null
  private config: NeuralConfig

  constructor(config: Partial<NeuralConfig> = {}) {
    this.config = { ...DEFAULT_NEURAL_CONFIG, ...config }

    if (this.config.modelPath) {
      this.onnxInference = new ONNXInference(this.config.modelPath)
      this.inference = this.onnxInference
    } else {
      // Use simulated inference when no model is provided
      this.inference = new SimulatedInference()
    }
  }

  /**
   * Load the ONNX model if configured.
   * Call this before using the engine with a real model.
   */
  async loadModel(): Promise<boolean> {
    if (this.onnxInference) {
      try {
        await this.onnxInference.load()
        return this.onnxInference.isLoaded()
      } catch (error) {
        console.error('Failed to load ONNX model:', error)
        // Fall back to simulated inference
        this.inference = new SimulatedInference()
        return false
      }
    }
    return true // Simulated inference is always ready
  }

  /**
   * Check if the engine is ready for inference.
   */
  isReady(): boolean {
    return this.inference.isLoaded()
  }

  async selectMove(
    board: Board,
    player: Player,
    config: EngineConfig,
    timeBudget: number
  ): Promise<MoveResult> {
    const startTime = Date.now()
    const deadline = startTime + timeBudget * 0.9

    // Get valid moves
    const validMoves: number[] = []
    for (let col = 0; col < COLUMNS; col++) {
      if (board[0][col] === null) validMoves.push(col)
    }

    if (validMoves.length === 0) {
      throw new Error('No valid moves available')
    }

    // Single valid move - return immediately
    if (validMoves.length === 1) {
      return {
        column: validMoves[0],
        confidence: 1,
        searchInfo: {
          depth: 0,
          nodesSearched: 1,
          timeUsed: Date.now() - startTime,
        },
      }
    }

    // Merge custom params with defaults
    const neuralConfig: NeuralConfig = {
      ...this.config,
      ...(config.customParams as Partial<NeuralConfig>),
    }

    // Get neural network policy
    const boardInput = encodeBoardForModel(board, player)
    const modelOutput = await this.inference.predict(boardInput)
    const policy = modelOutput.policy

    // Filter policy to valid moves only
    const validPolicy: Array<{ col: number; prob: number }> = validMoves
      .map((col) => ({ col, prob: policy[col] }))
      .filter((p) => p.prob > 0)

    if (validPolicy.length === 0) {
      // Fallback: uniform distribution over valid moves
      validPolicy.push(
        ...validMoves.map((col) => ({ col, prob: 1 / validMoves.length }))
      )
    }

    // Normalize probabilities
    const totalProb = validPolicy.reduce((sum, p) => sum + p.prob, 0)
    for (const p of validPolicy) {
      p.prob /= totalProb
    }

    // Sort by probability (descending)
    validPolicy.sort((a, b) => b.prob - a.prob)

    let selectedMove: number
    let confidence: number
    let searchInfo: SearchInfo

    if (neuralConfig.useHybridSearch) {
      // Hybrid mode: take top 3 candidates, evaluate with minimax
      const topCandidates = validPolicy.slice(0, Math.min(3, validPolicy.length))
      let bestScore = -Infinity
      let bestMove = topCandidates[0].col

      for (const candidate of topCandidates) {
        if (Date.now() > deadline) break

        const result = applyMove(board, candidate.col, player)
        if (!result.success || !result.board) continue

        // Quick minimax evaluation
        const nextPlayer: Player = player === 1 ? 2 : 1
        const searchResult = minimaxSearch(
          result.board,
          neuralConfig.hybridDepth,
          -Infinity,
          Infinity,
          false, // Next player minimizes
          player,
          nextPlayer,
          deadline
        )

        // Combine policy probability with search score
        const combinedScore = searchResult.score * 0.7 + candidate.prob * EVAL_WEIGHTS.THREE_IN_ROW * 0.3

        if (combinedScore > bestScore) {
          bestScore = combinedScore
          bestMove = candidate.col
        }
      }

      selectedMove = bestMove
      confidence = validPolicy.find((p) => p.col === bestMove)?.prob ?? 0.5
      searchInfo = {
        depth: neuralConfig.hybridDepth,
        nodesSearched: topCandidates.length,
        timeUsed: Date.now() - startTime,
      }
    } else {
      // Pure policy mode: sample from distribution with temperature
      if (neuralConfig.temperature <= 0.01) {
        // Greedy: pick highest probability
        selectedMove = validPolicy[0].col
        confidence = validPolicy[0].prob
      } else {
        // Temperature-adjusted sampling
        const tempAdjusted = validPolicy.map((p) => ({
          col: p.col,
          prob: p.prob ** (1 / neuralConfig.temperature),
        }))
        const tempSum = tempAdjusted.reduce((sum, p) => sum + p.prob, 0)
        for (const p of tempAdjusted) {
          p.prob /= tempSum
        }

        // Sample from distribution
        const rand = Math.random()
        let cumulative = 0
        selectedMove = tempAdjusted[tempAdjusted.length - 1].col

        for (const p of tempAdjusted) {
          cumulative += p.prob
          if (rand < cumulative) {
            selectedMove = p.col
            break
          }
        }

        confidence = validPolicy.find((p) => p.col === selectedMove)?.prob ?? 0.5
      }

      searchInfo = {
        depth: 0,
        nodesSearched: 1,
        timeUsed: Date.now() - startTime,
      }
    }

    // Apply error rate
    if (config.errorRate > 0 && Math.random() < config.errorRate) {
      const randomIndex = Math.floor(Math.random() * validMoves.length)
      return {
        column: validMoves[randomIndex],
        confidence: 0,
        searchInfo: {
          ...searchInfo,
          timeUsed: Date.now() - startTime,
        },
      }
    }

    return {
      column: selectedMove,
      confidence,
      searchInfo,
    }
  }

  evaluatePosition(board: Board, player: Player): number {
    // Use minimax evaluation for consistency
    return evaluatePosition(board, player)
  }

  explainMove(board: Board, move: number, player: Player): string {
    const result = applyMove(board, move, player)
    if (!result.success || !result.board) {
      return `Invalid move: column ${move}`
    }

    const winner = checkWinner(result.board)
    if (winner === player) {
      return `Pattern match: winning sequence detected in column ${move}`
    }

    // Check if it blocks opponent's win
    const opponent: Player = player === 1 ? 2 : 1
    const opponentResult = applyMove(board, move, opponent)
    if (opponentResult.success && opponentResult.board) {
      const opponentWin = checkWinner(opponentResult.board)
      if (opponentWin === opponent) {
        return `Pattern recognition: blocking threat in column ${move}`
      }
    }

    const centerCol = Math.floor(COLUMNS / 2)
    const distFromCenter = Math.abs(move - centerCol)

    if (distFromCenter === 0) {
      return `Intuition: center column ${move} feels strong`
    } else if (distFromCenter <= 1) {
      return `Pattern sense: column ${move} looks promising`
    }

    return `Playing column ${move} by feel`
  }
}

// Export singleton instance with default config
export const neuralEngine = new NeuralEngine()
