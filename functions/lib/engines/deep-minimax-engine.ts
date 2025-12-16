/**
 * Deep Minimax Engine
 *
 * A highly optimized minimax implementation with transposition tables
 * for the "Oracle" persona. Designed to play near-perfectly by searching
 * to maximum depth with efficient position caching.
 */

import type { AIEngine, EngineConfig, MoveResult, SearchInfo } from '../ai-engine'
import {
  type Board,
  type Player,
  applyMove,
  checkWinner,
  ROWS,
  COLUMNS,
  WIN_LENGTH,
} from '../game'

// ============================================================================
// TRANSPOSITION TABLE
// ============================================================================

/**
 * Entry types for transposition table bounds.
 */
enum TTEntryType {
  EXACT = 0, // Exact score
  LOWER = 1, // Score is a lower bound (beta cutoff)
  UPPER = 2, // Score is an upper bound (alpha cutoff)
}

/**
 * Transposition table entry.
 */
interface TTEntry {
  depth: number
  score: number
  type: TTEntryType
  bestMove: number | null
}

/**
 * Transposition table for storing evaluated positions.
 * Uses string-based hashing for reliability (not Zobrist hashing).
 */
class TranspositionTable {
  private table: Map<string, TTEntry> = new Map()
  private maxSize: number
  private hits = 0
  private misses = 0

  constructor(maxSize = 1000000) {
    this.maxSize = maxSize
  }

  /**
   * Generate a hash key for a board position.
   * Uses a simple string representation for reliability.
   */
  private hashBoard(board: Board): string {
    return board.map((row) => row.map((c) => (c === null ? '0' : c)).join('')).join('|')
  }

  /**
   * Store a position in the transposition table.
   */
  store(board: Board, depth: number, score: number, type: TTEntryType, bestMove: number | null): void {
    // Evict old entries if table is too large
    if (this.table.size >= this.maxSize) {
      // Simple eviction: clear half the table
      const entries = Array.from(this.table.entries())
      this.table.clear()
      for (let i = 0; i < entries.length / 2; i++) {
        this.table.set(entries[i][0], entries[i][1])
      }
    }

    const key = this.hashBoard(board)
    const existing = this.table.get(key)

    // Only overwrite if new entry has deeper or equal depth
    if (!existing || existing.depth <= depth) {
      this.table.set(key, { depth, score, type, bestMove })
    }
  }

  /**
   * Lookup a position in the transposition table.
   */
  lookup(board: Board, depth: number, alpha: number, beta: number): { score: number; bestMove: number | null } | null {
    const key = this.hashBoard(board)
    const entry = this.table.get(key)

    if (!entry) {
      this.misses++
      return null
    }

    // Only use if stored depth is sufficient
    if (entry.depth < depth) {
      this.misses++
      return null
    }

    this.hits++

    switch (entry.type) {
      case TTEntryType.EXACT:
        return { score: entry.score, bestMove: entry.bestMove }
      case TTEntryType.LOWER:
        if (entry.score >= beta) {
          return { score: entry.score, bestMove: entry.bestMove }
        }
        break
      case TTEntryType.UPPER:
        if (entry.score <= alpha) {
          return { score: entry.score, bestMove: entry.bestMove }
        }
        break
    }

    return null
  }

  /**
   * Get the best move from a stored position (for move ordering).
   */
  getBestMove(board: Board): number | null {
    const key = this.hashBoard(board)
    return this.table.get(key)?.bestMove ?? null
  }

  /**
   * Clear the transposition table.
   */
  clear(): void {
    this.table.clear()
    this.hits = 0
    this.misses = 0
  }

  /**
   * Get statistics about table usage.
   */
  getStats(): { size: number; hits: number; misses: number; hitRate: number } {
    const total = this.hits + this.misses
    return {
      size: this.table.size,
      hits: this.hits,
      misses: this.misses,
      hitRate: total > 0 ? this.hits / total : 0,
    }
  }
}

// ============================================================================
// EVALUATION (BALANCED)
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
// QUIESCENCE SEARCH - THREAT DETECTION
// ============================================================================

/**
 * Maximum additional depth for quiescence search.
 * Limits search explosion while ensuring tactical accuracy.
 */
const MAX_QUIESCENCE_DEPTH = 4

/**
 * Find columns where a player can complete a winning threat.
 * Returns columns where playing would result in 4-in-a-row.
 */
function findWinningMoves(board: Board, player: Player): number[] {
  const winningMoves: number[] = []

  for (let col = 0; col < COLUMNS; col++) {
    if (board[0][col] !== null) continue // Column full

    const result = applyMove(board, col, player)
    if (result.success && result.board) {
      const winner = checkWinner(result.board)
      if (winner === player) {
        winningMoves.push(col)
      }
    }
  }

  return winningMoves
}

/**
 * Check if a position is "unstable" - has immediate tactical threats.
 * An unstable position has winning moves available for either player.
 */
function isPositionUnstable(board: Board, currentPlayer: Player): boolean {
  const opponent: Player = currentPlayer === 1 ? 2 : 1

  // Check if current player can win immediately
  if (findWinningMoves(board, currentPlayer).length > 0) {
    return true
  }

  // Check if opponent has winning threats that must be blocked
  if (findWinningMoves(board, opponent).length > 0) {
    return true
  }

  return false
}

/**
 * Get "loud" moves - moves that are tactically significant.
 * In Connect Four, these are winning moves and blocking moves.
 */
function getLoudMoves(board: Board, currentPlayer: Player): number[] {
  const opponent: Player = currentPlayer === 1 ? 2 : 1
  const loudMoves = new Set<number>()

  // Winning moves for current player
  for (const col of findWinningMoves(board, currentPlayer)) {
    loudMoves.add(col)
  }

  // Blocking moves (opponent's winning moves)
  for (const col of findWinningMoves(board, opponent)) {
    loudMoves.add(col)
  }

  return Array.from(loudMoves)
}

/**
 * Quiescence search - continue searching in unstable positions.
 * Only considers "loud" moves (winning/blocking) to avoid search explosion.
 */
function quiescenceSearch(
  board: Board,
  alpha: number,
  beta: number,
  maximizingPlayer: boolean,
  player: Player,
  currentPlayer: Player,
  deadline: number,
  nodesSearched: number,
  qDepth: number
): { score: number; nodesSearched: number } {
  nodesSearched++

  // Check time limit
  if (nodesSearched % 500 === 0 && Date.now() > deadline) {
    return { score: evaluatePosition(board, player), nodesSearched }
  }

  // Check terminal states
  const winner = checkWinner(board)
  if (winner !== null) {
    if (winner === 'draw') return { score: 0, nodesSearched }
    const winScore = EVAL_WEIGHTS.WIN + qDepth * 100
    return {
      score: winner === player ? winScore : -winScore,
      nodesSearched,
    }
  }

  // Stand-pat: get static evaluation as baseline
  const standPat = evaluatePosition(board, player)

  // If we've reached max quiescence depth, return static eval
  if (qDepth <= 0) {
    return { score: standPat, nodesSearched }
  }

  // Check if position is quiet (no immediate threats)
  if (!isPositionUnstable(board, currentPlayer)) {
    return { score: standPat, nodesSearched }
  }

  // Get only loud moves (winning/blocking)
  const loudMoves = getLoudMoves(board, currentPlayer)
  if (loudMoves.length === 0) {
    return { score: standPat, nodesSearched }
  }

  const nextPlayer: Player = currentPlayer === 1 ? 2 : 1

  if (maximizingPlayer) {
    let bestScore = standPat // Can always "stand pat"

    if (bestScore >= beta) {
      return { score: bestScore, nodesSearched }
    }
    alpha = Math.max(alpha, bestScore)

    for (const move of loudMoves) {
      const result = applyMove(board, move, currentPlayer)
      if (!result.success || !result.board) continue

      const qResult = quiescenceSearch(
        result.board,
        alpha,
        beta,
        false,
        player,
        nextPlayer,
        deadline,
        nodesSearched,
        qDepth - 1
      )
      nodesSearched = qResult.nodesSearched

      if (qResult.score > bestScore) {
        bestScore = qResult.score
      }

      alpha = Math.max(alpha, bestScore)
      if (beta <= alpha) break
    }

    return { score: bestScore, nodesSearched }
  } else {
    let bestScore = standPat

    if (bestScore <= alpha) {
      return { score: bestScore, nodesSearched }
    }
    beta = Math.min(beta, bestScore)

    for (const move of loudMoves) {
      const result = applyMove(board, move, currentPlayer)
      if (!result.success || !result.board) continue

      const qResult = quiescenceSearch(
        result.board,
        alpha,
        beta,
        true,
        player,
        nextPlayer,
        deadline,
        nodesSearched,
        qDepth - 1
      )
      nodesSearched = qResult.nodesSearched

      if (qResult.score < bestScore) {
        bestScore = qResult.score
      }

      beta = Math.min(beta, bestScore)
      if (beta <= alpha) break
    }

    return { score: bestScore, nodesSearched }
  }
}

// ============================================================================
// MOVE GENERATION WITH ORDERING
// ============================================================================

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
 * Order moves for better alpha-beta pruning.
 * Prioritizes: 1) TT best move, 2) Center columns, 3) Winning/blocking moves
 */
function orderMoves(
  board: Board,
  moves: number[],
  player: Player,
  ttBestMove: number | null
): number[] {
  const opponent: Player = player === 1 ? 2 : 1
  const centerCol = Math.floor(COLUMNS / 2)

  const scored = moves.map((move) => {
    let score = 0

    // TT best move gets highest priority
    if (move === ttBestMove) {
      score += 10000
    }

    // Winning moves
    const result = applyMove(board, move, player)
    if (result.success && result.board) {
      const winner = checkWinner(result.board)
      if (winner === player) {
        score += 5000
      }
    }

    // Blocking moves
    const oppResult = applyMove(board, move, opponent)
    if (oppResult.success && oppResult.board) {
      const winner = checkWinner(oppResult.board)
      if (winner === opponent) {
        score += 4000
      }
    }

    // Center preference
    score += (COLUMNS - Math.abs(move - centerCol)) * 10

    return { move, score }
  })

  return scored.sort((a, b) => b.score - a.score).map((s) => s.move)
}

// ============================================================================
// DEEP MINIMAX WITH TRANSPOSITION TABLE
// ============================================================================

interface SearchResult {
  score: number
  move: number | null
  nodesSearched: number
}

function deepMinimax(
  board: Board,
  depth: number,
  alpha: number,
  beta: number,
  maximizingPlayer: boolean,
  player: Player,
  currentPlayer: Player,
  deadline: number,
  nodesSearched: number,
  tt: TranspositionTable
): SearchResult {
  nodesSearched++

  // Check time limit (but not too frequently - every 1000 nodes)
  if (nodesSearched % 1000 === 0 && Date.now() > deadline) {
    return { score: evaluatePosition(board, player), move: null, nodesSearched }
  }

  // Check terminal states
  const winner = checkWinner(board)
  if (winner !== null) {
    if (winner === 'draw') return { score: 0, move: null, nodesSearched }
    const winScore = EVAL_WEIGHTS.WIN + depth * 100
    return {
      score: winner === player ? winScore : -winScore,
      move: null,
      nodesSearched,
    }
  }

  const validMoves = getValidMoves(board)
  if (validMoves.length === 0) return { score: 0, move: null, nodesSearched }

  // Depth limit reached - use quiescence search for tactical accuracy
  if (depth === 0) {
    const qResult = quiescenceSearch(
      board,
      alpha,
      beta,
      maximizingPlayer,
      player,
      currentPlayer,
      deadline,
      nodesSearched,
      MAX_QUIESCENCE_DEPTH
    )
    return { score: qResult.score, move: null, nodesSearched: qResult.nodesSearched }
  }

  // Transposition table lookup
  const ttResult = tt.lookup(board, depth, alpha, beta)
  if (ttResult !== null) {
    return { score: ttResult.score, move: ttResult.bestMove, nodesSearched }
  }

  // Get TT best move for move ordering
  const ttBestMove = tt.getBestMove(board)
  const orderedMoves = orderMoves(board, validMoves, currentPlayer, ttBestMove)
  const nextPlayer: Player = currentPlayer === 1 ? 2 : 1

  let bestMove = orderedMoves[0]
  let bestScore: number
  let entryType: TTEntryType

  if (maximizingPlayer) {
    bestScore = -Infinity
    entryType = TTEntryType.UPPER

    for (const move of orderedMoves) {
      const result = applyMove(board, move, currentPlayer)
      if (!result.success || !result.board) continue

      const searchResult = deepMinimax(
        result.board,
        depth - 1,
        alpha,
        beta,
        false,
        player,
        nextPlayer,
        deadline,
        nodesSearched,
        tt
      )
      nodesSearched = searchResult.nodesSearched

      if (searchResult.score > bestScore) {
        bestScore = searchResult.score
        bestMove = move
      }

      if (bestScore > alpha) {
        alpha = bestScore
        entryType = TTEntryType.EXACT
      }

      if (beta <= alpha) {
        entryType = TTEntryType.LOWER
        break
      }
    }
  } else {
    bestScore = Infinity
    entryType = TTEntryType.UPPER

    for (const move of orderedMoves) {
      const result = applyMove(board, move, currentPlayer)
      if (!result.success || !result.board) continue

      const searchResult = deepMinimax(
        result.board,
        depth - 1,
        alpha,
        beta,
        true,
        player,
        nextPlayer,
        deadline,
        nodesSearched,
        tt
      )
      nodesSearched = searchResult.nodesSearched

      if (searchResult.score < bestScore) {
        bestScore = searchResult.score
        bestMove = move
      }

      if (bestScore < beta) {
        beta = bestScore
        entryType = TTEntryType.EXACT
      }

      if (beta <= alpha) {
        entryType = TTEntryType.LOWER
        break
      }
    }
  }

  // Store in transposition table
  tt.store(board, depth, bestScore, entryType, bestMove)

  return { score: bestScore, move: bestMove, nodesSearched }
}

// ============================================================================
// DEEP MINIMAX ENGINE
// ============================================================================

/**
 * Deep minimax engine for the Oracle persona.
 *
 * Characteristics:
 * - Maximum search depth (aims to solve the game)
 * - Transposition table for efficient position caching
 * - Advanced move ordering for better pruning
 * - Iterative deepening with time management
 */
export class DeepMinimaxEngine implements AIEngine {
  readonly name = 'deep-minimax'
  readonly description =
    'Deep minimax with transposition tables for near-perfect play (for Oracle)'

  private tt: TranspositionTable

  constructor(ttSize = 1000000) {
    this.tt = new TranspositionTable(ttSize)
  }

  async selectMove(
    board: Board,
    player: Player,
    config: EngineConfig,
    timeBudget: number
  ): Promise<MoveResult> {
    const startTime = Date.now()
    const validMoves = getValidMoves(board)

    if (validMoves.length === 0) {
      throw new Error('No valid moves available')
    }

    // If only one move, return it immediately
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

    // Use transposition table if enabled (default for this engine)
    const useTranspositionTable = config.customParams?.useTranspositionTable !== false

    if (!useTranspositionTable) {
      // Clear TT if disabled for this search
      this.tt.clear()
    }

    const deadline = startTime + timeBudget * 0.95 // Tight buffer for deep searches
    let bestMove = validMoves[Math.floor(validMoves.length / 2)]
    let bestScore = -Infinity
    let totalNodesSearched = 0
    let depthReached = 0
    const principalVariation: number[] = []

    // Iterative deepening - essential for time management
    for (let depth = 1; depth <= config.searchDepth; depth++) {
      if (Date.now() > deadline) break

      const result = deepMinimax(
        board,
        depth,
        -Infinity,
        Infinity,
        true,
        player,
        player,
        deadline,
        0,
        this.tt
      )

      totalNodesSearched += result.nodesSearched

      if (result.move !== null) {
        bestMove = result.move
        bestScore = result.score
        depthReached = depth

        // Track principal variation
        if (principalVariation.length === 0 || principalVariation[0] !== result.move) {
          principalVariation.length = 0
          principalVariation.push(result.move)
        }
      }

      // Stop early if we found a forced win/loss
      if (Math.abs(bestScore) >= EVAL_WEIGHTS.WIN) break
    }

    // Oracle never makes random errors (errorRate should be 0)
    if (config.errorRate > 0 && Math.random() < config.errorRate) {
      const randomIndex = Math.floor(Math.random() * validMoves.length)
      return {
        column: validMoves[randomIndex],
        confidence: 0,
        searchInfo: {
          depth: depthReached,
          nodesSearched: totalNodesSearched,
          timeUsed: Date.now() - startTime,
          principalVariation,
        },
      }
    }

    // Calculate confidence - Oracle should have high confidence
    const confidence = Math.min(
      1,
      Math.max(0, (bestScore + EVAL_WEIGHTS.WIN) / (2 * EVAL_WEIGHTS.WIN))
    )

    const ttStats = this.tt.getStats()

    return {
      column: bestMove,
      confidence,
      searchInfo: {
        depth: depthReached,
        nodesSearched: totalNodesSearched,
        timeUsed: Date.now() - startTime,
        principalVariation,
        // Include TT stats in custom data
        ...(useTranspositionTable && {
          ttHitRate: ttStats.hitRate,
          ttSize: ttStats.size,
        }),
      } as SearchInfo,
    }
  }

  evaluatePosition(board: Board, player: Player): number {
    return evaluatePosition(board, player)
  }

  explainMove(board: Board, move: number, player: Player): string {
    const result = applyMove(board, move, player)
    if (!result.success || !result.board) {
      return `Invalid move: column ${move}`
    }

    const winner = checkWinner(result.board)
    if (winner === player) {
      return `Column ${move} is the winning move. I foresaw this outcome.`
    }

    // Check if it blocks opponent's win
    const opponent: Player = player === 1 ? 2 : 1
    const opponentResult = applyMove(board, move, opponent)
    if (opponentResult.success && opponentResult.board) {
      const opponentWin = checkWinner(opponentResult.board)
      if (opponentWin === opponent) {
        return `Column ${move} blocks the only path to defeat. The game continues.`
      }
    }

    const centerCol = Math.floor(COLUMNS / 2)
    if (move === centerCol) {
      return `Column ${move} - the center holds strategic significance in the grand design.`
    }

    return `Column ${move} - the optimal path forward, as I have calculated.`
  }

  /**
   * Clear the transposition table.
   * Useful between games or when memory is constrained.
   */
  clearTable(): void {
    this.tt.clear()
  }

  /**
   * Get transposition table statistics.
   */
  getTableStats(): { size: number; hits: number; misses: number; hitRate: number } {
    return this.tt.getStats()
  }
}

// Export singleton instance
export const deepMinimaxEngine = new DeepMinimaxEngine()
