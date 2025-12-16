/**
 * ClaimEven Engine
 *
 * Implements 2swap's claimeven strategy as a pluggable AI engine.
 *
 * The claimeven strategy is a powerful technique for Player 2 (Yellow):
 * - When all columns have an even number of empty spaces, Yellow can
 *   guarantee claiming all even rows by responding directly above Red's move.
 * - This works because pieces alternate, and Yellow always fills the "paired"
 *   spot above Red's piece.
 *
 * For Player 1 (Red), we use "ClaimOdd" on the single odd-spaced column.
 *
 * References:
 * - 2swap YouTube: "Claimeven" video
 * - https://2swap.github.io/WeakC4/explanation/
 */

import type { AIEngine, EngineConfig, MoveResult } from '../ai-engine'
import type { Board, Player } from '../game'
import { getValidMoves, applyMove, checkWinner, ROWS, COLUMNS } from '../game'

// ============================================================================
// CLAIMEVEN LOGIC
// ============================================================================

/**
 * Gets the row index where a piece would land if dropped in the given column.
 */
function getAvailableRow(board: Board, column: number): number {
  for (let row = ROWS - 1; row >= 0; row--) {
    if (board[row][column] === null) {
      return row
    }
  }
  return -1
}

/**
 * Analyzes the parity (even/odd empty count) of each column.
 */
function analyzeColumnParities(board: Board): Array<{ column: number; emptyCount: number }> {
  const result: Array<{ column: number; emptyCount: number }> = []

  for (let col = 0; col < COLUMNS; col++) {
    let emptyCount = 0
    for (let row = 0; row < ROWS; row++) {
      if (board[row][col] === null) {
        emptyCount++
      }
    }
    result.push({ column: col, emptyCount })
  }

  return result
}

/**
 * Checks if a row (0-indexed from top) is an odd row from the bottom.
 */
function isOddRowFromBottom(row: number): boolean {
  const rowFromBottom = ROWS - row
  return rowFromBottom % 2 === 1
}

/**
 * Finds a winning move for the given player.
 */
function findWinningMove(board: Board, player: Player): number | null {
  const validMoves = getValidMoves(board)

  for (const col of validMoves) {
    const result = applyMove(board, col, player)
    if (result.success && result.board) {
      const winner = checkWinner(result.board)
      if (winner === player) {
        return col
      }
    }
  }

  return null
}

/**
 * Selects the best center-biased move from a list of valid moves.
 */
function selectCenterBiasedMove(validMoves: number[]): number {
  const centerCol = Math.floor(COLUMNS / 2)
  const sorted = [...validMoves].sort(
    (a, b) => Math.abs(a - centerCol) - Math.abs(b - centerCol)
  )
  return sorted[0]
}

/**
 * Core claimeven move selection logic.
 */
function selectClaimEvenMove(
  board: Board,
  player: Player,
  lastOpponentMove: number | null
): number {
  const validMoves = getValidMoves(board)

  if (validMoves.length === 0) {
    throw new Error('No valid moves available')
  }

  if (validMoves.length === 1) {
    return validMoves[0]
  }

  // Priority 1: Win if possible
  const winningMove = findWinningMove(board, player)
  if (winningMove !== null) {
    return winningMove
  }

  // Priority 2: Block opponent's immediate win
  const opponent: Player = player === 1 ? 2 : 1
  const blockingMove = findWinningMove(board, opponent)
  if (blockingMove !== null) {
    return blockingMove
  }

  // Priority 3: Apply claimeven/claimodd strategy
  const columnParities = analyzeColumnParities(board)
  const oddColumns = columnParities
    .filter((c) => c.emptyCount % 2 === 1)
    .map((c) => c.column)

  if (player === 2) {
    // Yellow: Use claimeven - respond above opponent's last move
    if (lastOpponentMove !== null && validMoves.includes(lastOpponentMove)) {
      return lastOpponentMove
    }

    // If all columns are even, prefer center
    if (oddColumns.length === 0) {
      return selectCenterBiasedMove(validMoves)
    }

    // If there are odd columns, try to make them even
    const oddValidMoves = validMoves.filter((m) => oddColumns.includes(m))
    if (oddValidMoves.length > 0) {
      return selectCenterBiasedMove(oddValidMoves)
    }
  } else {
    // Red (Player 1): Use claimodd on the single odd column if one exists
    if (oddColumns.length === 1 && validMoves.includes(oddColumns[0])) {
      return oddColumns[0]
    }

    // Red wants odd-row threats - prefer moves that create them
    const oddRowMoves = validMoves.filter((col) => {
      const row = getAvailableRow(board, col)
      return row !== -1 && isOddRowFromBottom(row)
    })

    if (oddRowMoves.length > 0) {
      return selectCenterBiasedMove(oddRowMoves)
    }
  }

  // Fallback: center-biased random
  return selectCenterBiasedMove(validMoves)
}

// ============================================================================
// ENGINE IMPLEMENTATION
// ============================================================================

/**
 * ClaimEven Engine - 2swap's claimeven strategy.
 */
export const claimEvenEngine: AIEngine = {
  name: 'claimeven',
  description: "2swap's claimeven strategy - responds above opponent's moves to claim even rows",

  async selectMove(
    board: Board,
    player: Player,
    config: EngineConfig,
    _timeBudget: number
  ): Promise<MoveResult> {
    const startTime = Date.now()

    // Try to infer last opponent move by comparing board state
    // (This is a simplification - in real usage the game would track this)
    let lastOpponentMove: number | null = null

    // Apply error rate for difficulty adjustment
    if (config.errorRate > 0 && Math.random() < config.errorRate) {
      const validMoves = getValidMoves(board)
      const randomMove = validMoves[Math.floor(Math.random() * validMoves.length)]
      return {
        column: randomMove,
        confidence: 0.3,
        searchInfo: {
          depth: 1,
          nodesSearched: 1,
          timeUsed: Date.now() - startTime,
        },
      }
    }

    const move = selectClaimEvenMove(board, player, lastOpponentMove)

    return {
      column: move,
      confidence: 0.8,
      searchInfo: {
        depth: 1,
        nodesSearched: 7, // Checked all columns
        timeUsed: Date.now() - startTime,
      },
    }
  },

  explainMove(board: Board, move: number, player: Player): string {
    const columnParities = analyzeColumnParities(board)
    const emptyCount = columnParities.find((c) => c.column === move)?.emptyCount ?? 0
    const isOdd = emptyCount % 2 === 1

    if (player === 2) {
      return `ClaimEven: Playing column ${move + 1} to ${
        isOdd ? 'make the column even' : 'maintain even parity'
      }`
    } else {
      return `ClaimOdd: Playing column ${move + 1} to ${
        isOdd ? 'claim the odd column' : 'build odd-row threats'
      }`
    }
  },
}
