/**
 * Parity Engine
 *
 * Implements 2swap's parity strategy as a pluggable AI engine.
 *
 * Key concepts:
 * - Player 1 (Red) wants threats on ODD rows from bottom (1, 3, 5)
 * - Player 2 (Yellow) wants threats on EVEN rows from bottom (2, 4, 6)
 * - Lower threats "undercut" higher threats in the same column
 * - The player whose threat is lowest (and on their favored parity) wins
 *
 * From 2swap: "Once you are sufficiently far into the game where all threats
 * have been developed, you can ALWAYS predict the result of the game totally
 * deterministically, ONLY by knowing the list of open threats and the parity
 * of the rows which they lie on."
 *
 * References:
 * - 2swap YouTube: "Parity" video
 * - https://2swap.github.io/WeakC4/explanation/
 */

import type { AIEngine, EngineConfig, MoveResult } from '../ai-engine'
import type { Board, Player } from '../game'
import { getValidMoves, applyMove, checkWinner, ROWS, COLUMNS, WIN_LENGTH } from '../game'

// ============================================================================
// THREAT DETECTION
// ============================================================================

/**
 * Represents a threat (3 pieces in a row with 1 empty space to complete).
 */
interface Threat {
  player: Player
  row: number
  column: number
  rowFromBottom: number
  isOddRow: boolean
}

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
 * Checks a 4-cell pattern for a threat (3 player pieces + 1 empty).
 */
function checkThreatPattern(
  board: Board,
  player: Player,
  startRow: number,
  startCol: number,
  dRow: number,
  dCol: number
): Threat | null {
  let playerCount = 0
  let emptyPos: { row: number; col: number } | null = null

  for (let i = 0; i < 4; i++) {
    const r = startRow + i * dRow
    const c = startCol + i * dCol
    const cell = board[r][c]

    if (cell === player) {
      playerCount++
    } else if (cell === null) {
      if (emptyPos === null) {
        emptyPos = { row: r, col: c }
      } else {
        return null // More than one empty, not a threat
      }
    } else {
      return null // Opponent piece blocks this pattern
    }
  }

  if (playerCount === 3 && emptyPos !== null) {
    const rowFromBottom = ROWS - emptyPos.row
    return {
      player,
      row: emptyPos.row,
      column: emptyPos.col,
      rowFromBottom,
      isOddRow: rowFromBottom % 2 === 1,
    }
  }

  return null
}

/**
 * Finds all threats (3-in-a-row with one empty completion spot) on the board.
 */
function findThreats(board: Board, player: Player): Threat[] {
  const threats: Threat[] = []

  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLUMNS; col++) {
      // Horizontal
      if (col + 3 < COLUMNS) {
        const threat = checkThreatPattern(board, player, row, col, 0, 1)
        if (threat) threats.push(threat)
      }

      // Vertical
      if (row + 3 < ROWS) {
        const threat = checkThreatPattern(board, player, row, col, 1, 0)
        if (threat) threats.push(threat)
      }

      // Diagonal down-right
      if (row + 3 < ROWS && col + 3 < COLUMNS) {
        const threat = checkThreatPattern(board, player, row, col, 1, 1)
        if (threat) threats.push(threat)
      }

      // Diagonal down-left
      if (row + 3 < ROWS && col - 3 >= 0) {
        const threat = checkThreatPattern(board, player, row, col, 1, -1)
        if (threat) threats.push(threat)
      }
    }
  }

  return threats
}

/**
 * Checks if a position can currently be played (is the next available row).
 */
function isPlayablePosition(board: Board, row: number, col: number): boolean {
  const availableRow = getAvailableRow(board, col)
  return availableRow === row
}

// ============================================================================
// MOVE SELECTION
// ============================================================================

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
 * Finds a move that creates a threat on the player's favored parity.
 */
function findParityMove(board: Board, player: Player): number | null {
  const validMoves = getValidMoves(board)
  const favoredParity = player === 1 // true = odd rows for player 1

  const scoredMoves: Array<{ col: number; score: number }> = []

  for (const col of validMoves) {
    const result = applyMove(board, col, player)
    if (!result.success || !result.board) continue

    // Find threats created by this move
    const newThreats = findThreats(result.board, player)
    const existingThreats = findThreats(board, player)

    // Threats newly created by this move
    const createdThreats = newThreats.filter(
      (nt) => !existingThreats.some((et) => et.row === nt.row && et.column === nt.column)
    )

    // Score based on threats on favored parity
    let score = 0
    for (const threat of createdThreats) {
      if (threat.isOddRow === favoredParity) {
        // Threat on favored parity - higher score for lower rows
        score += 10 + threat.rowFromBottom
      } else {
        // Threat on wrong parity - still worth something but less
        score += 2
      }
    }

    // Bonus for lower row placements
    const placementRow = getAvailableRow(board, col)
    if (placementRow !== -1) {
      const placementRowFromBottom = ROWS - placementRow
      if ((placementRowFromBottom % 2 === 1) === favoredParity) {
        score += 1
      }
    }

    if (score > 0) {
      scoredMoves.push({ col, score })
    }
  }

  if (scoredMoves.length === 0) {
    return null
  }

  scoredMoves.sort((a, b) => b.score - a.score)
  return scoredMoves[0].col
}

/**
 * Finds a move that blocks opponent's threats on their favored parity.
 */
function findBlockParityMove(board: Board, player: Player): number | null {
  const opponent: Player = player === 1 ? 2 : 1
  const opponentFavoredParity = opponent === 1

  const opponentThreats = findThreats(board, opponent)

  // Find the most dangerous threat (lowest row on opponent's favored parity)
  const dangerousThreats = opponentThreats
    .filter((t) => t.isOddRow === opponentFavoredParity)
    .sort((a, b) => b.rowFromBottom - a.rowFromBottom)

  for (const threat of dangerousThreats) {
    // Check if we can play at the threat's completion point
    if (isPlayablePosition(board, threat.row, threat.column)) {
      return threat.column
    }

    // Check if we can play below the threat
    const availableRow = getAvailableRow(board, threat.column)
    if (availableRow !== -1 && availableRow > threat.row) {
      const result = applyMove(board, threat.column, player)
      if (result.success && result.board) {
        const ourThreats = findThreats(result.board, player)
        if (ourThreats.length > 0) {
          return threat.column
        }
      }
    }
  }

  return null
}

/**
 * Core parity move selection logic.
 */
function selectParityMove(board: Board, player: Player): number {
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

  // Priority 3: Create threats on favored parity rows
  const parityMove = findParityMove(board, player)
  if (parityMove !== null) {
    return parityMove
  }

  // Priority 4: Block opponent's threats on their favored parity
  const blockParityMove = findBlockParityMove(board, player)
  if (blockParityMove !== null) {
    return blockParityMove
  }

  // Fallback: center-biased
  return selectCenterBiasedMove(validMoves)
}

// ============================================================================
// ENGINE IMPLEMENTATION
// ============================================================================

/**
 * Parity Engine - 2swap's parity-based threat strategy.
 */
export const parityEngine: AIEngine = {
  name: 'parity',
  description: "2swap's parity strategy - prioritizes threats on favored rows (odd for Red, even for Yellow)",

  async selectMove(
    board: Board,
    player: Player,
    config: EngineConfig,
    _timeBudget: number
  ): Promise<MoveResult> {
    const startTime = Date.now()

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

    const move = selectParityMove(board, player)

    // Calculate confidence based on threat analysis
    const myThreats = findThreats(board, player)
    const favoredParity = player === 1
    const goodThreats = myThreats.filter((t) => t.isOddRow === favoredParity)
    const confidence = Math.min(0.95, 0.6 + goodThreats.length * 0.1)

    return {
      column: move,
      confidence,
      searchInfo: {
        depth: 1,
        nodesSearched: 14, // Checked all positions for threats
        timeUsed: Date.now() - startTime,
      },
    }
  },

  evaluatePosition(board: Board, player: Player): number {
    const myThreats = findThreats(board, player)
    const opponent: Player = player === 1 ? 2 : 1
    const oppThreats = findThreats(board, opponent)

    const favoredParity = player === 1
    const oppFavoredParity = opponent === 1

    let score = 0

    // Score my threats (favored parity worth more, lower rows worth more)
    for (const threat of myThreats) {
      if (threat.isOddRow === favoredParity) {
        score += 100 + threat.rowFromBottom * 10
      } else {
        score += 20
      }
    }

    // Penalize opponent threats
    for (const threat of oppThreats) {
      if (threat.isOddRow === oppFavoredParity) {
        score -= 100 + threat.rowFromBottom * 10
      } else {
        score -= 20
      }
    }

    return score
  },

  explainMove(board: Board, move: number, player: Player): string {
    const favoredParity = player === 1 ? 'odd' : 'even'
    const row = getAvailableRow(board, move)
    const rowFromBottom = ROWS - row
    const isOnFavoredParity = (rowFromBottom % 2 === 1) === (player === 1)

    return `Parity: Playing column ${move + 1} (row ${rowFromBottom} from bottom) - ${
      isOnFavoredParity ? 'on favored ' + favoredParity + ' row' : 'setting up threats'
    }`
  },
}
