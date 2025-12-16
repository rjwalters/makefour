/**
 * ThreatPairs Engine
 *
 * Implements 2swap's threat pairs / combinatoric wins strategy.
 *
 * Key concepts:
 * - Major Threat (T): 3 pieces in a row + 1 empty = playing there wins
 * - Minor Threat (t): 2 pieces in a row + 2 empty = can become major threat
 * - Combinatoric Win: Two simultaneous threats opponent can't block both
 * - Stacked Threats: Threats directly above each other (the "7" shape)
 *
 * Strategy:
 * - Look for moves that create two major threats simultaneously
 * - Prioritize stacked threats (vertical pairs)
 * - Defensively identify and block opponent's minor threats
 *
 * From 2swap: "Identifying these positions is a learned skill"
 *
 * References:
 * - 2swap YouTube: "Threat Pairs" / "Combinatoric Wins" video
 */

import type { AIEngine, EngineConfig, MoveResult } from '../ai-engine'
import type { Board, Player } from '../game'
import { getValidMoves, applyMove, checkWinner, ROWS, COLUMNS } from '../game'

// ============================================================================
// THREAT TYPES
// ============================================================================

interface MajorThreat {
  player: Player
  row: number
  column: number
  direction: 'horizontal' | 'vertical' | 'diagonal-up' | 'diagonal-down'
}

interface MinorThreat {
  player: Player
  emptyPositions: Array<{ row: number; column: number }>
  direction: 'horizontal' | 'vertical' | 'diagonal-up' | 'diagonal-down'
}

// ============================================================================
// THREAT DETECTION
// ============================================================================

function getAvailableRow(board: Board, column: number): number {
  for (let row = ROWS - 1; row >= 0; row--) {
    if (board[row][column] === null) {
      return row
    }
  }
  return -1
}

function checkMajorThreatPattern(
  board: Board,
  player: Player,
  startRow: number,
  startCol: number,
  dRow: number,
  dCol: number,
  direction: MajorThreat['direction']
): MajorThreat | null {
  let playerCount = 0
  let emptyPos: { row: number; col: number } | null = null

  for (let i = 0; i < 4; i++) {
    const r = startRow + i * dRow
    const c = startCol + i * dCol

    if (r < 0 || r >= ROWS || c < 0 || c >= COLUMNS) return null

    const cell = board[r][c]

    if (cell === player) {
      playerCount++
    } else if (cell === null) {
      if (emptyPos === null) {
        emptyPos = { row: r, col: c }
      } else {
        return null
      }
    } else {
      return null
    }
  }

  if (playerCount === 3 && emptyPos !== null) {
    return {
      player,
      row: emptyPos.row,
      column: emptyPos.col,
      direction,
    }
  }

  return null
}

function findMajorThreats(board: Board, player: Player): MajorThreat[] {
  const threats: MajorThreat[] = []

  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLUMNS; col++) {
      if (col + 3 < COLUMNS) {
        const threat = checkMajorThreatPattern(board, player, row, col, 0, 1, 'horizontal')
        if (threat) threats.push(threat)
      }
      if (row + 3 < ROWS) {
        const threat = checkMajorThreatPattern(board, player, row, col, 1, 0, 'vertical')
        if (threat) threats.push(threat)
      }
      if (row + 3 < ROWS && col + 3 < COLUMNS) {
        const threat = checkMajorThreatPattern(board, player, row, col, 1, 1, 'diagonal-down')
        if (threat) threats.push(threat)
      }
      if (row - 3 >= 0 && col + 3 < COLUMNS) {
        const threat = checkMajorThreatPattern(board, player, row, col, -1, 1, 'diagonal-up')
        if (threat) threats.push(threat)
      }
    }
  }

  return threats
}

function checkMinorThreatPattern(
  board: Board,
  player: Player,
  startRow: number,
  startCol: number,
  dRow: number,
  dCol: number,
  direction: MinorThreat['direction']
): MinorThreat | null {
  let playerCount = 0
  const emptyPositions: Array<{ row: number; column: number }> = []

  for (let i = 0; i < 4; i++) {
    const r = startRow + i * dRow
    const c = startCol + i * dCol

    if (r < 0 || r >= ROWS || c < 0 || c >= COLUMNS) return null

    const cell = board[r][c]

    if (cell === player) {
      playerCount++
    } else if (cell === null) {
      emptyPositions.push({ row: r, column: c })
    } else {
      return null
    }
  }

  if (playerCount === 2 && emptyPositions.length === 2) {
    return { player, emptyPositions, direction }
  }

  return null
}

function findMinorThreats(board: Board, player: Player): MinorThreat[] {
  const threats: MinorThreat[] = []

  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLUMNS; col++) {
      if (col + 3 < COLUMNS) {
        const threat = checkMinorThreatPattern(board, player, row, col, 0, 1, 'horizontal')
        if (threat) threats.push(threat)
      }
      if (row + 3 < ROWS) {
        const threat = checkMinorThreatPattern(board, player, row, col, 1, 0, 'vertical')
        if (threat) threats.push(threat)
      }
      if (row + 3 < ROWS && col + 3 < COLUMNS) {
        const threat = checkMinorThreatPattern(board, player, row, col, 1, 1, 'diagonal-down')
        if (threat) threats.push(threat)
      }
      if (row - 3 >= 0 && col + 3 < COLUMNS) {
        const threat = checkMinorThreatPattern(board, player, row, col, -1, 1, 'diagonal-up')
        if (threat) threats.push(threat)
      }
    }
  }

  return threats
}

function findStackedThreats(threats: MajorThreat[]): Array<[MajorThreat, MajorThreat]> {
  const pairs: Array<[MajorThreat, MajorThreat]> = []

  for (let i = 0; i < threats.length; i++) {
    for (let j = i + 1; j < threats.length; j++) {
      const t1 = threats[i]
      const t2 = threats[j]

      if (t1.column === t2.column && Math.abs(t1.row - t2.row) === 1) {
        pairs.push([t1, t2])
      }
    }
  }

  return pairs
}

// ============================================================================
// MOVE SELECTION
// ============================================================================

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

function findDoubleThreatMove(board: Board, player: Player): number | null {
  const validMoves = getValidMoves(board)

  for (const col of validMoves) {
    const result = applyMove(board, col, player)
    if (!result.success || !result.board) continue

    const threats = findMajorThreats(result.board, player)

    if (threats.length >= 2) {
      const stackedPairs = findStackedThreats(threats)
      if (stackedPairs.length > 0) {
        return col
      }
      return col
    }
  }

  return null
}

function findBlockDoubleThreatMove(board: Board, player: Player): number | null {
  const opponent: Player = player === 1 ? 2 : 1
  const validMoves = getValidMoves(board)

  for (const col of validMoves) {
    const oppResult = applyMove(board, col, opponent)
    if (!oppResult.success || !oppResult.board) continue

    const oppThreats = findMajorThreats(oppResult.board, opponent)

    if (oppThreats.length >= 2) {
      return col
    }
  }

  const minorThreats = findMinorThreats(board, opponent)
  for (const threat of minorThreats) {
    for (const emptyPos of threat.emptyPositions) {
      const availableRow = getAvailableRow(board, emptyPos.column)
      if (availableRow === emptyPos.row && validMoves.includes(emptyPos.column)) {
        return emptyPos.column
      }
    }
  }

  return null
}

function findCreateThreatMove(board: Board, player: Player): number | null {
  const validMoves = getValidMoves(board)
  const existingThreats = findMajorThreats(board, player)

  for (const col of validMoves) {
    const result = applyMove(board, col, player)
    if (!result.success || !result.board) continue

    const newThreats = findMajorThreats(result.board, player)
    const createdThreats = newThreats.filter(
      (nt) => !existingThreats.some((et) => et.row === nt.row && et.column === nt.column)
    )

    if (createdThreats.length > 0) {
      return col
    }
  }

  return null
}

function selectCenterBiasedMove(validMoves: number[]): number {
  const centerCol = Math.floor(COLUMNS / 2)
  const sorted = [...validMoves].sort(
    (a, b) => Math.abs(a - centerCol) - Math.abs(b - centerCol)
  )
  return sorted[0]
}

function selectThreatPairsMove(board: Board, player: Player): number {
  const validMoves = getValidMoves(board)

  if (validMoves.length === 0) {
    throw new Error('No valid moves available')
  }

  if (validMoves.length === 1) {
    return validMoves[0]
  }

  // Priority 1: Win
  const winningMove = findWinningMove(board, player)
  if (winningMove !== null) return winningMove

  // Priority 2: Block opponent win
  const opponent: Player = player === 1 ? 2 : 1
  const blockingMove = findWinningMove(board, opponent)
  if (blockingMove !== null) return blockingMove

  // Priority 3: Create double threat
  const doubleTheatMove = findDoubleThreatMove(board, player)
  if (doubleTheatMove !== null) return doubleTheatMove

  // Priority 4: Block opponent's double threat
  const blockDoubleMove = findBlockDoubleThreatMove(board, player)
  if (blockDoubleMove !== null) return blockDoubleMove

  // Priority 5: Create any threat
  const threatMove = findCreateThreatMove(board, player)
  if (threatMove !== null) return threatMove

  // Fallback: center-biased
  return selectCenterBiasedMove(validMoves)
}

// ============================================================================
// ENGINE IMPLEMENTATION
// ============================================================================

export const threatPairsEngine: AIEngine = {
  name: 'threat-pairs',
  description: "2swap's threat pairs strategy - creates combinatoric wins through double threats",

  async selectMove(
    board: Board,
    player: Player,
    config: EngineConfig,
    _timeBudget: number
  ): Promise<MoveResult> {
    const startTime = Date.now()

    // Apply error rate
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

    const move = selectThreatPairsMove(board, player)

    // Calculate confidence based on threat situation
    const myThreats = findMajorThreats(board, player)
    const confidence = Math.min(0.95, 0.5 + myThreats.length * 0.15)

    return {
      column: move,
      confidence,
      searchInfo: {
        depth: 2,
        nodesSearched: 21,
        timeUsed: Date.now() - startTime,
      },
    }
  },

  evaluatePosition(board: Board, player: Player): number {
    const myThreats = findMajorThreats(board, player)
    const opponent: Player = player === 1 ? 2 : 1
    const oppThreats = findMajorThreats(board, opponent)

    let score = 0

    // Major threats are very valuable
    score += myThreats.length * 100

    // Stacked threats are even more valuable
    const myStackedPairs = findStackedThreats(myThreats)
    score += myStackedPairs.length * 500

    // Penalize opponent threats
    score -= oppThreats.length * 100
    const oppStackedPairs = findStackedThreats(oppThreats)
    score -= oppStackedPairs.length * 500

    return score
  },

  explainMove(board: Board, move: number, player: Player): string {
    const result = applyMove(board, move, player)
    if (!result.success || !result.board) {
      return `ThreatPairs: Playing column ${move + 1}`
    }

    const threats = findMajorThreats(result.board, player)
    if (threats.length >= 2) {
      const stackedPairs = findStackedThreats(threats)
      if (stackedPairs.length > 0) {
        return `ThreatPairs: Creating STACKED double threat in column ${move + 1} - combinatoric win!`
      }
      return `ThreatPairs: Creating double threat with column ${move + 1} - opponent can't block both!`
    }

    if (threats.length === 1) {
      return `ThreatPairs: Creating major threat in column ${move + 1}`
    }

    return `ThreatPairs: Playing column ${move + 1} to build position`
  },
}
