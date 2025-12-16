/**
 * Heuristic Engines for MakeFour
 *
 * Lightweight, rule-based AI engines that don't use tree search.
 * These create distinct play styles and are computationally cheap (<1ms per move).
 *
 * Engines:
 * - RandomEngine: For Rusty bot at very low skill levels
 * - ThreatHeuristicEngine: For Sentinel bot with priority-based move selection
 */

import {
  type Board,
  type Player,
  getValidMoves,
  getAvailableRow,
  applyMove,
  checkWinner,
  COLUMNS,
  ROWS,
} from '../game/makefour'
import { analyzeThreats } from './coach'

// ============================================================================
// CONFIGURATION
// ============================================================================

/**
 * Configuration options for heuristic engines.
 */
export interface HeuristicConfig {
  /**
   * Probability of following optimal heuristic vs random (0.0 - 1.0).
   * - 0.0 = always random
   * - 1.0 = always follow heuristic
   */
  accuracy: number

  /**
   * Whether to look ahead for "trap" setups (two-way threats).
   * When enabled, the engine will try to create positions where
   * it has multiple winning moves.
   */
  detectTraps: boolean

  /**
   * Bias weights for column selection.
   * Higher values make center columns more attractive.
   */
  centerBias: number
}

/**
 * Default configuration for heuristic engines.
 */
export const DEFAULT_HEURISTIC_CONFIG: HeuristicConfig = {
  accuracy: 0.8,
  detectTraps: false,
  centerBias: 1.5,
}

// ============================================================================
// ENGINE INTERFACE
// ============================================================================

/**
 * Interface for heuristic engines.
 */
export interface HeuristicEngine {
  /** Unique identifier for the engine */
  readonly id: string

  /** Human-readable name */
  readonly name: string

  /** Description of the engine's play style */
  readonly description: string

  /**
   * Selects a move for the given position.
   * @param board - Current board state
   * @param player - Player to move (1 or 2)
   * @returns Column index (0-6) for the selected move
   */
  selectMove(board: Board, player: Player): number

  /**
   * Gets the current configuration.
   */
  getConfig(): HeuristicConfig
}

// ============================================================================
// RANDOM ENGINE
// ============================================================================

/**
 * Random Engine for Rusty at very low skill levels.
 *
 * Characteristics:
 * - Picks randomly from valid moves
 * - Slight center bias (controlled by config)
 * - Used for "I'm learning too!" personality
 * - Makes obvious mistakes, misses wins
 */
export class RandomEngine implements HeuristicEngine {
  readonly id = 'random'
  readonly name = 'Random Engine'
  readonly description = 'Picks randomly from valid moves with slight center bias'

  private config: HeuristicConfig

  constructor(config: Partial<HeuristicConfig> = {}) {
    this.config = { ...DEFAULT_HEURISTIC_CONFIG, ...config }
  }

  getConfig(): HeuristicConfig {
    return { ...this.config }
  }

  selectMove(board: Board, _player: Player): number {
    const validMoves = getValidMoves(board)

    if (validMoves.length === 0) {
      throw new Error('No valid moves available')
    }

    if (validMoves.length === 1) {
      return validMoves[0]
    }

    // Apply center bias weights
    const weights = validMoves.map((col) => this.getCenterWeight(col))
    return this.weightedRandomSelect(validMoves, weights)
  }

  /**
   * Gets the weight for a column based on center bias.
   * Center column (3) has the highest weight.
   */
  private getCenterWeight(column: number): number {
    const centerCol = Math.floor(COLUMNS / 2) // 3
    const distanceFromCenter = Math.abs(column - centerCol)

    // Weight decreases as distance from center increases
    // center = 1.0 * centerBias, edges = 1.0
    const centerMultiplier = 1 + (1 - distanceFromCenter / centerCol) * (this.config.centerBias - 1)
    return Math.max(centerMultiplier, 1)
  }

  /**
   * Selects a random element from array with weighted probabilities.
   */
  private weightedRandomSelect(items: number[], weights: number[]): number {
    const totalWeight = weights.reduce((sum, w) => sum + w, 0)
    let random = Math.random() * totalWeight

    for (let i = 0; i < items.length; i++) {
      random -= weights[i]
      if (random <= 0) {
        return items[i]
      }
    }

    // Fallback to last item (shouldn't happen)
    return items[items.length - 1]
  }
}

// ============================================================================
// THREAT HEURISTIC ENGINE
// ============================================================================

/**
 * Threat Heuristic Engine for Sentinel bot.
 *
 * Priority-based move selection:
 * 1. Win if possible (complete own 4-in-a-row)
 * 2. Block opponent's immediate win
 * 3. Block opponent's 3-in-a-row with open end (if detectTraps enabled)
 * 4. Create own 3-in-a-row (if detectTraps enabled)
 * 5. Prefer center column
 * 6. Random from remaining
 *
 * Characteristics:
 * - High accuracy = rarely loses to tactics, but doesn't create threats
 * - Defensive focus
 * - Predictable but characterful play
 */
export class ThreatHeuristicEngine implements HeuristicEngine {
  readonly id = 'threat-heuristic'
  readonly name = 'Threat Heuristic Engine'
  readonly description = 'Priority-based defensive engine that focuses on blocking threats'

  private config: HeuristicConfig

  constructor(config: Partial<HeuristicConfig> = {}) {
    this.config = { ...DEFAULT_HEURISTIC_CONFIG, ...config }
  }

  getConfig(): HeuristicConfig {
    return { ...this.config }
  }

  selectMove(board: Board, player: Player): number {
    const validMoves = getValidMoves(board)

    if (validMoves.length === 0) {
      throw new Error('No valid moves available')
    }

    if (validMoves.length === 1) {
      return validMoves[0]
    }

    // Check if we should follow heuristic or play randomly
    if (Math.random() > this.config.accuracy) {
      return this.selectRandomMove(validMoves)
    }

    // Priority 1: Win if possible
    const winningMove = this.findWinningMove(board, player)
    if (winningMove !== null) {
      return winningMove
    }

    // Priority 2: Block opponent's immediate win
    const opponent: Player = player === 1 ? 2 : 1
    const blockingMove = this.findWinningMove(board, opponent)
    if (blockingMove !== null) {
      return blockingMove
    }

    // Priority 3 & 4: Handle threats (if detectTraps enabled)
    if (this.config.detectTraps) {
      // Block opponent's 3-in-a-row with open end
      const blockThreeMove = this.findBlockThreeMove(board, player)
      if (blockThreeMove !== null) {
        return blockThreeMove
      }

      // Create own 3-in-a-row
      const createThreeMove = this.findCreateThreeMove(board, player)
      if (createThreeMove !== null) {
        return createThreeMove
      }

      // Look for trap setups (double threats)
      const trapMove = this.findTrapMove(board, player)
      if (trapMove !== null) {
        return trapMove
      }
    }

    // Priority 5: Prefer center column
    const centerCol = Math.floor(COLUMNS / 2)
    if (validMoves.includes(centerCol)) {
      return centerCol
    }

    // Priority 6: Random from remaining (with center bias)
    return this.selectCenterBiasedMove(validMoves)
  }

  /**
   * Finds a winning move for the given player.
   * @returns Column index if winning move exists, null otherwise
   */
  private findWinningMove(board: Board, player: Player): number | null {
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
   * Finds a move that blocks opponent's 3-in-a-row with an open end.
   * Uses the threat analysis from coach.ts.
   */
  private findBlockThreeMove(board: Board, player: Player): number | null {
    const opponent: Player = player === 1 ? 2 : 1

    // Check each valid move to see if opponent would have multiple threats after
    const validMoves = getValidMoves(board)

    for (const col of validMoves) {
      // Simulate our move
      const afterOurMove = applyMove(board, col, player)
      if (!afterOurMove.success || !afterOurMove.board) continue

      // Count opponent's potential threats
      const potentialThreats = this.countPotentialThreats(afterOurMove.board, opponent)

      // If this move significantly reduces opponent's threats, consider it
      const currentThreats = this.countPotentialThreats(board, opponent)
      if (potentialThreats < currentThreats) {
        return col
      }
    }

    return null
  }

  /**
   * Finds a move that creates a 3-in-a-row for us.
   */
  private findCreateThreeMove(board: Board, player: Player): number | null {
    const validMoves = getValidMoves(board)

    for (const col of validMoves) {
      const result = applyMove(board, col, player)
      if (!result.success || !result.board) continue

      // Check if this creates a threat
      const threats = analyzeThreats(result.board, player)
      if (threats.winningMoves.length > 0) {
        return col
      }
    }

    return null
  }

  /**
   * Finds a move that creates a double threat (trap).
   * A trap is when we can win in two different ways.
   */
  private findTrapMove(board: Board, player: Player): number | null {
    const validMoves = getValidMoves(board)

    for (const col of validMoves) {
      const result = applyMove(board, col, player)
      if (!result.success || !result.board) continue

      // Check if after our move, we have multiple winning options
      const threats = analyzeThreats(result.board, player)
      if (threats.winningMoves.length >= 2) {
        return col // This creates a winning trap!
      }
    }

    return null
  }

  /**
   * Counts potential threats (3-in-a-row patterns) for a player.
   */
  private countPotentialThreats(board: Board, player: Player): number {
    const threats = analyzeThreats(board, player)
    return threats.winningMoves.length
  }

  /**
   * Selects a random move from the given list.
   */
  private selectRandomMove(validMoves: number[]): number {
    const index = Math.floor(Math.random() * validMoves.length)
    return validMoves[index]
  }

  /**
   * Selects a move with bias toward center columns.
   */
  private selectCenterBiasedMove(validMoves: number[]): number {
    const centerCol = Math.floor(COLUMNS / 2)

    // Sort by distance from center
    const sorted = [...validMoves].sort(
      (a, b) => Math.abs(a - centerCol) - Math.abs(b - centerCol)
    )

    // Apply center bias - pick from the more central moves more often
    const weights = sorted.map((_, i) => this.config.centerBias ** (sorted.length - i - 1))
    const totalWeight = weights.reduce((sum, w) => sum + w, 0)
    let random = Math.random() * totalWeight

    for (let i = 0; i < sorted.length; i++) {
      random -= weights[i]
      if (random <= 0) {
        return sorted[i]
      }
    }

    return sorted[0]
  }
}

// ============================================================================
// 2SWAP CLAIMEVEN ENGINE
// ============================================================================

/**
 * ClaimEven Engine - Based on 2swap's Connect 4 strategy videos.
 *
 * The claimeven strategy is a powerful technique for Player 2 (Yellow):
 * - When all columns have an even number of empty spaces, Yellow can
 *   guarantee claiming all even rows by simply responding directly above
 *   Red's move in the same column.
 * - This works because pieces alternate, and Yellow always fills the "paired"
 *   spot above Red's piece.
 *
 * For Player 1 (Red), we use "ClaimOdd" on the single odd-spaced column.
 *
 * Key insight from 2swap: "Wherever Red goes, Yellow fills in the unpaired spot."
 *
 * References:
 * - 2swap YouTube: "Claimeven" video
 * - https://2swap.github.io/WeakC4/explanation/
 */
export class ClaimEvenEngine implements HeuristicEngine {
  readonly id = 'claimeven'
  readonly name = 'ClaimEven Engine'
  readonly description = "2swap's claimeven strategy - responds above opponent's moves"

  private config: HeuristicConfig
  private lastOpponentMove: number | null = null

  constructor(config: Partial<HeuristicConfig> = {}) {
    this.config = { ...DEFAULT_HEURISTIC_CONFIG, ...config }
  }

  getConfig(): HeuristicConfig {
    return { ...this.config }
  }

  /**
   * Records the opponent's last move so we can respond to it.
   * This should be called before selectMove when opponent has moved.
   */
  setLastOpponentMove(column: number): void {
    this.lastOpponentMove = column
  }

  selectMove(board: Board, player: Player): number {
    const validMoves = getValidMoves(board)

    if (validMoves.length === 0) {
      throw new Error('No valid moves available')
    }

    if (validMoves.length === 1) {
      return validMoves[0]
    }

    // Check if we should follow heuristic or play randomly
    if (Math.random() > this.config.accuracy) {
      return this.selectRandomMove(validMoves)
    }

    // Priority 1: Win if possible
    const winningMove = this.findWinningMove(board, player)
    if (winningMove !== null) {
      return winningMove
    }

    // Priority 2: Block opponent's immediate win
    const opponent: Player = player === 1 ? 2 : 1
    const blockingMove = this.findWinningMove(board, opponent)
    if (blockingMove !== null) {
      return blockingMove
    }

    // Priority 3: Apply claimeven/claimodd strategy
    const claimMove = this.findClaimMove(board, player)
    if (claimMove !== null) {
      return claimMove
    }

    // Priority 4: If we can't use claimeven, try to set it up
    const setupMove = this.findSetupMove(board, player)
    if (setupMove !== null) {
      return setupMove
    }

    // Fallback: center-biased random
    return this.selectCenterBiasedMove(validMoves)
  }

  /**
   * Finds a move using the claimeven/claimodd principle.
   *
   * For Player 2 (Yellow): Respond directly above opponent's last move.
   * For Player 1 (Red): If there's exactly one odd column, play claimodd there.
   */
  private findClaimMove(board: Board, player: Player): number | null {
    const validMoves = getValidMoves(board)

    // Analyze column parities
    const columnParities = this.analyzeColumnParities(board)
    const oddColumns = columnParities.filter((c) => c.emptyCount % 2 === 1).map((c) => c.column)

    if (player === 2) {
      // Yellow: Use claimeven - respond above opponent's last move
      if (this.lastOpponentMove !== null && validMoves.includes(this.lastOpponentMove)) {
        // Check if this column has even empty spaces (ideal for claimeven)
        // or if we should respond anyway
        return this.lastOpponentMove
      }

      // If all columns are even and it's our turn, we're in a good position
      // Play to maintain even columns if possible
      if (oddColumns.length === 0) {
        // Ideal claimeven position - prefer center
        return this.selectCenterBiasedMove(validMoves)
      }

      // If there are odd columns, try to make them even
      const oddValidMoves = validMoves.filter((m) => oddColumns.includes(m))
      if (oddValidMoves.length > 0) {
        // Play in an odd column to make it even
        return this.selectCenterBiasedMove(oddValidMoves)
      }
    } else {
      // Red (Player 1): Use claimodd on the single odd column if one exists
      if (oddColumns.length === 1 && validMoves.includes(oddColumns[0])) {
        // Check if there's a threat we can claim on an odd row in this column
        const oddCol = oddColumns[0]
        return oddCol
      }

      // Red wants odd-row threats - prefer moves that create them
      // Odd rows from bottom (1,3,5) = our rows 5,3,1
      const oddRowMoves = validMoves.filter((col) => {
        const row = getAvailableRow(board, col)
        return row !== -1 && this.isOddRowFromBottom(row)
      })

      if (oddRowMoves.length > 0) {
        return this.selectCenterBiasedMove(oddRowMoves)
      }
    }

    return null
  }

  /**
   * Tries to set up a favorable claimeven position.
   * For Yellow: Make all columns have even empty spaces.
   * For Red: Create a single odd column for claimodd.
   */
  private findSetupMove(board: Board, player: Player): number | null {
    const validMoves = getValidMoves(board)
    const columnParities = this.analyzeColumnParities(board)
    const oddColumns = columnParities.filter((c) => c.emptyCount % 2 === 1).map((c) => c.column)

    if (player === 2 && oddColumns.length > 0) {
      // Yellow wants to eliminate odd columns (pair them up)
      // If there are 2 odd columns, playing in one makes it even
      const oddValidMoves = validMoves.filter((m) => oddColumns.includes(m))
      if (oddValidMoves.length > 0) {
        return this.selectCenterBiasedMove(oddValidMoves)
      }
    }

    return null
  }

  /**
   * Analyzes the parity (even/odd empty count) of each column.
   */
  private analyzeColumnParities(board: Board): Array<{ column: number; emptyCount: number }> {
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
   * Bottom row (row 5) = row 1 from bottom (odd)
   * Row 4 = row 2 from bottom (even)
   * Row 3 = row 3 from bottom (odd)
   * etc.
   */
  private isOddRowFromBottom(row: number): boolean {
    const rowFromBottom = ROWS - row // 1-indexed from bottom
    return rowFromBottom % 2 === 1
  }

  private findWinningMove(board: Board, player: Player): number | null {
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

  private selectRandomMove(validMoves: number[]): number {
    const index = Math.floor(Math.random() * validMoves.length)
    return validMoves[index]
  }

  private selectCenterBiasedMove(validMoves: number[]): number {
    const centerCol = Math.floor(COLUMNS / 2)
    const sorted = [...validMoves].sort(
      (a, b) => Math.abs(a - centerCol) - Math.abs(b - centerCol)
    )

    const weights = sorted.map((_, i) => this.config.centerBias ** (sorted.length - i - 1))
    const totalWeight = weights.reduce((sum, w) => sum + w, 0)
    let random = Math.random() * totalWeight

    for (let i = 0; i < sorted.length; i++) {
      random -= weights[i]
      if (random <= 0) {
        return sorted[i]
      }
    }

    return sorted[0]
  }
}

// ============================================================================
// 2SWAP PARITY ENGINE
// ============================================================================

/**
 * Represents a threat (3 pieces in a row with 1 empty space to complete).
 */
interface Threat {
  player: Player
  row: number // 0-indexed from top
  column: number // Column where the completing move would be
  rowFromBottom: number // 1-indexed from bottom (for parity analysis)
  isOddRow: boolean // True if odd row from bottom (favors player 1)
}

/**
 * Parity Engine - Based on 2swap's Connect 4 parity strategy.
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
export class ParityEngine implements HeuristicEngine {
  readonly id = 'parity'
  readonly name = 'Parity Engine'
  readonly description = "2swap's parity strategy - prioritizes threats on favored rows"

  private config: HeuristicConfig

  constructor(config: Partial<HeuristicConfig> = {}) {
    this.config = { ...DEFAULT_HEURISTIC_CONFIG, ...config }
  }

  getConfig(): HeuristicConfig {
    return { ...this.config }
  }

  selectMove(board: Board, player: Player): number {
    const validMoves = getValidMoves(board)

    if (validMoves.length === 0) {
      throw new Error('No valid moves available')
    }

    if (validMoves.length === 1) {
      return validMoves[0]
    }

    // Check if we should follow heuristic or play randomly
    if (Math.random() > this.config.accuracy) {
      return this.selectRandomMove(validMoves)
    }

    // Priority 1: Win if possible
    const winningMove = this.findWinningMove(board, player)
    if (winningMove !== null) {
      return winningMove
    }

    // Priority 2: Block opponent's immediate win
    const opponent: Player = player === 1 ? 2 : 1
    const blockingMove = this.findWinningMove(board, opponent)
    if (blockingMove !== null) {
      return blockingMove
    }

    // Priority 3: Create threats on favored parity rows
    const parityMove = this.findParityMove(board, player)
    if (parityMove !== null) {
      return parityMove
    }

    // Priority 4: Block opponent's threats on their favored parity
    const blockParityMove = this.findBlockParityMove(board, player)
    if (blockParityMove !== null) {
      return blockParityMove
    }

    // Fallback: center-biased random
    return this.selectCenterBiasedMove(validMoves)
  }

  /**
   * Finds threats (3-in-a-row with one empty completion spot) on the board.
   */
  private findThreats(board: Board, player: Player): Threat[] {
    const threats: Threat[] = []

    // Check all possible 4-in-a-row patterns
    for (let row = 0; row < ROWS; row++) {
      for (let col = 0; col < COLUMNS; col++) {
        // Horizontal
        if (col + 3 < COLUMNS) {
          const threat = this.checkThreatPattern(board, player, row, col, 0, 1)
          if (threat) threats.push(threat)
        }

        // Vertical
        if (row + 3 < ROWS) {
          const threat = this.checkThreatPattern(board, player, row, col, 1, 0)
          if (threat) threats.push(threat)
        }

        // Diagonal down-right
        if (row + 3 < ROWS && col + 3 < COLUMNS) {
          const threat = this.checkThreatPattern(board, player, row, col, 1, 1)
          if (threat) threats.push(threat)
        }

        // Diagonal down-left
        if (row + 3 < ROWS && col - 3 >= 0) {
          const threat = this.checkThreatPattern(board, player, row, col, 1, -1)
          if (threat) threats.push(threat)
        }
      }
    }

    return threats
  }

  /**
   * Checks a 4-cell pattern for a threat (3 player pieces + 1 empty).
   */
  private checkThreatPattern(
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
   * Checks if a position can currently be played (is the next available row).
   */
  private isPlayablePosition(board: Board, row: number, col: number): boolean {
    const availableRow = getAvailableRow(board, col)
    return availableRow === row
  }

  /**
   * Finds a move that creates a threat on the player's favored parity.
   * Player 1 (Red) favors odd rows (1, 3, 5 from bottom).
   * Player 2 (Yellow) favors even rows (2, 4, 6 from bottom).
   */
  private findParityMove(board: Board, player: Player): number | null {
    const validMoves = getValidMoves(board)
    const favoredParity = player === 1 // true = odd rows for player 1

    const scoredMoves: Array<{ col: number; score: number }> = []

    for (const col of validMoves) {
      const result = applyMove(board, col, player)
      if (!result.success || !result.board) continue

      // Find threats created by this move
      const newThreats = this.findThreats(result.board, player)
      const existingThreats = this.findThreats(board, player)

      // Threats newly created by this move
      const createdThreats = newThreats.filter(
        (nt) => !existingThreats.some((et) => et.row === nt.row && et.column === nt.column)
      )

      // Score based on threats on favored parity
      let score = 0
      for (const threat of createdThreats) {
        if (threat.isOddRow === favoredParity) {
          // Threat on favored parity - higher score for lower rows (harder to undercut)
          score += 10 + threat.rowFromBottom
        } else {
          // Threat on wrong parity - still worth something but less
          score += 2
        }
      }

      // Bonus for lower row placements (controls parity better)
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

    // Return the move with the highest score
    scoredMoves.sort((a, b) => b.score - a.score)
    return scoredMoves[0].col
  }

  /**
   * Finds a move that blocks opponent's threats on their favored parity.
   */
  private findBlockParityMove(board: Board, player: Player): number | null {
    const opponent: Player = player === 1 ? 2 : 1
    const opponentFavoredParity = opponent === 1 // true = odd rows for opponent

    const opponentThreats = this.findThreats(board, opponent)

    // Find the most dangerous threat (lowest row on opponent's favored parity)
    const dangerousThreats = opponentThreats
      .filter((t) => t.isOddRow === opponentFavoredParity)
      .sort((a, b) => b.rowFromBottom - a.rowFromBottom) // Lowest threats first

    for (const threat of dangerousThreats) {
      // Check if we can play at the threat's completion point
      if (this.isPlayablePosition(board, threat.row, threat.column)) {
        return threat.column
      }

      // Check if we can play below the threat to prevent it later
      const availableRow = getAvailableRow(board, threat.column)
      if (availableRow !== -1 && availableRow > threat.row) {
        // We can play in this column - consider if it helps
        const result = applyMove(board, threat.column, player)
        if (result.success && result.board) {
          // Check if this creates our own threat
          const ourThreats = this.findThreats(result.board, player)
          if (ourThreats.length > 0) {
            return threat.column
          }
        }
      }
    }

    return null
  }

  private findWinningMove(board: Board, player: Player): number | null {
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

  private selectRandomMove(validMoves: number[]): number {
    const index = Math.floor(Math.random() * validMoves.length)
    return validMoves[index]
  }

  private selectCenterBiasedMove(validMoves: number[]): number {
    const centerCol = Math.floor(COLUMNS / 2)
    const sorted = [...validMoves].sort(
      (a, b) => Math.abs(a - centerCol) - Math.abs(b - centerCol)
    )

    const weights = sorted.map((_, i) => this.config.centerBias ** (sorted.length - i - 1))
    const totalWeight = weights.reduce((sum, w) => sum + w, 0)
    let random = Math.random() * totalWeight

    for (let i = 0; i < sorted.length; i++) {
      random -= weights[i]
      if (random <= 0) {
        return sorted[i]
      }
    }

    return sorted[0]
  }
}

// ============================================================================
// 2SWAP THREAT PAIRS ENGINE
// ============================================================================

/**
 * Represents a major threat (3 pieces + 1 empty to win).
 */
interface MajorThreat {
  player: Player
  row: number
  column: number
  direction: 'horizontal' | 'vertical' | 'diagonal-up' | 'diagonal-down'
}

/**
 * Represents a minor threat (2 pieces + 2 empty, potential to become major).
 */
interface MinorThreat {
  player: Player
  emptyPositions: Array<{ row: number; column: number }>
  direction: 'horizontal' | 'vertical' | 'diagonal-up' | 'diagonal-down'
}

/**
 * ThreatPairs Engine - Based on 2swap's Connect 4 combinatoric wins video.
 *
 * Key concepts:
 * - Major Threat (T): 3 pieces in a row + 1 empty = playing there wins
 * - Minor Threat (t): 2 pieces in a row + 2 empty = can become major threat
 * - Combinatoric Win: Two simultaneous threats opponent can't block both
 * - Stacked Threats: Threats directly above each other - extremely powerful
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
export class ThreatPairsEngine implements HeuristicEngine {
  readonly id = 'threat-pairs'
  readonly name = 'ThreatPairs Engine'
  readonly description = "2swap's threat pairs strategy - creates combinatoric wins"

  private config: HeuristicConfig

  constructor(config: Partial<HeuristicConfig> = {}) {
    this.config = { ...DEFAULT_HEURISTIC_CONFIG, ...config }
  }

  getConfig(): HeuristicConfig {
    return { ...this.config }
  }

  selectMove(board: Board, player: Player): number {
    const validMoves = getValidMoves(board)

    if (validMoves.length === 0) {
      throw new Error('No valid moves available')
    }

    if (validMoves.length === 1) {
      return validMoves[0]
    }

    // Check if we should follow heuristic or play randomly
    if (Math.random() > this.config.accuracy) {
      return this.selectRandomMove(validMoves)
    }

    // Priority 1: Win if possible
    const winningMove = this.findWinningMove(board, player)
    if (winningMove !== null) {
      return winningMove
    }

    // Priority 2: Block opponent's immediate win
    const opponent: Player = player === 1 ? 2 : 1
    const blockingMove = this.findWinningMove(board, opponent)
    if (blockingMove !== null) {
      return blockingMove
    }

    // Priority 3: Create a combinatoric win (double threat)
    const doubleTheatMove = this.findDoubleThreatMove(board, player)
    if (doubleTheatMove !== null) {
      return doubleTheatMove
    }

    // Priority 4: Block opponent's potential double threat
    const blockDoubleMove = this.findBlockDoubleThreatMove(board, player)
    if (blockDoubleMove !== null) {
      return blockDoubleMove
    }

    // Priority 5: Create a stacked threat setup
    const stackedSetupMove = this.findStackedThreatSetup(board, player)
    if (stackedSetupMove !== null) {
      return stackedSetupMove
    }

    // Priority 6: Create any major threat
    const threatMove = this.findCreateThreatMove(board, player)
    if (threatMove !== null) {
      return threatMove
    }

    // Fallback: center-biased
    return this.selectCenterBiasedMove(validMoves)
  }

  /**
   * Finds all major threats for a player.
   * A major threat is 3 pieces in a row with 1 empty space to complete.
   */
  private findMajorThreats(board: Board, player: Player): MajorThreat[] {
    const threats: MajorThreat[] = []

    for (let row = 0; row < ROWS; row++) {
      for (let col = 0; col < COLUMNS; col++) {
        // Horizontal
        if (col + 3 < COLUMNS) {
          const threat = this.checkMajorThreatPattern(board, player, row, col, 0, 1, 'horizontal')
          if (threat) threats.push(threat)
        }

        // Vertical
        if (row + 3 < ROWS) {
          const threat = this.checkMajorThreatPattern(board, player, row, col, 1, 0, 'vertical')
          if (threat) threats.push(threat)
        }

        // Diagonal down-right
        if (row + 3 < ROWS && col + 3 < COLUMNS) {
          const threat = this.checkMajorThreatPattern(board, player, row, col, 1, 1, 'diagonal-down')
          if (threat) threats.push(threat)
        }

        // Diagonal up-right (down-left when viewed from start)
        if (row - 3 >= 0 && col + 3 < COLUMNS) {
          const threat = this.checkMajorThreatPattern(board, player, row, col, -1, 1, 'diagonal-up')
          if (threat) threats.push(threat)
        }
      }
    }

    return threats
  }

  /**
   * Checks a 4-cell pattern for a major threat.
   */
  private checkMajorThreatPattern(
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
          return null // More than one empty
        }
      } else {
        return null // Opponent blocks
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

  /**
   * Finds all minor threats for a player.
   * A minor threat is 2 pieces in a row with 2 empty spaces.
   */
  private findMinorThreats(board: Board, player: Player): MinorThreat[] {
    const threats: MinorThreat[] = []

    for (let row = 0; row < ROWS; row++) {
      for (let col = 0; col < COLUMNS; col++) {
        // Horizontal
        if (col + 3 < COLUMNS) {
          const threat = this.checkMinorThreatPattern(board, player, row, col, 0, 1, 'horizontal')
          if (threat) threats.push(threat)
        }

        // Vertical
        if (row + 3 < ROWS) {
          const threat = this.checkMinorThreatPattern(board, player, row, col, 1, 0, 'vertical')
          if (threat) threats.push(threat)
        }

        // Diagonal down-right
        if (row + 3 < ROWS && col + 3 < COLUMNS) {
          const threat = this.checkMinorThreatPattern(board, player, row, col, 1, 1, 'diagonal-down')
          if (threat) threats.push(threat)
        }

        // Diagonal up-right
        if (row - 3 >= 0 && col + 3 < COLUMNS) {
          const threat = this.checkMinorThreatPattern(board, player, row, col, -1, 1, 'diagonal-up')
          if (threat) threats.push(threat)
        }
      }
    }

    return threats
  }

  /**
   * Checks a 4-cell pattern for a minor threat (2 pieces + 2 empty).
   */
  private checkMinorThreatPattern(
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
        return null // Opponent blocks
      }
    }

    if (playerCount === 2 && emptyPositions.length === 2) {
      return {
        player,
        emptyPositions,
        direction,
      }
    }

    return null
  }

  /**
   * Finds a move that creates two major threats simultaneously (combinatoric win).
   */
  private findDoubleThreatMove(board: Board, player: Player): number | null {
    const validMoves = getValidMoves(board)

    for (const col of validMoves) {
      const result = applyMove(board, col, player)
      if (!result.success || !result.board) continue

      const threats = this.findMajorThreats(result.board, player)

      // Check for two or more threats (combinatoric win!)
      if (threats.length >= 2) {
        // Extra bonus: check for stacked threats (same column, adjacent rows)
        const stackedPairs = this.findStackedThreats(threats)
        if (stackedPairs.length > 0) {
          return col // Stacked threats are especially powerful
        }

        return col // Any double threat is a win
      }
    }

    return null
  }

  /**
   * Finds stacked threat pairs (threats in same column, adjacent rows).
   */
  private findStackedThreats(threats: MajorThreat[]): Array<[MajorThreat, MajorThreat]> {
    const pairs: Array<[MajorThreat, MajorThreat]> = []

    for (let i = 0; i < threats.length; i++) {
      for (let j = i + 1; j < threats.length; j++) {
        const t1 = threats[i]
        const t2 = threats[j]

        // Same column, adjacent rows (stacked)
        if (t1.column === t2.column && Math.abs(t1.row - t2.row) === 1) {
          pairs.push([t1, t2])
        }
      }
    }

    return pairs
  }

  /**
   * Finds a move that blocks opponent's potential double threat.
   * Looks for opponent's minor threats that could become dangerous.
   */
  private findBlockDoubleThreatMove(board: Board, player: Player): number | null {
    const opponent: Player = player === 1 ? 2 : 1
    const validMoves = getValidMoves(board)

    // Check each opponent move to see if it would create a double threat
    for (const col of validMoves) {
      const oppResult = applyMove(board, col, opponent)
      if (!oppResult.success || !oppResult.board) continue

      const oppThreats = this.findMajorThreats(oppResult.board, opponent)

      if (oppThreats.length >= 2) {
        // Opponent would get a double threat here - we should play here to block!
        return col
      }
    }

    // Also look for opponent's minor threats we could disrupt
    const minorThreats = this.findMinorThreats(board, opponent)
    for (const threat of minorThreats) {
      for (const emptyPos of threat.emptyPositions) {
        // Check if we can play at this position
        const availableRow = getAvailableRow(board, emptyPos.column)
        if (availableRow === emptyPos.row && validMoves.includes(emptyPos.column)) {
          // Playing here blocks the minor threat
          return emptyPos.column
        }
      }
    }

    return null
  }

  /**
   * Finds a move that sets up a stacked threat (the "7" shape).
   * Creates a threat where another threat can be placed directly above.
   */
  private findStackedThreatSetup(board: Board, player: Player): number | null {
    const validMoves = getValidMoves(board)

    for (const col of validMoves) {
      const result = applyMove(board, col, player)
      if (!result.success || !result.board) continue

      const threats = this.findMajorThreats(result.board, player)

      for (const threat of threats) {
        // Check if there's potential for a stacked threat above this one
        if (threat.row > 0) {
          const rowAbove = threat.row - 1

          // Simulate playing at the threat point, then check for another threat above
          const threatResult = applyMove(result.board, threat.column, player)
          if (threatResult.success && threatResult.board) {
            const secondaryThreats = this.findMajorThreats(threatResult.board, player)
            const stackedAbove = secondaryThreats.some(
              (t) => t.column === threat.column && t.row === rowAbove
            )
            if (stackedAbove) {
              return col // This move sets up a stacked threat
            }
          }
        }
      }
    }

    return null
  }

  /**
   * Finds a move that creates any major threat.
   */
  private findCreateThreatMove(board: Board, player: Player): number | null {
    const validMoves = getValidMoves(board)
    const existingThreats = this.findMajorThreats(board, player)

    for (const col of validMoves) {
      const result = applyMove(board, col, player)
      if (!result.success || !result.board) continue

      const newThreats = this.findMajorThreats(result.board, player)

      // Check if this move creates a new threat
      const createdThreats = newThreats.filter(
        (nt) => !existingThreats.some((et) => et.row === nt.row && et.column === nt.column)
      )

      if (createdThreats.length > 0) {
        return col
      }
    }

    return null
  }

  private findWinningMove(board: Board, player: Player): number | null {
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

  private selectRandomMove(validMoves: number[]): number {
    const index = Math.floor(Math.random() * validMoves.length)
    return validMoves[index]
  }

  private selectCenterBiasedMove(validMoves: number[]): number {
    const centerCol = Math.floor(COLUMNS / 2)
    const sorted = [...validMoves].sort(
      (a, b) => Math.abs(a - centerCol) - Math.abs(b - centerCol)
    )

    const weights = sorted.map((_, i) => this.config.centerBias ** (sorted.length - i - 1))
    const totalWeight = weights.reduce((sum, w) => sum + w, 0)
    let random = Math.random() * totalWeight

    for (let i = 0; i < sorted.length; i++) {
      random -= weights[i]
      if (random <= 0) {
        return sorted[i]
      }
    }

    return sorted[0]
  }
}

// ============================================================================
// ENGINE REGISTRY
// ============================================================================

/**
 * Registry entry for a heuristic engine.
 */
export interface HeuristicEngineEntry {
  id: string
  name: string
  description: string
  create: (config?: Partial<HeuristicConfig>) => HeuristicEngine
}

/**
 * Registry of available heuristic engines.
 */
const HEURISTIC_ENGINE_REGISTRY: HeuristicEngineEntry[] = [
  {
    id: 'random',
    name: 'Random Engine',
    description: 'Picks randomly from valid moves with slight center bias',
    create: (config) => new RandomEngine(config),
  },
  {
    id: 'threat-heuristic',
    name: 'Threat Heuristic Engine',
    description: 'Priority-based defensive engine that focuses on blocking threats',
    create: (config) => new ThreatHeuristicEngine(config),
  },
  {
    id: 'claimeven',
    name: 'ClaimEven Engine',
    description: "2swap's claimeven strategy - responds above opponent's moves",
    create: (config) => new ClaimEvenEngine(config),
  },
  {
    id: 'parity',
    name: 'Parity Engine',
    description: "2swap's parity strategy - prioritizes threats on favored rows",
    create: (config) => new ParityEngine(config),
  },
  {
    id: 'threat-pairs',
    name: 'ThreatPairs Engine',
    description: "2swap's threat pairs strategy - creates combinatoric wins",
    create: (config) => new ThreatPairsEngine(config),
  },
]

/**
 * Creates a heuristic engine by ID.
 *
 * @param engineId - The engine ID ('random' or 'threat-heuristic')
 * @param config - Optional configuration overrides
 * @returns The created engine
 * @throws Error if engine ID is not found
 */
export function createHeuristicEngine(
  engineId: string,
  config?: Partial<HeuristicConfig>
): HeuristicEngine {
  const entry = HEURISTIC_ENGINE_REGISTRY.find((e) => e.id === engineId)
  if (!entry) {
    throw new Error(`Heuristic engine not found: ${engineId}`)
  }
  return entry.create(config)
}

/**
 * Lists all available heuristic engines.
 */
export function listHeuristicEngines(): HeuristicEngineEntry[] {
  return [...HEURISTIC_ENGINE_REGISTRY]
}

/**
 * Registers a custom heuristic engine.
 *
 * @param entry - The engine entry to register
 */
export function registerHeuristicEngine(entry: HeuristicEngineEntry): void {
  const existingIndex = HEURISTIC_ENGINE_REGISTRY.findIndex((e) => e.id === entry.id)
  if (existingIndex >= 0) {
    HEURISTIC_ENGINE_REGISTRY[existingIndex] = entry
  } else {
    HEURISTIC_ENGINE_REGISTRY.push(entry)
  }
}

// ============================================================================
// PRESET CONFIGURATIONS
// ============================================================================

/**
 * Preset configurations for bot personas.
 */
export const HEURISTIC_PRESETS = {
  /**
   * Rusty at very low skill - makes obvious mistakes.
   */
  rusty: {
    engineId: 'random',
    config: {
      accuracy: 0.3, // 30% chance of following any heuristic
      detectTraps: false,
      centerBias: 1.2,
    },
  },

  /**
   * Sentinel - defensive, rarely loses to tactics.
   */
  sentinel: {
    engineId: 'threat-heuristic',
    config: {
      accuracy: 0.95, // 95% accuracy
      detectTraps: true,
      centerBias: 1.5,
    },
  },

  /**
   * Sentinel at lower skill - still defensive but makes more mistakes.
   */
  sentinelEasy: {
    engineId: 'threat-heuristic',
    config: {
      accuracy: 0.7,
      detectTraps: false,
      centerBias: 1.3,
    },
  },

  /**
   * 2swap ClaimEven - uses the claimeven strategy (best for player 2/Yellow).
   */
  claimeven: {
    engineId: 'claimeven',
    config: {
      accuracy: 0.95,
      detectTraps: false,
      centerBias: 1.4,
    },
  },

  /**
   * 2swap Parity - uses parity-based threat analysis.
   */
  parity: {
    engineId: 'parity',
    config: {
      accuracy: 0.95,
      detectTraps: false,
      centerBias: 1.4,
    },
  },

  /**
   * 2swap ThreatPairs - creates combinatoric wins through double threats.
   */
  threatPairs: {
    engineId: 'threat-pairs',
    config: {
      accuracy: 0.95,
      detectTraps: false,
      centerBias: 1.4,
    },
  },
} as const

export type HeuristicPreset = keyof typeof HEURISTIC_PRESETS

/**
 * Creates a heuristic engine from a preset.
 *
 * @param preset - The preset name
 * @returns The created engine
 */
export function createFromPreset(preset: HeuristicPreset): HeuristicEngine {
  const { engineId, config } = HEURISTIC_PRESETS[preset]
  return createHeuristicEngine(engineId, config)
}
