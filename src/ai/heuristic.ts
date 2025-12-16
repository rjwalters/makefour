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
  applyMove,
  checkWinner,
  COLUMNS,
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
