/**
 * AI Engine Abstraction Layer
 *
 * Provides a pluggable interface for different bot AI implementations.
 * Engines can use different decision-making approaches: heuristic, minimax,
 * MCTS, neural networks, or hybrid approaches.
 */

import type { Board, Player } from './game'

// ============================================================================
// CORE TYPES
// ============================================================================

/**
 * Configuration passed to engines for move selection.
 */
export interface EngineConfig {
  /** Search depth for tree-search algorithms */
  searchDepth: number
  /** Rate at which the engine makes intentional mistakes (0-1) */
  errorRate: number
  /** Optional custom parameters for specific engines */
  customParams?: Record<string, unknown>
}

/**
 * Result returned from move selection.
 */
export interface MoveResult {
  /** Selected column (0-6) */
  column: number
  /** Confidence in the move (0-1), if available */
  confidence?: number
  /** Optional search statistics */
  searchInfo?: SearchInfo
}

/**
 * Search statistics for debugging and analysis.
 */
export interface SearchInfo {
  /** Maximum depth reached during search */
  depth?: number
  /** Number of positions evaluated */
  nodesSearched?: number
  /** Time spent on move selection (ms) */
  timeUsed?: number
  /** Principal variation (best move sequence) */
  principalVariation?: number[]
}

/**
 * Pluggable AI engine interface.
 *
 * Engines are stateless - they don't maintain game-to-game memory.
 * Time budget management is handled at the engine level.
 */
export interface AIEngine {
  /** Unique engine identifier */
  readonly name: string

  /** Human-readable description */
  readonly description: string

  /**
   * Select the best move for the given position.
   *
   * @param board - Current board state
   * @param player - Player to move (1 or 2)
   * @param config - Engine configuration
   * @param timeBudget - Time budget in milliseconds
   * @returns Promise resolving to the selected move and metadata
   */
  selectMove(
    board: Board,
    player: Player,
    config: EngineConfig,
    timeBudget: number
  ): Promise<MoveResult>

  /**
   * Evaluate the position from the perspective of the given player.
   * Optional - not all engines support static evaluation.
   *
   * @param board - Current board state
   * @param player - Player to evaluate for
   * @returns Evaluation score (positive = good for player)
   */
  evaluatePosition?(board: Board, player: Player): number

  /**
   * Generate a human-readable explanation for a move.
   * Optional - useful for debugging and spectating.
   *
   * @param board - Board state before the move
   * @param move - Column that was played
   * @param player - Player who made the move
   * @returns Human-readable explanation
   */
  explainMove?(board: Board, move: number, player: Player): string
}

// ============================================================================
// ENGINE REGISTRY
// ============================================================================

/**
 * Engine registration entry with metadata.
 */
interface EngineEntry {
  engine: AIEngine
  /** Whether this engine is available (e.g., has required models loaded) */
  available: boolean
}

/**
 * Registry for AI engines.
 * Provides lookup by name with graceful fallback.
 */
class EngineRegistry {
  private engines: Map<string, EngineEntry> = new Map()
  private defaultEngineName: string | null = null

  /**
   * Register an engine.
   *
   * @param engine - Engine instance to register
   * @param available - Whether the engine is currently available
   */
  register(engine: AIEngine, available = true): void {
    this.engines.set(engine.name, { engine, available })

    // First registered engine becomes the default
    if (this.defaultEngineName === null) {
      this.defaultEngineName = engine.name
    }
  }

  /**
   * Unregister an engine.
   *
   * @param name - Engine name to unregister
   */
  unregister(name: string): void {
    this.engines.delete(name)

    // Update default if we removed it
    if (this.defaultEngineName === name) {
      const firstAvailable = Array.from(this.engines.entries()).find(
        ([, entry]) => entry.available
      )
      this.defaultEngineName = firstAvailable?.[0] ?? null
    }
  }

  /**
   * Get an engine by name.
   *
   * @param name - Engine name to look up
   * @returns Engine instance or null if not found
   */
  get(name: string): AIEngine | null {
    const entry = this.engines.get(name)
    return entry?.engine ?? null
  }

  /**
   * Get an engine by name with fallback to default.
   *
   * @param name - Preferred engine name
   * @returns Engine instance (falls back to default if preferred unavailable)
   * @throws Error if no engines are registered
   */
  getWithFallback(name: string): AIEngine {
    const entry = this.engines.get(name)

    if (entry?.available) {
      return entry.engine
    }

    // Fall back to default
    if (this.defaultEngineName) {
      const defaultEntry = this.engines.get(this.defaultEngineName)
      if (defaultEntry?.available) {
        console.warn(
          `Engine "${name}" unavailable, falling back to "${this.defaultEngineName}"`
        )
        return defaultEntry.engine
      }
    }

    // Try any available engine
    const anyAvailable = Array.from(this.engines.values()).find(
      (e) => e.available
    )
    if (anyAvailable) {
      console.warn(
        `Engine "${name}" unavailable, falling back to "${anyAvailable.engine.name}"`
      )
      return anyAvailable.engine
    }

    throw new Error('No AI engines available')
  }

  /**
   * Set the default engine.
   *
   * @param name - Engine name to set as default
   * @throws Error if engine not registered
   */
  setDefault(name: string): void {
    if (!this.engines.has(name)) {
      throw new Error(`Engine "${name}" not registered`)
    }
    this.defaultEngineName = name
  }

  /**
   * Get the default engine.
   *
   * @returns Default engine or null if none registered
   */
  getDefault(): AIEngine | null {
    if (this.defaultEngineName === null) return null
    return this.get(this.defaultEngineName)
  }

  /**
   * Mark an engine as available or unavailable.
   *
   * @param name - Engine name
   * @param available - Availability status
   */
  setAvailability(name: string, available: boolean): void {
    const entry = this.engines.get(name)
    if (entry) {
      entry.available = available
    }
  }

  /**
   * List all registered engines.
   *
   * @returns Array of engine info objects
   */
  list(): Array<{ name: string; description: string; available: boolean }> {
    return Array.from(this.engines.entries()).map(([name, entry]) => ({
      name,
      description: entry.engine.description,
      available: entry.available,
    }))
  }

  /**
   * Check if an engine is registered.
   *
   * @param name - Engine name to check
   * @returns True if engine is registered
   */
  has(name: string): boolean {
    return this.engines.has(name)
  }
}

// Global engine registry instance
export const engineRegistry = new EngineRegistry()

// ============================================================================
// ENGINE TYPE DEFINITIONS
// ============================================================================

/**
 * Supported engine types.
 * Used in bot persona configuration to specify which engine to use.
 */
export type EngineType =
  | 'heuristic' // Simple rule-based (check threats, prefer center)
  | 'minimax' // Minimax with alpha-beta pruning
  | 'aggressive-minimax' // Aggressive minimax favoring threats (for Blitz)
  | 'deep-minimax' // Deep minimax with transposition tables (for Oracle)
  | 'mcts' // Monte Carlo Tree Search
  | 'neural' // Neural network inference
  | 'hybrid' // Combines approaches (e.g., NN eval + minimax search)

/**
 * Default engine configurations by type.
 */
export const DEFAULT_ENGINE_CONFIGS: Record<EngineType, Partial<EngineConfig>> =
  {
    heuristic: { searchDepth: 1, errorRate: 0 },
    minimax: { searchDepth: 6, errorRate: 0 },
    'aggressive-minimax': {
      searchDepth: 6,
      errorRate: 0,
      customParams: {
        evalWeights: {
          ownThreats: 150,
          opponentThreats: 80,
          centerControl: 5,
          doubleThreats: 500,
        },
      },
    },
    'deep-minimax': {
      searchDepth: 42, // Solve completely when possible
      errorRate: 0,
      customParams: {
        useTranspositionTable: true,
      },
    },
    mcts: { searchDepth: 1000, errorRate: 0 }, // depth = simulations for MCTS
    neural: { searchDepth: 1, errorRate: 0 },
    hybrid: { searchDepth: 4, errorRate: 0 },
  }
