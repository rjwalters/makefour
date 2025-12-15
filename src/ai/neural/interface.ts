/**
 * Neural Network Agent Interface
 *
 * Defines the interface for neural network-based game evaluation and move selection.
 * This enables research on minimal viable networks for specific ELO thresholds.
 */

import type { GameState, Player } from '../../game/makefour'

/**
 * Position representation for neural network input.
 * Uses a standardized encoding that can be adapted for different architectures.
 */
export interface EncodedPosition {
  /** Flattened input tensor for the neural network */
  input: Float32Array
  /** Shape of the input tensor [batch, channels, height, width] or [batch, features] */
  shape: number[]
  /** Current player (1 or 2) */
  currentPlayer: Player
  /** Number of moves played */
  moveCount: number
}

/**
 * Neural network evaluation result.
 */
export interface NeuralEvaluation {
  /** Position value from -1 (losing) to 1 (winning) for current player */
  value: number
  /** Move probabilities for each column (7 values, sum to 1) */
  policy: number[]
  /** Confidence in the evaluation (0-1) */
  confidence: number
  /** Inference time in milliseconds */
  inferenceTimeMs: number
}

/**
 * Model metadata for UI display and selection.
 */
export interface ModelMetadata {
  /** Unique identifier for the model */
  id: string
  /** Human-readable name */
  name: string
  /** Model architecture type */
  architecture: 'mlp' | 'cnn' | 'transformer'
  /** Expected ELO rating based on training */
  expectedElo: number
  /** Model file size in bytes */
  sizeBytes: number
  /** URL to download the model */
  url: string
  /** Model version */
  version: string
  /** Training information */
  training?: {
    /** Number of games trained on */
    games: number
    /** Training epochs */
    epochs: number
    /** Date trained */
    date: string
  }
  /** Position encoding used by this model */
  encoding: PositionEncodingType
}

/**
 * Types of position encoding supported.
 */
export type PositionEncodingType = 'onehot-6x7x3' | 'bitboard' | 'flat-binary'

/**
 * Neural network agent interface.
 * Implementations provide evaluation and move suggestion using neural networks.
 */
export interface NeuralAgent {
  /**
   * Evaluates a position and returns a value between -1 and 1.
   * Positive values favor the current player.
   *
   * @param state - The game state to evaluate
   * @returns Value between -1 (losing) and 1 (winning) for current player
   */
  evaluate(state: GameState): Promise<number>

  /**
   * Suggests a move for the current position.
   *
   * @param state - The game state
   * @returns Column index (0-6) for the suggested move
   */
  suggestMove(state: GameState): Promise<number>

  /**
   * Gets move probabilities for all columns.
   *
   * @param state - The game state
   * @returns Array of 7 probabilities (one per column), normalized to sum to 1
   */
  getMoveProbabilities(state: GameState): Promise<number[]>

  /**
   * Gets full evaluation including value, policy, and timing.
   *
   * @param state - The game state
   * @returns Complete neural evaluation
   */
  getFullEvaluation(state: GameState): Promise<NeuralEvaluation>

  /**
   * Checks if the model is loaded and ready.
   */
  isReady(): boolean

  /**
   * Gets the model metadata.
   */
  getMetadata(): ModelMetadata

  /**
   * Disposes of the model to free resources.
   */
  dispose(): void
}

/**
 * Factory for creating neural agents.
 */
export interface NeuralAgentFactory {
  /**
   * Creates a neural agent from a model ID.
   * Downloads and caches the model if needed.
   *
   * @param modelId - The model identifier
   * @returns Promise resolving to a neural agent
   */
  create(modelId: string): Promise<NeuralAgent>

  /**
   * Lists available models.
   *
   * @returns Promise resolving to array of model metadata
   */
  listModels(): Promise<ModelMetadata[]>

  /**
   * Checks if a model is cached locally.
   *
   * @param modelId - The model identifier
   * @returns Promise resolving to true if cached
   */
  isCached(modelId: string): Promise<boolean>

  /**
   * Clears the model cache.
   */
  clearCache(): Promise<void>
}

/**
 * Configuration for neural inference.
 */
export interface NeuralInferenceConfig {
  /** Use Web Workers for inference (default: true) */
  useWorkers: boolean
  /** Maximum inference time in ms before timeout */
  inferenceTimeout: number
  /** Enable debug logging */
  debug: boolean
}

/**
 * Default inference configuration.
 */
export const DEFAULT_INFERENCE_CONFIG: NeuralInferenceConfig = {
  useWorkers: true,
  inferenceTimeout: 100,
  debug: false,
}

/**
 * A/B testing result for comparing models.
 */
export interface ModelComparisonResult {
  /** First model ID */
  modelA: string
  /** Second model ID */
  modelB: string
  /** Number of games played */
  games: number
  /** Wins for model A */
  winsA: number
  /** Wins for model B */
  winsB: number
  /** Draws */
  draws: number
  /** Statistical significance (p-value) */
  pValue: number
  /** Estimated ELO difference */
  eloDifference: number
}
