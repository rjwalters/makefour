/**
 * Neural Model Loader and Registry
 *
 * Manages available neural network models and provides factory methods
 * for creating neural agents.
 */

import type { GameState } from '../../game/makefour'
import { getValidMoves } from '../../game/makefour'
import type {
  NeuralAgent,
  NeuralAgentFactory,
  NeuralEvaluation,
  ModelMetadata,
  NeuralInferenceConfig,
} from './interface'
import { createOnnxAgent, clearModelCache, isModelCached } from './onnx'
import { evaluatePosition, findBestMove } from '../coach'

/**
 * Registry of available models.
 * Models can be served from API or bundled with the app.
 * This registry is populated with built-in models and can be
 * extended by fetching from the API.
 */
/** Base URL for model storage (Cloudflare R2 with public access) */
const MODELS_BASE_URL = 'https://pub-dd38ba981172498a918d3b50f5ebae6c.r2.dev'

const MODEL_REGISTRY: ModelMetadata[] = [
  {
    id: 'heuristic-v1',
    name: 'Heuristic Baseline',
    architecture: 'mlp',
    expectedElo: 1200,
    sizeBytes: 0,
    url: '', // Built-in, no download needed
    version: '1.0.0',
    encoding: 'flat-binary',
  },
  {
    id: 'mlp-tiny-v1',
    name: 'MLP Tiny v1',
    architecture: 'mlp',
    expectedElo: 800, // Weak model, trained on synthetic data
    sizeBytes: 13733,
    url: `${MODELS_BASE_URL}/mlp-tiny-v1.onnx`,
    version: '1.0.0',
    encoding: 'flat-binary',
  },
  {
    id: 'selfplay-v1',
    name: 'Self-Play v1',
    architecture: 'mlp',
    expectedElo: 900, // ~50% win rate vs random
    sizeBytes: 13507,
    url: `${MODELS_BASE_URL}/selfplay-v1.onnx`,
    version: '1.0.0',
    encoding: 'flat-binary',
  },
  {
    id: 'selfplay-v2',
    name: 'Self-Play v2',
    architecture: 'mlp',
    expectedElo: 1000, // 54-80% win rate vs random
    sizeBytes: 13685,
    url: `${MODELS_BASE_URL}/selfplay-v2.onnx`,
    version: '1.0.0',
    encoding: 'flat-binary',
  },
  {
    id: 'selfplay-v3',
    name: 'Self-Play v3',
    architecture: 'mlp',
    expectedElo: 1100, // 90% win rate vs random, trained with curriculum
    sizeBytes: 16983,
    url: `${MODELS_BASE_URL}/selfplay-v3.onnx`,
    version: '1.0.0',
    encoding: 'flat-binary',
  },
]

/** API endpoint for fetching available models */
const MODELS_API_URL = '/api/models'

/** Whether the registry has been initialized from the API */
let registryInitialized = false

/** Promise for initialization to avoid duplicate fetches */
let initializationPromise: Promise<void> | null = null

/**
 * Heuristic-based neural agent that uses the existing minimax evaluation.
 * Serves as a baseline and fallback when real neural models aren't available.
 */
export class HeuristicNeuralAgent implements NeuralAgent {
  private metadata: ModelMetadata = {
    id: 'heuristic-v1',
    name: 'Heuristic Baseline',
    architecture: 'mlp',
    expectedElo: 1200,
    sizeBytes: 0,
    url: '',
    version: '1.0.0',
    encoding: 'flat-binary',
  }

  isReady(): boolean {
    return true
  }

  getMetadata(): ModelMetadata {
    return this.metadata
  }

  async evaluate(state: GameState): Promise<number> {
    const evaluation = await this.getFullEvaluation(state)
    return evaluation.value
  }

  async suggestMove(state: GameState): Promise<number> {
    // Use minimax with moderate depth for reasonable performance
    const { move } = findBestMove(state.board, state.currentPlayer, 4)
    return move
  }

  async getMoveProbabilities(state: GameState): Promise<number[]> {
    const evaluation = await this.getFullEvaluation(state)
    return evaluation.policy
  }

  async getFullEvaluation(state: GameState): Promise<NeuralEvaluation> {
    const startTime = performance.now()

    // Evaluate position using existing heuristic
    const score = evaluatePosition(state.board, state.currentPlayer)

    // Convert score to value in [-1, 1] range
    // Score typically ranges from -100000 to 100000
    const normalizedValue = Math.tanh(score / 1000)

    // Get valid moves
    const validMoves = getValidMoves(state.board)

    // Generate policy by evaluating each move
    const policy = new Array(7).fill(0)

    for (const col of validMoves) {
      const { score: moveScore } = findBestMove(state.board, state.currentPlayer, 2)
      // Convert to positive probability
      policy[col] = Math.exp(moveScore / 500)
    }

    // Normalize policy (only over valid moves)
    const sum = policy.reduce((a, b) => a + b, 0)
    if (sum > 0) {
      for (const col of validMoves) {
        policy[col] /= sum
      }
    } else if (validMoves.length > 0) {
      // Uniform distribution over valid moves
      for (const col of validMoves) {
        policy[col] = 1 / validMoves.length
      }
    }

    const inferenceTimeMs = performance.now() - startTime

    return {
      value: normalizedValue,
      policy,
      confidence: 0.6, // Moderate confidence for heuristic
      inferenceTimeMs,
    }
  }

  dispose(): void {
    // Nothing to dispose for heuristic agent
  }
}

/**
 * Neural agent factory implementation.
 */
export class NeuralModelFactory implements NeuralAgentFactory {
  private config: Partial<NeuralInferenceConfig>

  constructor(config: Partial<NeuralInferenceConfig> = {}) {
    this.config = config
  }

  async create(modelId: string): Promise<NeuralAgent> {
    // Special case: heuristic model is built-in
    if (modelId === 'heuristic-v1') {
      return new HeuristicNeuralAgent()
    }

    // Find model metadata
    const metadata = MODEL_REGISTRY.find((m) => m.id === modelId)
    if (!metadata) {
      throw new Error(`Model not found: ${modelId}`)
    }

    // Create ONNX agent
    return createOnnxAgent(metadata, this.config)
  }

  async listModels(): Promise<ModelMetadata[]> {
    return [...MODEL_REGISTRY]
  }

  async isCached(modelId: string): Promise<boolean> {
    if (modelId === 'heuristic-v1') {
      return true // Always available
    }
    return isModelCached(modelId)
  }

  async clearCache(): Promise<void> {
    return clearModelCache()
  }
}

/**
 * Default model factory instance.
 */
let defaultFactory: NeuralModelFactory | null = null

/**
 * Gets the default model factory.
 */
export function getModelFactory(): NeuralModelFactory {
  if (!defaultFactory) {
    defaultFactory = new NeuralModelFactory()
  }
  return defaultFactory
}

/**
 * Creates a neural agent with the specified model.
 * Convenience function that uses the default factory.
 *
 * @param modelId - The model ID (default: 'heuristic-v1')
 * @returns Promise resolving to a neural agent
 */
export async function createNeuralAgent(modelId = 'heuristic-v1'): Promise<NeuralAgent> {
  const factory = getModelFactory()
  return factory.create(modelId)
}

/**
 * Lists all available models.
 * Convenience function that uses the default factory.
 */
export async function listAvailableModels(): Promise<ModelMetadata[]> {
  const factory = getModelFactory()
  return factory.listModels()
}

/**
 * Registers a custom model in the registry.
 * Useful for adding models dynamically or from API.
 */
export function registerModel(metadata: ModelMetadata): void {
  // Check if model already exists
  const existingIndex = MODEL_REGISTRY.findIndex((m) => m.id === metadata.id)
  if (existingIndex >= 0) {
    MODEL_REGISTRY[existingIndex] = metadata
  } else {
    MODEL_REGISTRY.push(metadata)
  }
}

/**
 * Loads models from an API endpoint.
 * Merges with local registry.
 *
 * @param apiUrl - URL to fetch model list from (defaults to /api/models)
 */
export async function loadModelsFromApi(apiUrl: string = MODELS_API_URL): Promise<void> {
  try {
    const response = await fetch(apiUrl)
    if (!response.ok) {
      throw new Error(`Failed to fetch models: ${response.status}`)
    }

    const data = await response.json()
    const models: ModelMetadata[] = data.models || data
    for (const model of models) {
      registerModel(model)
    }
    registryInitialized = true
  } catch (error) {
    console.warn('Failed to load models from API:', error)
    // Mark as initialized even on failure to prevent repeated attempts
    registryInitialized = true
  }
}

/**
 * Initialize the model registry by fetching from the API.
 * Safe to call multiple times - only fetches once.
 */
export async function initializeModelRegistry(): Promise<void> {
  if (registryInitialized) {
    return
  }

  if (initializationPromise) {
    return initializationPromise
  }

  initializationPromise = loadModelsFromApi()
  await initializationPromise
  initializationPromise = null
}

/**
 * Check if the registry has been initialized.
 */
export function isRegistryInitialized(): boolean {
  return registryInitialized
}

/** Cached neural agents for reuse */
const agentCache = new Map<string, NeuralAgent>()

/**
 * Gets or creates a cached neural agent.
 *
 * @param modelId - Model ID to get agent for
 * @returns Promise resolving to a neural agent
 */
export async function getCachedAgent(modelId = 'heuristic-v1'): Promise<NeuralAgent> {
  let agent = agentCache.get(modelId)
  if (!agent) {
    agent = await createNeuralAgent(modelId)
    agentCache.set(modelId, agent)
  }
  return agent
}

/**
 * Disposes all cached neural agents to free resources.
 * Should be called when agents are no longer needed.
 */
export function disposeNeuralAgents(): void {
  for (const agent of agentCache.values()) {
    agent.dispose()
  }
  agentCache.clear()
}
