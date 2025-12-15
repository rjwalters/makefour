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
 */
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
  // Future models will be added here as they are trained
  // {
  //   id: 'mlp-tiny-v1',
  //   name: 'Tiny MLP',
  //   architecture: 'mlp',
  //   expectedElo: 1400,
  //   sizeBytes: 50000,
  //   url: '/models/mlp-tiny-v1.onnx',
  //   version: '1.0.0',
  //   encoding: 'flat-binary',
  //   training: {
  //     games: 100000,
  //     epochs: 50,
  //     date: '2024-01-01',
  //   },
  // },
]

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
 * @param apiUrl - URL to fetch model list from
 */
export async function loadModelsFromApi(apiUrl: string): Promise<void> {
  try {
    const response = await fetch(apiUrl)
    if (!response.ok) {
      throw new Error(`Failed to fetch models: ${response.status}`)
    }

    const models: ModelMetadata[] = await response.json()
    for (const model of models) {
      registerModel(model)
    }
  } catch (error) {
    console.warn('Failed to load models from API:', error)
  }
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
