/**
 * Neural Coach Module
 *
 * Provides neural network-based AI coaching that integrates with the existing
 * difficulty system. This module wraps the neural agent infrastructure to provide
 * a compatible interface with coach.ts.
 */

import type { GameState } from '../game/makefour'
import { getValidMoves } from '../game/makefour'
import type { Position, Analysis } from './coach'
import type { NeuralAgent, ModelMetadata, NeuralEvaluation } from './neural'
import { createNeuralAgent, listAvailableModels } from './neural'

/**
 * Neural difficulty configuration.
 */
export interface NeuralDifficultyConfig {
  /** Model ID to use */
  modelId: string
  /** Temperature for move sampling (0 = deterministic, higher = more random) */
  temperature: number
  /** Error rate for introducing mistakes */
  errorRate: number
  /** Name for display */
  name: string
  /** Description */
  description: string
}

/**
 * Neural difficulty levels.
 * These can be used alongside or instead of traditional minimax levels.
 */
export const NEURAL_DIFFICULTY_LEVELS: Record<string, NeuralDifficultyConfig> = {
  'neural-easy': {
    modelId: 'heuristic-v1',
    temperature: 1.5,
    errorRate: 0.2,
    name: 'Neural Easy',
    description: 'Neural network with high randomness',
  },
  'neural-medium': {
    modelId: 'heuristic-v1',
    temperature: 0.5,
    errorRate: 0.1,
    name: 'Neural Medium',
    description: 'Neural network with moderate play',
  },
  'neural-hard': {
    modelId: 'heuristic-v1',
    temperature: 0,
    errorRate: 0,
    name: 'Neural Hard',
    description: 'Neural network at full strength',
  },
}

/** Cached neural agents by model ID */
const agentCache = new Map<string, NeuralAgent>()

/**
 * Gets or creates a neural agent for the given model.
 */
async function getAgent(modelId: string): Promise<NeuralAgent> {
  let agent = agentCache.get(modelId)
  if (!agent) {
    agent = await createNeuralAgent(modelId)
    agentCache.set(modelId, agent)
  }
  return agent
}

/**
 * Converts a Position to GameState format.
 */
function positionToGameState(position: Position): GameState {
  return {
    board: position.board,
    currentPlayer: position.currentPlayer,
    winner: null, // Assume game is ongoing
    moveHistory: position.moveHistory,
  }
}

/**
 * Suggests a move using a neural network.
 *
 * @param position - The current game position
 * @param config - Neural difficulty configuration
 * @returns Promise resolving to the suggested column (0-6)
 */
export async function suggestNeuralMove(
  position: Position,
  config: NeuralDifficultyConfig
): Promise<number> {
  const agent = await getAgent(config.modelId)
  const state = positionToGameState(position)

  // Check for random error
  if (config.errorRate > 0 && Math.random() < config.errorRate) {
    const validMoves = getValidMoves(position.board)
    return validMoves[Math.floor(Math.random() * validMoves.length)]
  }

  // Get move probabilities
  const probs = await agent.getMoveProbabilities(state)

  // Apply temperature sampling
  if (config.temperature > 0) {
    return sampleWithTemperature(probs, config.temperature)
  }

  // Deterministic: pick best move
  return probs.indexOf(Math.max(...probs))
}

/**
 * Samples a move from probabilities with temperature scaling.
 */
function sampleWithTemperature(probs: number[], temperature: number): number {
  const scaled = probs.map((p) => p ** (1 / temperature))
  const sum = scaled.reduce((a, b) => a + b, 0)
  const normalized = scaled.map((p) => p / sum)

  const rand = Math.random()
  let cumulative = 0
  for (let col = 0; col < normalized.length; col++) {
    cumulative += normalized[col]
    if (rand < cumulative) {
      return col
    }
  }
  return normalized.length - 1
}

/**
 * Analyzes a position using a neural network.
 *
 * @param position - The game position to analyze
 * @param modelId - Model ID to use
 * @returns Promise resolving to analysis results
 */
export async function analyzeWithNeural(
  position: Position,
  modelId = 'heuristic-v1'
): Promise<Analysis & { neuralEval: NeuralEvaluation }> {
  const agent = await getAgent(modelId)
  const state = positionToGameState(position)

  const evaluation = await agent.getFullEvaluation(state)
  const bestMove = evaluation.policy.indexOf(Math.max(...evaluation.policy))

  // Convert neural value to score (scale for display)
  const score = Math.round(evaluation.value * 1000)

  return {
    bestMove,
    score,
    evaluation: getNeuralEvaluationDescription(evaluation.value),
    theoreticalResult: getTheoreticalResult(evaluation.value),
    confidence: evaluation.confidence,
    neuralEval: evaluation,
  }
}

/**
 * Gets evaluation description from neural value.
 */
function getNeuralEvaluationDescription(value: number): string {
  if (value > 0.8) return 'Strongly winning position'
  if (value > 0.4) return 'Favorable position'
  if (value > 0.1) return 'Slightly favorable position'
  if (value > -0.1) return 'Roughly equal position'
  if (value > -0.4) return 'Slightly unfavorable position'
  if (value > -0.8) return 'Unfavorable position'
  return 'Strongly losing position'
}

/**
 * Determines theoretical result from neural value.
 */
function getTheoreticalResult(value: number): 'win' | 'loss' | 'draw' | 'unknown' {
  if (value > 0.9) return 'win'
  if (value < -0.9) return 'loss'
  if (Math.abs(value) < 0.05) return 'draw'
  return 'unknown'
}

/**
 * Gets all available neural models.
 */
export async function getAvailableNeuralModels(): Promise<ModelMetadata[]> {
  return listAvailableModels()
}

/**
 * Gets move rankings using neural network evaluation.
 *
 * @param position - The game position
 * @param modelId - Model ID to use
 * @returns Array of moves with probabilities and comments
 */
export async function rankNeuralMoves(
  position: Position,
  modelId = 'heuristic-v1'
): Promise<Array<{ column: number; probability: number; comment: string }>> {
  const agent = await getAgent(modelId)
  const state = positionToGameState(position)

  const evaluation = await agent.getFullEvaluation(state)
  const validMoves = getValidMoves(position.board)

  return validMoves
    .map((col) => ({
      column: col,
      probability: evaluation.policy[col],
      comment: getMoveComment(evaluation.policy[col], evaluation.policy),
    }))
    .sort((a, b) => b.probability - a.probability)
}

/**
 * Gets a comment for a move based on its probability.
 */
function getMoveComment(prob: number, allProbs: number[]): string {
  const maxProb = Math.max(...allProbs)

  if (prob === maxProb) return 'Best move'
  if (prob > maxProb * 0.8) return 'Strong alternative'
  if (prob > maxProb * 0.5) return 'Decent option'
  if (prob > maxProb * 0.2) return 'Weak move'
  return 'Poor move'
}

/**
 * Disposes all cached agents to free resources.
 */
export function disposeNeuralAgents(): void {
  for (const agent of agentCache.values()) {
    agent.dispose()
  }
  agentCache.clear()
}

/**
 * Checks if neural inference is available.
 * Returns true if at least the heuristic model is ready.
 */
export async function isNeuralAvailable(): Promise<boolean> {
  try {
    const agent = await getAgent('heuristic-v1')
    return agent.isReady()
  } catch {
    return false
  }
}
