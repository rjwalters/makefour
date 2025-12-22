/**
 * React hook for client-side neural bot inference.
 *
 * Loads ONNX models and runs inference locally in the browser,
 * eliminating the need for server-side neural computation.
 */

import { useState, useCallback, useEffect, useRef } from 'react'
import { getCachedAgent, type NeuralAgent } from '../ai/neural'
import type { Board } from '../game/makefour'
import { sampleFromPolicy } from '../ai/neural/encoding'

export interface NeuralBotState {
  isLoading: boolean
  isReady: boolean
  error: string | null
  modelId: string | null
  modelName: string | null
}

export interface NeuralMoveResult {
  column: number
  policy: number[]
  value: number
  confidence: number
  inferenceTimeMs: number
}

/**
 * Hook for using neural network bots on the client side.
 *
 * @param modelId - Optional model ID to load on mount
 * @returns Object with state and functions for neural inference
 */
export function useNeuralBot(modelId?: string) {
  const [state, setState] = useState<NeuralBotState>({
    isLoading: false,
    isReady: false,
    error: null,
    modelId: null,
    modelName: null,
  })

  const agentRef = useRef<NeuralAgent | null>(null)
  const currentModelIdRef = useRef<string | null>(null)

  /**
   * Load a neural model by ID.
   * Caches loaded models for reuse.
   */
  const loadModel = useCallback(async (id: string): Promise<boolean> => {
    // Skip if already loading or loaded this model
    if (currentModelIdRef.current === id && agentRef.current) {
      return true
    }

    setState((prev) => ({
      ...prev,
      isLoading: true,
      error: null,
    }))

    try {
      const agent = await getCachedAgent(id)
      agentRef.current = agent
      currentModelIdRef.current = id

      const metadata = agent.getMetadata()
      setState({
        isLoading: false,
        isReady: true,
        error: null,
        modelId: id,
        modelName: metadata.name,
      })
      return true
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load model'
      setState({
        isLoading: false,
        isReady: false,
        error: message,
        modelId: null,
        modelName: null,
      })
      return false
    }
  }, [])

  /**
   * Compute a move given the current board state.
   * Returns the selected column and additional info.
   */
  const computeMove = useCallback(
    async (
      board: Board,
      currentPlayer: 1 | 2,
      moves: number[],
      temperature = 0
    ): Promise<NeuralMoveResult | null> => {
      if (!agentRef.current) {
        console.error('Neural agent not loaded')
        return null
      }

      try {
        // Create game state for the neural agent
        const gameState = {
          board,
          currentPlayer,
          moveHistory: moves,
          winner: null,
        }

        const evaluation = await agentRef.current.getFullEvaluation(gameState)

        // Select move from policy
        const column = sampleFromPolicy(evaluation.policy, temperature)

        return {
          column,
          policy: evaluation.policy,
          value: evaluation.value,
          confidence: evaluation.confidence,
          inferenceTimeMs: evaluation.inferenceTimeMs,
        }
      } catch (error) {
        console.error('Neural inference error:', error)
        return null
      }
    },
    []
  )

  /**
   * Get move probabilities without selecting a move.
   * Useful for visualization or analysis.
   */
  const getMoveProbabilities = useCallback(
    async (board: Board, currentPlayer: 1 | 2, moves: number[]): Promise<number[] | null> => {
      if (!agentRef.current) {
        return null
      }

      try {
        const gameState = {
          board,
          currentPlayer,
          moveHistory: moves,
          winner: null,
        }

        return await agentRef.current.getMoveProbabilities(gameState)
      } catch (error) {
        console.error('Neural probability error:', error)
        return null
      }
    },
    []
  )

  /**
   * Check if the specified model is supported for client-side inference.
   */
  const isNeuralModel = useCallback((personaConfig: { ai_engine?: string }) => {
    return personaConfig.ai_engine === 'neural'
  }, [])

  // Load model on mount if modelId provided
  useEffect(() => {
    if (modelId) {
      loadModel(modelId)
    }
  }, [modelId, loadModel])

  return {
    ...state,
    loadModel,
    computeMove,
    getMoveProbabilities,
    isNeuralModel,
  }
}

/**
 * Prefetch and cache a neural model for faster loading later.
 *
 * @param modelId - Model ID to prefetch
 */
export async function prefetchNeuralModel(modelId: string): Promise<void> {
  try {
    await getCachedAgent(modelId)
  } catch (error) {
    console.warn(`Failed to prefetch model ${modelId}:`, error)
  }
}

/**
 * Get list of available neural models.
 */
export { listAvailableModels } from '../ai/neural'
