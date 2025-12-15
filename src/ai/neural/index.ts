/**
 * Neural Network Agent Module
 *
 * Provides neural network-based game evaluation and move selection
 * for Connect Four.
 *
 * @example
 * ```typescript
 * import { createNeuralAgent, listAvailableModels } from './ai/neural'
 *
 * // List available models
 * const models = await listAvailableModels()
 *
 * // Create an agent with the default model
 * const agent = await createNeuralAgent()
 *
 * // Get move suggestion
 * const move = await agent.suggestMove(gameState)
 *
 * // Get full evaluation
 * const evaluation = await agent.getFullEvaluation(gameState)
 * console.log(`Value: ${evaluation.value}, Best move: ${evaluation.policy}`)
 * ```
 */

// Core interface types
export type {
  NeuralAgent,
  NeuralAgentFactory,
  NeuralEvaluation,
  ModelMetadata,
  EncodedPosition,
  PositionEncodingType,
  NeuralInferenceConfig,
  ModelComparisonResult,
} from './interface'

export { DEFAULT_INFERENCE_CONFIG } from './interface'

// Position encoding utilities
export {
  encodePosition,
  encodeOneHot,
  encodeBitboard,
  encodeFlatBinary,
  decodeOneHot,
  maskInvalidMoves,
  sampleFromPolicy,
  getInputShape,
  getInputSize,
} from './encoding'

// ONNX runtime wrapper
export {
  OnnxNeuralAgent,
  createOnnxAgent,
  clearModelCache,
  isModelCached,
} from './onnx'

// Model loading and factory
export {
  HeuristicNeuralAgent,
  NeuralModelFactory,
  getModelFactory,
  createNeuralAgent,
  listAvailableModels,
  registerModel,
  loadModelsFromApi,
  getCachedAgent,
  disposeNeuralAgents,
} from './loader'
