/**
 * ONNX Runtime Wrapper for Neural Network Inference
 *
 * Provides browser-based inference using ONNX Runtime Web.
 * Supports loading models from URLs with IndexedDB caching.
 */

import type { GameState } from '../../game/makefour'
import {
  type NeuralAgent,
  type NeuralEvaluation,
  type ModelMetadata,
  type NeuralInferenceConfig,
  DEFAULT_INFERENCE_CONFIG,
} from './interface'
import { encodePosition, maskInvalidMoves, sampleFromPolicy } from './encoding'

/**
 * ONNX inference session type (from onnxruntime-web).
 * We use dynamic import to avoid loading the library until needed.
 */
interface InferenceSession {
  run(feeds: Record<string, unknown>): Promise<Record<string, { data: Float32Array | number[] }>>
  dispose(): void
}

/**
 * ONNX Runtime module interface.
 */
interface OnnxRuntimeModule {
  InferenceSession: {
    create(uri: string | ArrayBuffer): Promise<InferenceSession>
  }
  Tensor: new (type: string, data: Float32Array, dims: number[]) => unknown
}

/** Cached ONNX runtime module */
let onnxRuntime: OnnxRuntimeModule | null = null

/**
 * Lazily loads the ONNX runtime module.
 * Uses dynamic import to avoid bundling unless needed.
 */
async function getOnnxRuntime(): Promise<OnnxRuntimeModule> {
  if (onnxRuntime) {
    return onnxRuntime
  }

  try {
    // Dynamic import of onnxruntime-web
    const ort = await import('onnxruntime-web')
    onnxRuntime = ort as unknown as OnnxRuntimeModule
    return onnxRuntime
  } catch (error) {
    throw new Error(
      'Failed to load ONNX Runtime. Make sure onnxruntime-web is installed: npm install onnxruntime-web'
    )
  }
}

/**
 * IndexedDB storage for model caching.
 */
const MODEL_CACHE_DB = 'neural-model-cache'
const MODEL_CACHE_STORE = 'models'
const MODEL_CACHE_VERSION = 1

/**
 * Opens the IndexedDB for model caching.
 */
async function openModelCache(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(MODEL_CACHE_DB, MODEL_CACHE_VERSION)

    request.onerror = () => reject(request.error)
    request.onsuccess = () => resolve(request.result)

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result
      if (!db.objectStoreNames.contains(MODEL_CACHE_STORE)) {
        db.createObjectStore(MODEL_CACHE_STORE)
      }
    }
  })
}

/**
 * Gets a cached model from IndexedDB.
 */
async function getCachedModel(modelId: string): Promise<ArrayBuffer | null> {
  try {
    const db = await openModelCache()
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(MODEL_CACHE_STORE, 'readonly')
      const store = transaction.objectStore(MODEL_CACHE_STORE)
      const request = store.get(modelId)

      request.onerror = () => reject(request.error)
      request.onsuccess = () => resolve(request.result || null)

      transaction.oncomplete = () => db.close()
    })
  } catch {
    // IndexedDB not available, return null
    return null
  }
}

/**
 * Caches a model in IndexedDB.
 */
async function cacheModel(modelId: string, data: ArrayBuffer): Promise<void> {
  try {
    const db = await openModelCache()
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(MODEL_CACHE_STORE, 'readwrite')
      const store = transaction.objectStore(MODEL_CACHE_STORE)
      const request = store.put(data, modelId)

      request.onerror = () => reject(request.error)
      request.onsuccess = () => resolve()

      transaction.oncomplete = () => db.close()
    })
  } catch {
    // Caching failed, but we can continue without it
    console.warn('Failed to cache model in IndexedDB')
  }
}

/**
 * Downloads a model from a URL with caching.
 */
async function downloadModel(metadata: ModelMetadata): Promise<ArrayBuffer> {
  // Check cache first
  const cached = await getCachedModel(metadata.id)
  if (cached) {
    return cached
  }

  // Download from URL
  const response = await fetch(metadata.url)
  if (!response.ok) {
    throw new Error(`Failed to download model: ${response.status} ${response.statusText}`)
  }

  const data = await response.arrayBuffer()

  // Cache for future use
  await cacheModel(metadata.id, data)

  return data
}

/**
 * ONNX-based neural agent implementation.
 */
export class OnnxNeuralAgent implements NeuralAgent {
  private session: InferenceSession | null = null
  private metadata: ModelMetadata
  private _config: NeuralInferenceConfig // Reserved for future timeout/worker configuration
  private ort: OnnxRuntimeModule | null = null

  constructor(metadata: ModelMetadata, config: Partial<NeuralInferenceConfig> = {}) {
    this.metadata = metadata
    this._config = { ...DEFAULT_INFERENCE_CONFIG, ...config }
  }

  /**
   * Loads the model from URL or cache.
   */
  async load(): Promise<void> {
    if (this.session) {
      return // Already loaded
    }

    this.ort = await getOnnxRuntime()
    const modelData = await downloadModel(this.metadata)
    this.session = await this.ort.InferenceSession.create(modelData)
  }

  /**
   * Ensures the model is loaded before inference.
   */
  private async ensureLoaded(): Promise<void> {
    if (!this.session) {
      await this.load()
    }
  }

  isReady(): boolean {
    return this.session !== null
  }

  getMetadata(): ModelMetadata {
    return this.metadata
  }

  async evaluate(state: GameState): Promise<number> {
    const evaluation = await this.getFullEvaluation(state)
    return evaluation.value
  }

  async suggestMove(state: GameState): Promise<number> {
    const probabilities = await this.getMoveProbabilities(state)
    return sampleFromPolicy(probabilities, 0) // Deterministic selection
  }

  async getMoveProbabilities(state: GameState): Promise<number[]> {
    const evaluation = await this.getFullEvaluation(state)
    return evaluation.policy
  }

  async getFullEvaluation(state: GameState): Promise<NeuralEvaluation> {
    await this.ensureLoaded()

    if (!this.session || !this.ort) {
      throw new Error('Model not loaded')
    }

    const startTime = performance.now()

    if (this._config.debug) {
      console.log(`[Neural] Evaluating position with ${this.metadata.id}`)
    }

    // Encode the position
    const encoded = encodePosition(state, this.metadata.encoding)

    // Create input tensor
    const inputTensor = new this.ort.Tensor('float32', encoded.input, encoded.shape)

    // Run inference (input name is 'board' to match ONNX model export)
    const results = await this.session.run({ board: inputTensor })

    // Parse outputs - model should output 'value' and 'policy'
    const valueOutput = results.value?.data || results.output_value?.data
    const policyOutput = results.policy?.data || results.output_policy?.data

    // Handle missing outputs gracefully
    let value = 0
    if (valueOutput && valueOutput.length > 0) {
      value = Number(valueOutput[0])
      // Clamp to [-1, 1]
      value = Math.max(-1, Math.min(1, value))
    }

    let rawPolicy = Array(7).fill(1 / 7) // Default to uniform
    if (policyOutput && policyOutput.length >= 7) {
      rawPolicy = Array.from(policyOutput).slice(0, 7).map(Number)
    }

    // Mask invalid moves and normalize
    const policy = maskInvalidMoves(rawPolicy, state.board)

    const inferenceTimeMs = performance.now() - startTime

    // Confidence based on value magnitude and inference success
    const confidence = Math.min(1, Math.abs(value) * 0.5 + 0.5)

    return {
      value,
      policy,
      confidence,
      inferenceTimeMs,
    }
  }

  dispose(): void {
    if (this.session) {
      this.session.dispose()
      this.session = null
    }
  }
}

/**
 * Creates an ONNX neural agent from model metadata.
 */
export async function createOnnxAgent(
  metadata: ModelMetadata,
  config?: Partial<NeuralInferenceConfig>
): Promise<NeuralAgent> {
  const agent = new OnnxNeuralAgent(metadata, config)
  await agent.load()
  return agent
}

/**
 * Clears the model cache.
 */
export async function clearModelCache(): Promise<void> {
  try {
    const db = await openModelCache()
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(MODEL_CACHE_STORE, 'readwrite')
      const store = transaction.objectStore(MODEL_CACHE_STORE)
      const request = store.clear()

      request.onerror = () => reject(request.error)
      request.onsuccess = () => resolve()

      transaction.oncomplete = () => db.close()
    })
  } catch {
    // Cache clear failed, but that's okay
  }
}

/**
 * Checks if a model is cached.
 */
export async function isModelCached(modelId: string): Promise<boolean> {
  const cached = await getCachedModel(modelId)
  return cached !== null
}
