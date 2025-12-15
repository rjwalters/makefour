/**
 * Neural Model API endpoint
 *
 * GET /api/models/:id - Get model metadata by ID
 */

import { jsonResponse, errorResponse } from '../../lib/auth'
import { applyRateLimit } from '../../lib/rateLimit'

interface Env {
  DB: D1Database
  RATE_LIMITER: DurableObjectNamespace
}

/**
 * Model metadata.
 */
interface ModelMetadata {
  id: string
  name: string
  architecture: 'mlp' | 'cnn' | 'transformer'
  expectedElo: number
  sizeBytes: number
  url: string
  version: string
  encoding: 'onehot-6x7x3' | 'bitboard' | 'flat-binary'
  training?: {
    games: number
    epochs: number
    date: string
  }
}

/**
 * Registry of available neural network models.
 */
const MODEL_REGISTRY: Record<string, ModelMetadata> = {
  'heuristic-v1': {
    id: 'heuristic-v1',
    name: 'Heuristic Baseline',
    architecture: 'mlp',
    expectedElo: 1200,
    sizeBytes: 0,
    url: '',
    version: '1.0.0',
    encoding: 'flat-binary',
  },
}

/**
 * GET /api/models/:id - Get model metadata
 *
 * Returns metadata for a specific model.
 * No authentication required.
 */
export async function onRequestGet(context: EventContext<Env, any, { id: string }>) {
  const { DB, RATE_LIMITER } = context.env
  const modelId = context.params.id

  try {
    // Apply rate limiting
    const rateLimitResult = await applyRateLimit(context.request, {
      rateLimiter: RATE_LIMITER,
      db: DB,
      limit: 60,
      window: 60,
    })

    if (!rateLimitResult.allowed) {
      return errorResponse('Rate limit exceeded', 429)
    }

    // Look up the model
    const model = MODEL_REGISTRY[modelId]

    if (!model) {
      return errorResponse('Model not found', 404)
    }

    return jsonResponse(model)
  } catch (error) {
    console.error(`GET /api/models/${modelId} error:`, error)
    return errorResponse('Internal server error', 500)
  }
}

/**
 * Handle OPTIONS for CORS preflight
 */
export async function onRequestOptions() {
  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  })
}
