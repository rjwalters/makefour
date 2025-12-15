/**
 * Neural Models API endpoint
 *
 * GET /api/models - List available neural network models
 */

import { jsonResponse, errorResponse } from '../../lib/auth'
import { applyRateLimit } from '../../lib/rateLimit'

interface Env {
  DB: D1Database
  RATE_LIMITER: DurableObjectNamespace
}

/**
 * Model metadata returned by the API.
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
 * The heuristic-v1 model is built-in and always available.
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
  // Models can also be served from cloud storage (R2, etc.)
]

/**
 * GET /api/models - List available neural network models
 *
 * Returns a list of all available models with their metadata.
 * No authentication required.
 */
export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB, RATE_LIMITER } = context.env

  try {
    // Apply rate limiting (public endpoint, but still limit abuse)
    const rateLimitResult = await applyRateLimit(context.request, {
      rateLimiter: RATE_LIMITER,
      db: DB,
      limit: 60,
      window: 60,
    })

    if (!rateLimitResult.allowed) {
      return errorResponse('Rate limit exceeded', 429)
    }

    // Return the model registry
    return jsonResponse({
      models: MODEL_REGISTRY,
      count: MODEL_REGISTRY.length,
    })
  } catch (error) {
    console.error('GET /api/models error:', error)
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
