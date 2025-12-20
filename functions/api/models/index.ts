/**
 * Neural Models API endpoint
 *
 * GET /api/models - List available neural network models
 */

import { jsonResponse } from '../../lib/auth'
import { MODEL_REGISTRY } from '../../lib/engines/neural-engine'

/**
 * GET /api/models - List available neural network models
 *
 * Returns a list of all available models with their metadata.
 * No authentication required.
 */
export async function onRequestGet() {
  // Return the model registry (static data, no rate limiting needed)
  return jsonResponse({
    models: MODEL_REGISTRY,
    count: MODEL_REGISTRY.length,
  })
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
