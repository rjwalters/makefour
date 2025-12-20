/**
 * Neural Model Data API endpoint
 *
 * GET /api/models/:id/data - Download model file from R2
 */

import { errorResponse } from '../../../lib/auth'

interface Env {
  MODELS: R2Bucket
}

/**
 * GET /api/models/:id/data - Get model ONNX file
 *
 * Streams the model file from R2 bucket.
 * No authentication required.
 */
export async function onRequestGet(context: EventContext<Env, any, { id: string }>) {
  const { MODELS } = context.env
  const modelId = context.params.id

  // Validate model ID format (alphanumeric with hyphens)
  if (!/^[a-z0-9-]+$/i.test(modelId)) {
    return errorResponse('Invalid model ID', 400)
  }

  try {
    // Try to get the main ONNX file
    const onnxObject = await MODELS.get(`${modelId}.onnx`)

    if (!onnxObject) {
      return errorResponse('Model not found', 404)
    }

    // Check if there's a separate data file (external weights)
    const dataObject = await MODELS.get(`${modelId}.onnx.data`)

    // For now, just return the main ONNX file
    // The client will need to handle fetching the data file separately if needed
    return new Response(onnxObject.body, {
      headers: {
        'Content-Type': 'application/octet-stream',
        'Content-Disposition': `attachment; filename="${modelId}.onnx"`,
        'Cache-Control': 'public, max-age=31536000, immutable',
        'Access-Control-Allow-Origin': '*',
        // Include info about data file in header
        'X-Has-External-Data': dataObject ? 'true' : 'false',
      },
    })
  } catch (error) {
    console.error(`GET /api/models/${modelId}/data error:`, error)
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
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  })
}
