/**
 * Neural Model Weights API endpoint
 *
 * GET /api/models/:id/weights - Download model weights file from R2
 *
 * Some ONNX models store weights in a separate .onnx.data file.
 * This endpoint serves that file.
 */

import { errorResponse } from '../../../lib/auth'

interface Env {
  MODELS: R2Bucket
}

/**
 * GET /api/models/:id/weights - Get model weights file
 *
 * Streams the .onnx.data file from R2 bucket.
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
    // Get the external weights file
    const dataObject = await MODELS.get(`${modelId}.onnx.data`)

    if (!dataObject) {
      return errorResponse('Weights file not found', 404)
    }

    return new Response(dataObject.body, {
      headers: {
        'Content-Type': 'application/octet-stream',
        'Content-Disposition': `attachment; filename="${modelId}.onnx.data"`,
        'Cache-Control': 'public, max-age=31536000, immutable',
        'Access-Control-Allow-Origin': '*',
      },
    })
  } catch (error) {
    console.error(`GET /api/models/${modelId}/weights error:`, error)
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
