/**
 * Middleware for Cloudflare Pages Functions
 * Adds CORS headers, rate limiting, and error handling
 */

import {
  checkRateLimit,
  getClientIP,
  rateLimitResponse,
  applyRateLimitHeaders,
  type RateLimitResult,
} from './lib/rateLimit'
import { validateSession } from './lib/auth'

interface Env {
  DB: D1Database
  RATE_LIMIT: KVNamespace
}

/**
 * Determine which rate limit rule applies based on URL path and method
 * Returns the config key and identifier to use
 */
async function getRateLimitRule(
  request: Request,
  env: Env,
  clientIP: string
): Promise<{ configKey: string; identifier: string } | null> {
  const url = new URL(request.url)
  const path = url.pathname
  const method = request.method

  // Auth endpoints - rate limit by IP
  if (path === '/api/auth/login' && method === 'POST') {
    return { configKey: 'login', identifier: clientIP }
  }

  if (path === '/api/auth/register' && method === 'POST') {
    return { configKey: 'register', identifier: clientIP }
  }

  if (path === '/api/auth/forgot-password' && method === 'POST') {
    // For forgot password, we try to extract email from body
    // but fall back to IP if we can't
    try {
      const body = await request.clone().json()
      if (body && typeof body.email === 'string') {
        return { configKey: 'forgotPassword', identifier: body.email.toLowerCase() }
      }
    } catch {
      // Couldn't parse body, use IP
    }
    return { configKey: 'forgotPassword', identifier: clientIP }
  }

  // Game endpoints - rate limit by user ID (requires auth) or IP for unauthenticated
  if (path === '/api/games') {
    if (method === 'POST') {
      // Try to get user ID from session
      const session = await validateSession(request, env.DB)
      const identifier = session.valid ? session.userId : clientIP
      return { configKey: 'createGame', identifier }
    }

    if (method === 'GET') {
      const session = await validateSession(request, env.DB)
      const identifier = session.valid ? session.userId : clientIP
      return { configKey: 'getGames', identifier }
    }
  }

  // No specific rule for this endpoint
  return null
}

export async function onRequest(context: EventContext<Env, any, any>) {
  const { request, env } = context

  // Skip rate limiting for OPTIONS (CORS preflight)
  if (request.method === 'OPTIONS') {
    const response = await context.next()
    response.headers.set('Access-Control-Allow-Origin', '*')
    response.headers.set('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
    response.headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    return response
  }

  // Only apply rate limiting to /api routes
  const url = new URL(request.url)
  const isApiRoute = url.pathname.startsWith('/api')

  if (!isApiRoute || !env.RATE_LIMIT) {
    // Not an API route or KV not configured, skip rate limiting
    const response = await context.next()
    response.headers.set('Access-Control-Allow-Origin', '*')
    response.headers.set('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
    response.headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    return response
  }

  const clientIP = getClientIP(request)
  let rateLimitResult: RateLimitResult | null = null

  // Skip global rate limiting for GET requests (reads) to reduce KV traffic
  // GET requests are cheap and frequent (polling). Rate limit mutations only.
  const isMutation = request.method !== 'GET'

  // 1. Check global rate limit only for mutations (POST/PUT/DELETE)
  if (isMutation) {
    const globalResult = await checkRateLimit(env.RATE_LIMIT, 'global', clientIP)
    if (!globalResult.allowed) {
      return rateLimitResponse(globalResult)
    }
    rateLimitResult = globalResult
  }

  // 2. Check endpoint-specific rate limit (only for mutations)
  if (isMutation) {
    const rule = await getRateLimitRule(request, env, clientIP)
    if (rule) {
      const specificResult = await checkRateLimit(env.RATE_LIMIT, rule.configKey, rule.identifier)
      if (!specificResult.allowed) {
        return rateLimitResponse(specificResult)
      }
      rateLimitResult = specificResult
    }
  }

  // Proceed with the request
  const response = await context.next()

  // Add CORS headers
  response.headers.set('Access-Control-Allow-Origin', '*')
  response.headers.set('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
  response.headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization')

  // Add rate limit headers to response (only for mutations that were rate limited)
  if (rateLimitResult) {
    return applyRateLimitHeaders(response, rateLimitResult)
  }

  return response
}
