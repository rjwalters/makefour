/**
 * Rate limiting utilities using Cloudflare KV storage
 *
 * Implements a sliding window rate limiter with configurable limits
 * for different endpoints and identifiers (IP, user ID, email).
 */

interface RateLimitConfig {
  limit: number // Max requests allowed
  windowMs: number // Time window in milliseconds
  keyPrefix: string // Prefix for KV key (e.g., 'login', 'register')
}

interface RateLimitResult {
  allowed: boolean
  limit: number
  remaining: number
  resetAt: number // Unix timestamp when window resets
  retryAfter?: number // Seconds until next allowed request (only on blocked)
}

interface RateLimitData {
  count: number
  windowStart: number
}

// Rate limit configurations for different endpoints
export const RATE_LIMITS: Record<string, RateLimitConfig> = {
  // Auth endpoints (strict)
  login: {
    limit: 5,
    windowMs: 60 * 1000, // 1 minute
    keyPrefix: 'rl:login',
  },
  register: {
    limit: 3,
    windowMs: 60 * 60 * 1000, // 1 hour
    keyPrefix: 'rl:register',
  },
  forgotPassword: {
    limit: 3,
    windowMs: 60 * 60 * 1000, // 1 hour
    keyPrefix: 'rl:forgot',
  },

  // Game endpoints (moderate)
  createGame: {
    limit: 60,
    windowMs: 60 * 60 * 1000, // 1 hour
    keyPrefix: 'rl:create-game',
  },
  getGames: {
    limit: 100,
    windowMs: 60 * 1000, // 1 minute
    keyPrefix: 'rl:get-games',
  },

  // Global limit
  global: {
    limit: 1000,
    windowMs: 60 * 1000, // 1 minute
    keyPrefix: 'rl:global',
  },
}

/**
 * Check and update rate limit for a given identifier
 *
 * Uses sliding window algorithm:
 * - Stores count and window start time in KV
 * - If within window, increments count
 * - If window expired, resets count
 *
 * @param kv - KV namespace for rate limit storage
 * @param configKey - Key from RATE_LIMITS (e.g., 'login', 'global')
 * @param identifier - Unique identifier (IP address, user ID, or email)
 * @returns RateLimitResult with allowed status and headers info
 */
export async function checkRateLimit(
  kv: KVNamespace,
  configKey: string,
  identifier: string
): Promise<RateLimitResult> {
  const config = RATE_LIMITS[configKey]
  if (!config) {
    // Unknown config, allow by default
    return { allowed: true, limit: 0, remaining: 0, resetAt: 0 }
  }

  const key = `${config.keyPrefix}:${identifier}`
  const now = Date.now()

  // Get current rate limit data
  const stored = await kv.get<RateLimitData>(key, 'json')

  let data: RateLimitData
  if (!stored || now - stored.windowStart >= config.windowMs) {
    // No data or window expired, start fresh
    data = { count: 1, windowStart: now }
  } else {
    // Within window, increment count
    data = { count: stored.count + 1, windowStart: stored.windowStart }
  }

  const resetAt = data.windowStart + config.windowMs
  const remaining = Math.max(0, config.limit - data.count)
  const allowed = data.count <= config.limit

  // Calculate TTL for KV entry (window duration + small buffer)
  const ttlSeconds = Math.ceil(config.windowMs / 1000) + 60

  // Store updated data
  await kv.put(key, JSON.stringify(data), { expirationTtl: ttlSeconds })

  const result: RateLimitResult = {
    allowed,
    limit: config.limit,
    remaining: allowed ? remaining : 0,
    resetAt: Math.ceil(resetAt / 1000), // Convert to Unix timestamp
  }

  if (!allowed) {
    result.retryAfter = Math.ceil((resetAt - now) / 1000)
  }

  return result
}

/**
 * Create rate limit response headers
 */
export function rateLimitHeaders(result: RateLimitResult): Record<string, string> {
  const headers: Record<string, string> = {
    'X-RateLimit-Limit': result.limit.toString(),
    'X-RateLimit-Remaining': result.remaining.toString(),
    'X-RateLimit-Reset': result.resetAt.toString(),
  }

  if (result.retryAfter !== undefined) {
    headers['Retry-After'] = result.retryAfter.toString()
  }

  return headers
}

/**
 * Create a 429 Too Many Requests response
 */
export function rateLimitResponse(result: RateLimitResult): Response {
  return new Response(
    JSON.stringify({
      error: 'Too many requests',
      retry_after: result.retryAfter,
      limit: result.limit,
      remaining: 0,
    }),
    {
      status: 429,
      headers: {
        'Content-Type': 'application/json',
        ...rateLimitHeaders(result),
      },
    }
  )
}

/**
 * Get client IP address from request
 * Cloudflare provides the real client IP in CF-Connecting-IP header
 */
export function getClientIP(request: Request): string {
  return (
    request.headers.get('CF-Connecting-IP') ||
    request.headers.get('X-Forwarded-For')?.split(',')[0]?.trim() ||
    'unknown'
  )
}

/**
 * Apply rate limit headers to a response
 */
export function applyRateLimitHeaders(response: Response, result: RateLimitResult): Response {
  const newResponse = new Response(response.body, response)
  const headers = rateLimitHeaders(result)

  for (const [key, value] of Object.entries(headers)) {
    newResponse.headers.set(key, value)
  }

  return newResponse
}
