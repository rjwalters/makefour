/**
 * Test utilities for API endpoint testing
 *
 * Provides:
 * - D1Database mock implementation
 * - Helper functions for creating test users and sessions
 * - Request/Response helpers for testing handlers
 */

import * as bcrypt from 'bcryptjs'

// Types for database row structure
export interface UserRow {
  id: string
  email: string
  email_verified: number
  password_hash: string | null
  oauth_provider: string | null
  oauth_id: string | null
  encrypted_dek: string | null
  rating: number
  games_played: number
  wins: number
  losses: number
  draws: number
  preferences: string
  created_at: number
  last_login: number
  updated_at: number
}

export interface SessionRow {
  id: string
  user_id: string
  expires_at: number
  created_at: number
}

export interface GameRow {
  id: string
  user_id: string
  outcome: string
  moves: string
  move_count: number
  rating_change: number | null
  opponent_type: string
  ai_difficulty: string | null
  player_number: number
  created_at: number
}

export interface VerificationTokenRow {
  id: string
  user_id: string
  expires_at: number
  used: number
  created_at: number
}

/**
 * In-memory mock implementation of D1Database
 */
export class MockD1Database implements D1Database {
  private users: Map<string, UserRow> = new Map()
  private sessions: Map<string, SessionRow> = new Map()
  private games: Map<string, GameRow> = new Map()
  private verificationTokens: Map<string, VerificationTokenRow> = new Map()

  // Track prepared statements for verification
  private preparedStatements: string[] = []

  prepare(query: string): D1PreparedStatement {
    this.preparedStatements.push(query)
    return new MockD1PreparedStatement(query, this)
  }

  dump(): Promise<ArrayBuffer> {
    throw new Error('Not implemented in mock')
  }

  batch<T = unknown>(statements: D1PreparedStatement[]): Promise<D1Result<T>[]> {
    return Promise.all(statements.map((stmt) => (stmt as MockD1PreparedStatement).run() as Promise<D1Result<T>>))
  }

  exec(query: string): Promise<D1ExecResult> {
    throw new Error('Not implemented in mock')
  }

  // Internal methods for the mock
  _getUsers(): Map<string, UserRow> {
    return this.users
  }

  _getSessions(): Map<string, SessionRow> {
    return this.sessions
  }

  _getGames(): Map<string, GameRow> {
    return this.games
  }

  _addUser(user: UserRow): void {
    this.users.set(user.id, user)
  }

  _addSession(session: SessionRow): void {
    this.sessions.set(session.id, session)
  }

  _addGame(game: GameRow): void {
    this.games.set(game.id, game)
  }

  _getVerificationTokens(): Map<string, VerificationTokenRow> {
    return this.verificationTokens
  }

  _addVerificationToken(token: VerificationTokenRow): void {
    this.verificationTokens.set(token.id, token)
  }

  _reset(): void {
    this.users.clear()
    this.sessions.clear()
    this.games.clear()
    this.verificationTokens.clear()
    this.preparedStatements = []
  }

  _getPreparedStatements(): string[] {
    return this.preparedStatements
  }
}

class MockD1PreparedStatement implements D1PreparedStatement {
  private query: string
  private db: MockD1Database
  private boundParams: unknown[] = []

  constructor(query: string, db: MockD1Database) {
    this.query = query
    this.db = db
  }

  bind(...values: unknown[]): D1PreparedStatement {
    this.boundParams = values
    return this
  }

  async first<T = unknown>(colName?: string): Promise<T | null> {
    const results = await this.all<T>()
    if (results.results.length === 0) return null
    if (colName) {
      return (results.results[0] as Record<string, unknown>)[colName] as T
    }
    return results.results[0]
  }

  async all<T = unknown>(): Promise<D1Result<T>> {
    const results = this.executeQuery<T>()
    return {
      results,
      success: true,
      meta: {
        duration: 0,
        size_after: 0,
        rows_read: results.length,
        rows_written: 0,
        last_row_id: 0,
        changed_db: false,
        changes: 0,
      },
    }
  }

  async run(): Promise<D1Result<unknown>> {
    this.executeQuery()
    return {
      results: [],
      success: true,
      meta: {
        duration: 0,
        size_after: 0,
        rows_read: 0,
        rows_written: 1,
        last_row_id: 0,
        changed_db: true,
        changes: 1,
      },
    }
  }

  async raw<T = unknown[]>(): Promise<T[]> {
    // Drizzle uses raw() to get results as arrays of values in SELECT column order
    const results = this.executeQuery<Record<string, unknown>>()

    // Parse column order from SELECT statement
    // Drizzle format: select "col1", "col2" from "table" OR select "table"."col1" from "table"
    const selectMatch = this.query.match(/select\s+(.*?)\s+from/is)
    if (!selectMatch) {
      return results.map(row => Object.values(row)) as T[]
    }

    const selectClause = selectMatch[1]

    // Match patterns like "column" or "table"."column" and extract just the column name
    // This regex finds column names: either standalone "col" or after "table"."col"
    const columnPattern = /(?:"\w+"\.)?"(\w+)"/g
    const columns: string[] = []
    let match: RegExpExecArray | null

    while ((match = columnPattern.exec(selectClause)) !== null) {
      const colName = match[1]
      // Convert snake_case to camelCase for Drizzle schema field names
      const camelName = colName.replace(/_([a-z])/g, (_, c) => c.toUpperCase())
      columns.push(camelName)
    }

    // If we couldn't parse columns, fall back to Object.values
    if (columns.length === 0) {
      return results.map(row => Object.values(row)) as T[]
    }

    // Return values in the column order
    return results.map(row => columns.map(col => row[col])) as T[]
  }

  private executeQuery<T>(): T[] {
    const q = this.query.toLowerCase().trim()

    // Handle SELECT queries
    if (q.startsWith('select')) {
      return this.handleSelect<T>()
    }

    // Handle INSERT queries
    if (q.startsWith('insert')) {
      this.handleInsert()
      return []
    }

    // Handle UPDATE queries
    if (q.startsWith('update')) {
      this.handleUpdate()
      return []
    }

    // Handle DELETE queries
    if (q.startsWith('delete')) {
      this.handleDelete()
      return []
    }

    return []
  }

  private handleSelect<T>(): T[] {
    const q = this.query.toLowerCase()

    // Users table - handle both raw SQL and Drizzle formats
    if (q.includes('from "users"') || q.includes('from users')) {
      const users = this.db._getUsers()

      // Helper to convert user to camelCase
      const toCamelCase = (user: UserRow) => ({
        id: user.id,
        email: user.email,
        emailVerified: user.email_verified,
        passwordHash: user.password_hash,
        oauthProvider: user.oauth_provider,
        oauthId: user.oauth_id,
        encryptedDek: user.encrypted_dek,
        rating: user.rating,
        gamesPlayed: user.games_played,
        wins: user.wins,
        losses: user.losses,
        draws: user.draws,
        preferences: user.preferences,
        isBot: 0,
        botPersonaId: null,
        username: null,
        usernameChangedAt: null,
        createdAt: user.created_at,
        lastLogin: user.last_login,
        updatedAt: user.updated_at,
      })

      // Check for WHERE clause patterns (Drizzle uses "table"."column" format)
      const whereClause = q.includes('where') ? q.substring(q.indexOf('where')) : ''

      // By email (Drizzle: where "users"."email" = ?, Raw: where email =)
      if (whereClause.includes('"email"') || whereClause.includes('email =')) {
        const email = this.boundParams[0] as string
        for (const user of users.values()) {
          if (user.email === email) {
            return [toCamelCase(user) as unknown as T]
          }
        }
        return []
      }

      // By id (Drizzle: where "users"."id" = ?, Raw: where id =)
      if (whereClause.includes('"id"') || whereClause.includes('id =')) {
        const id = this.boundParams[0] as string
        const user = users.get(id)
        return user ? [toCamelCase(user) as unknown as T] : []
      }

      // All users (for leaderboard)
      return Array.from(users.values()).map(toCamelCase) as unknown as T[]
    }

    // Session tokens table - handle both formats
    if (q.includes('from "session_tokens"') || q.includes('from session_tokens')) {
      const sessions = this.db._getSessions()

      if (q.includes('where')) {
        const id = this.boundParams[0] as string
        const session = sessions.get(id)
        if (session) {
          // Return camelCase keys for Drizzle
          return [{
            id: session.id,
            userId: session.user_id,
            expiresAt: session.expires_at,
            createdAt: session.created_at,
          } as unknown as T]
        }
        return []
      }

      return []
    }

    // Games table - handle both formats
    if (q.includes('from "games"') || q.includes('from games')) {
      const games = this.db._getGames()

      // Helper to convert game to camelCase
      const toCamelCase = (game: GameRow) => ({
        id: game.id,
        userId: game.user_id,
        outcome: game.outcome,
        moves: game.moves,
        moveCount: game.move_count,
        ratingChange: game.rating_change,
        opponentType: game.opponent_type,
        opponentId: null,
        aiDifficulty: game.ai_difficulty,
        botPersonaId: null,
        playerNumber: game.player_number,
        createdAt: game.created_at,
      })

      // Extract WHERE clause for more accurate matching
      const whereClause = q.includes('where') ? q.substring(q.indexOf('where')) : ''

      // Count games - check this FIRST before other patterns
      if (q.includes('count(*)') || q.includes('count(')) {
        const userId = this.boundParams[0] as string
        const count = Array.from(games.values()).filter((g) => g.user_id === userId).length
        return [{ count } as unknown as T]
      }

      // Single game by id and user_id - must have BOTH in WHERE clause
      if ((whereClause.includes('"id"') && whereClause.includes('"user_id"')) ||
          whereClause.includes('where id = ? and user_id = ?')) {
        const [gameId, userId] = this.boundParams as [string, string]
        const game = games.get(gameId)
        if (game && game.user_id === userId) {
          return [toCamelCase(game) as unknown as T]
        }
        return []
      }

      // Games by user_id with pagination - only user_id in WHERE
      if (whereClause.includes('"user_id"') || whereClause.includes('user_id =')) {
        const userId = this.boundParams[0] as string
        const limit = (this.boundParams[1] as number) || 100
        const offset = (this.boundParams[2] as number) || 0

        const userGames = Array.from(games.values())
          .filter((g) => g.user_id === userId)
          .sort((a, b) => b.created_at - a.created_at)
          .slice(offset, offset + limit)
          .map(toCamelCase)

        return userGames as unknown as T[]
      }
    }

    // Handle rating_history table (for batch insert)
    if (q.includes('into rating_history') || q.includes('into "rating_history"')) {
      // Just ignore for now - we don't track rating history in tests
      return []
    }

    // Email verification tokens table - handle both formats
    if (q.includes('from "email_verification_tokens"') || q.includes('from email_verification_tokens')) {
      const tokens = this.db._getVerificationTokens()

      // Helper to convert token to camelCase
      const toCamelCase = (token: VerificationTokenRow) => ({
        id: token.id,
        userId: token.user_id,
        expiresAt: token.expires_at,
        used: token.used,
        createdAt: token.created_at,
      })

      if (q.includes('where')) {
        const id = this.boundParams[0] as string
        const token = tokens.get(id)
        return token ? [toCamelCase(token) as unknown as T] : []
      }

      return []
    }

    return []
  }

  private handleInsert(): void {
    const q = this.query.toLowerCase()

    // Insert into users - handle both formats
    if (q.includes('into "users"') || q.includes('into users')) {
      // Drizzle sends all columns, we need to map them correctly
      // The order depends on the schema definition
      const params = this.boundParams
      const now = Date.now()

      this.db._addUser({
        id: params[0] as string,
        email: params[1] as string,
        email_verified: (params[2] as number) ?? 0,
        password_hash: params[3] as string | null,
        oauth_provider: params[4] as string | null,
        oauth_id: params[5] as string | null,
        encrypted_dek: params[6] as string | null,
        rating: (params[7] as number) ?? 1200,
        games_played: (params[8] as number) ?? 0,
        wins: (params[9] as number) ?? 0,
        losses: (params[10] as number) ?? 0,
        draws: (params[11] as number) ?? 0,
        preferences: (params[12] as string) ?? '{}',
        created_at: (params[15] as number) ?? now,
        last_login: (params[16] as number) ?? now,
        updated_at: (params[17] as number) ?? now,
      })
    }

    // Insert into session_tokens - handle both formats
    if (q.includes('into "session_tokens"') || q.includes('into session_tokens')) {
      const [id, user_id, expires_at, created_at] = this.boundParams as [string, string, number, number]
      this.db._addSession({ id, user_id, expires_at, created_at })
    }

    // Insert into games - handle both formats
    if (q.includes('into "games"') || q.includes('into games')) {
      const [id, user_id, outcome, moves, move_count, rating_change, opponent_type, _opponentId, ai_difficulty, _botPersonaId, player_number, created_at] =
        this.boundParams as [string, string, string, string, number, number, string, string | null, string | null, string | null, number, number]
      this.db._addGame({
        id,
        user_id,
        outcome,
        moves,
        move_count,
        rating_change,
        opponent_type,
        ai_difficulty,
        player_number,
        created_at,
      })
    }

    // Insert into email_verification_tokens - handle both formats
    if (q.includes('into "email_verification_tokens"') || q.includes('into email_verification_tokens')) {
      const [id, user_id, expires_at, used, created_at] = this.boundParams as [string, string, number, number, number]
      this.db._addVerificationToken({
        id,
        user_id,
        expires_at,
        used,
        created_at,
      })
    }
  }

  private handleUpdate(): void {
    const q = this.query.toLowerCase()

    // Update users - handle both formats
    if (q.includes('update "users"') || q.includes('update users')) {
      const users = this.db._getUsers()
      const userId = this.boundParams[this.boundParams.length - 1] as string
      const user = users.get(userId)

      if (user) {
        // Update last_login
        if (q.includes('last_login') || q.includes('"last_login"')) {
          user.last_login = this.boundParams[0] as number
          if (this.boundParams.length > 2) {
            user.updated_at = this.boundParams[1] as number
          }
        }

        // Update rating and stats - Drizzle sends absolute values, not increments
        if (q.includes('rating') || q.includes('"rating"')) {
          const [rating, gamesPlayed, wins, losses, draws, updated_at] = this.boundParams as [
            number,
            number,
            number,
            number,
            number,
            number,
            string,
          ]
          user.rating = rating
          user.games_played = gamesPlayed
          user.wins = wins
          user.losses = losses
          user.draws = draws
          user.updated_at = updated_at
        }

        // Update email_verified
        if (q.includes('email_verified') || q.includes('"email_verified"')) {
          user.email_verified = 1
          user.updated_at = this.boundParams[0] as number
        }
      }
    }

    // Update email_verification_tokens - handle both formats
    if (q.includes('update "email_verification_tokens"') || q.includes('update email_verification_tokens')) {
      if (q.includes('used') || q.includes('"used"')) {
        const tokenId = this.boundParams[this.boundParams.length - 1] as string
        const tokens = this.db._getVerificationTokens()
        const token = tokens.get(tokenId)
        if (token) {
          token.used = 1
        }
      }
    }
  }

  private handleDelete(): void {
    const q = this.query.toLowerCase()

    // Delete from session_tokens - handle both formats
    if (q.includes('from "session_tokens"') || q.includes('from session_tokens')) {
      const sessions = this.db._getSessions()
      const id = this.boundParams[0] as string
      sessions.delete(id)
    }

    // Delete from email_verification_tokens - handle both formats
    if (q.includes('from "email_verification_tokens"') || q.includes('from email_verification_tokens')) {
      const tokens = this.db._getVerificationTokens()
      const userId = this.boundParams[0] as string
      // Delete all unused tokens for user
      for (const [tokenId, token] of tokens) {
        if (token.user_id === userId && token.used === 0) {
          tokens.delete(tokenId)
        }
      }
    }

    // Delete from users - handle both formats
    if (q.includes('from "users"') || q.includes('from users')) {
      const users = this.db._getUsers()
      const id = this.boundParams[0] as string
      users.delete(id)
    }
  }
}

/**
 * Creates a test user in the mock database
 */
export async function createTestUser(
  db: MockD1Database,
  overrides: Partial<UserRow> = {}
): Promise<{ user: UserRow; password: string }> {
  const password = 'TestPassword123!'
  const password_hash = await bcrypt.hash(password, 10)
  const now = Date.now()

  const user: UserRow = {
    id: crypto.randomUUID(),
    email: `test-${crypto.randomUUID().slice(0, 8)}@example.com`,
    email_verified: 1,
    password_hash,
    oauth_provider: null,
    oauth_id: null,
    encrypted_dek: 'test-encrypted-dek',
    rating: 1200,
    games_played: 0,
    wins: 0,
    losses: 0,
    draws: 0,
    preferences: '{}',
    created_at: now,
    last_login: now,
    updated_at: now,
    ...overrides,
  }

  db._addUser(user)
  return { user, password }
}

/**
 * Creates a valid session token for a user
 */
export function createTestSession(db: MockD1Database, userId: string, expiresInMs = 30 * 24 * 60 * 60 * 1000): string {
  const sessionId = crypto.randomUUID()
  const now = Date.now()

  db._addSession({
    id: sessionId,
    user_id: userId,
    expires_at: now + expiresInMs,
    created_at: now,
  })

  return sessionId
}

/**
 * Creates an expired session token for testing
 */
export function createExpiredSession(db: MockD1Database, userId: string): string {
  const sessionId = crypto.randomUUID()
  const now = Date.now()

  db._addSession({
    id: sessionId,
    user_id: userId,
    expires_at: now - 1000, // Already expired
    created_at: now - 60000,
  })

  return sessionId
}

/**
 * Creates a test game in the mock database
 */
export function createTestGame(db: MockD1Database, userId: string, overrides: Partial<GameRow> = {}): GameRow {
  const game: GameRow = {
    id: crypto.randomUUID(),
    user_id: userId,
    outcome: 'win',
    moves: JSON.stringify([3, 2, 3, 2, 3, 2, 3]),
    move_count: 7,
    rating_change: 25,
    opponent_type: 'ai',
    ai_difficulty: 'intermediate',
    player_number: 1,
    created_at: Date.now(),
    ...overrides,
  }

  db._addGame(game)
  return game
}

/**
 * Creates a mock Request object
 */
export function createMockRequest(
  url: string,
  options: {
    method?: string
    body?: unknown
    headers?: Record<string, string>
  } = {}
): Request {
  const { method = 'GET', body, headers = {} } = options

  const init: RequestInit = {
    method,
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
  }

  if (body) {
    init.body = JSON.stringify(body)
  }

  return new Request(url, init)
}

/**
 * Creates an authenticated request with Bearer token
 */
export function createAuthenticatedRequest(
  url: string,
  sessionToken: string,
  options: {
    method?: string
    body?: unknown
    headers?: Record<string, string>
  } = {}
): Request {
  return createMockRequest(url, {
    ...options,
    headers: {
      ...options.headers,
      Authorization: `Bearer ${sessionToken}`,
    },
  })
}

/**
 * Creates a mock EventContext for Cloudflare Pages Functions
 */
export function createMockContext<Params extends Record<string, string> = Record<string, string>>(
  db: MockD1Database,
  request: Request,
  params: Params = {} as Params
): EventContext<{ DB: D1Database }, string, Params> {
  return {
    request,
    env: { DB: db as unknown as D1Database },
    params,
    waitUntil: () => {},
    passThroughOnException: () => {},
    next: () => Promise.resolve(new Response()),
    data: {},
    functionPath: '',
  }
}

/**
 * Helper to parse JSON response
 */
export async function parseResponse<T>(response: Response): Promise<T> {
  return response.json() as Promise<T>
}

/**
 * Helper to assert response status and parse body
 */
export async function expectResponse<T>(response: Response, expectedStatus: number): Promise<T> {
  if (response.status !== expectedStatus) {
    const body = await response.text()
    throw new Error(`Expected status ${expectedStatus}, got ${response.status}. Body: ${body}`)
  }
  return parseResponse<T>(response)
}

/**
 * Creates a test verification token in the mock database
 */
export function createTestVerificationToken(
  db: MockD1Database,
  userId: string,
  overrides: Partial<VerificationTokenRow> = {}
): VerificationTokenRow {
  const now = Date.now()
  const token: VerificationTokenRow = {
    id: crypto.randomUUID(),
    user_id: userId,
    expires_at: now + 24 * 60 * 60 * 1000, // 24 hours
    used: 0,
    created_at: now,
    ...overrides,
  }

  db._addVerificationToken(token)
  return token
}

/**
 * Creates an expired verification token for testing
 */
export function createExpiredVerificationToken(
  db: MockD1Database,
  userId: string
): VerificationTokenRow {
  return createTestVerificationToken(db, userId, {
    expires_at: Date.now() - 1000, // Already expired
  })
}

/**
 * Creates an unverified test user
 */
export async function createUnverifiedTestUser(
  db: MockD1Database,
  overrides: Partial<UserRow> = {}
): Promise<{ user: UserRow; password: string }> {
  return createTestUser(db, { email_verified: 0, ...overrides })
}
