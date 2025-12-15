import { describe, it, expect, beforeEach } from 'vitest'
import { onRequestGet } from './me'
import {
  MockD1Database,
  createMockContext,
  createTestUser,
  createTestSession,
  createExpiredSession,
  createAuthenticatedRequest,
  createMockRequest,
} from '../../lib/test-utils'

describe('GET /api/auth/me', () => {
  let db: MockD1Database

  beforeEach(() => {
    db = new MockD1Database()
  })

  describe('returns user info', () => {
    it('returns user data for valid session', async () => {
      const { user } = await createTestUser(db, {
        email: 'me@example.com',
        rating: 1350,
        games_played: 10,
        wins: 6,
        losses: 3,
        draws: 1,
      })
      const sessionToken = createTestSession(db, user.id)

      const request = createAuthenticatedRequest(
        'https://example.com/api/auth/me',
        sessionToken
      )

      const context = createMockContext(db, request)
      const response = await onRequestGet(context)

      expect(response.status).toBe(200)

      const data = await response.json()
      expect(data.user).toBeDefined()
      expect(data.user.email).toBe('me@example.com')
      expect(data.user.id).toBe(user.id)
      expect(data.user.rating).toBe(1350)
      expect(data.user.gamesPlayed).toBe(10)
      expect(data.user.wins).toBe(6)
      expect(data.user.losses).toBe(3)
      expect(data.user.draws).toBe(1)
    })

    it('returns email_verified as boolean', async () => {
      const { user } = await createTestUser(db, { email_verified: 1 })
      const sessionToken = createTestSession(db, user.id)

      const request = createAuthenticatedRequest(
        'https://example.com/api/auth/me',
        sessionToken
      )

      const context = createMockContext(db, request)
      const response = await onRequestGet(context)

      const data = await response.json()
      expect(data.user.email_verified).toBe(true)
    })

    it('accepts session_token as query parameter', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)

      const request = createMockRequest(
        `https://example.com/api/auth/me?session_token=${sessionToken}`
      )

      const context = createMockContext(db, request)
      const response = await onRequestGet(context)

      expect(response.status).toBe(200)

      const data = await response.json()
      expect(data.user.id).toBe(user.id)
    })

    it('prefers Authorization header over query parameter', async () => {
      const { user: user1 } = await createTestUser(db)
      const { user: user2 } = await createTestUser(db)
      const sessionToken1 = createTestSession(db, user1.id)
      const sessionToken2 = createTestSession(db, user2.id)

      // Pass user2's token in query, but user1's token in header
      const request = createAuthenticatedRequest(
        `https://example.com/api/auth/me?session_token=${sessionToken2}`,
        sessionToken1
      )

      const context = createMockContext(db, request)
      const response = await onRequestGet(context)

      const data = await response.json()
      // Should return user1 (from header), not user2 (from query)
      expect(data.user.id).toBe(user1.id)
    })
  })

  describe('rejects unauthenticated requests', () => {
    it('returns 401 when no token provided', async () => {
      const request = createMockRequest('https://example.com/api/auth/me')

      const context = createMockContext(db, request)
      const response = await onRequestGet(context)

      expect(response.status).toBe(401)

      const data = await response.json()
      expect(data.error).toBe('Session token required')
    })

    it('returns 401 for invalid session token', async () => {
      const request = createAuthenticatedRequest(
        'https://example.com/api/auth/me',
        'invalid-token-123'
      )

      const context = createMockContext(db, request)
      const response = await onRequestGet(context)

      expect(response.status).toBe(401)

      const data = await response.json()
      expect(data.error).toBe('Invalid session token')
    })

    it('returns 401 for expired session token', async () => {
      const { user } = await createTestUser(db)
      const expiredToken = createExpiredSession(db, user.id)

      const request = createAuthenticatedRequest(
        'https://example.com/api/auth/me',
        expiredToken
      )

      const context = createMockContext(db, request)
      const response = await onRequestGet(context)

      expect(response.status).toBe(401)

      const data = await response.json()
      expect(data.error).toBe('Session expired')
    })

    it('deletes expired session token from database', async () => {
      const { user } = await createTestUser(db)
      const expiredToken = createExpiredSession(db, user.id)

      // Verify session exists
      expect(db._getSessions().has(expiredToken)).toBe(true)

      const request = createAuthenticatedRequest(
        'https://example.com/api/auth/me',
        expiredToken
      )

      const context = createMockContext(db, request)
      await onRequestGet(context)

      // Verify session was deleted
      expect(db._getSessions().has(expiredToken)).toBe(false)
    })

    it('returns 401 for empty Bearer token', async () => {
      const request = createMockRequest('https://example.com/api/auth/me', {
        headers: {
          Authorization: 'Bearer ',
        },
      })

      const context = createMockContext(db, request)
      const response = await onRequestGet(context)

      expect(response.status).toBe(401)
    })
  })

  describe('user not found', () => {
    it('returns 404 if user was deleted but session exists', async () => {
      // Create session but don't have the user in database
      const fakeUserId = crypto.randomUUID()
      const sessionToken = createTestSession(db, fakeUserId)

      const request = createAuthenticatedRequest(
        'https://example.com/api/auth/me',
        sessionToken
      )

      const context = createMockContext(db, request)
      const response = await onRequestGet(context)

      expect(response.status).toBe(404)

      const data = await response.json()
      expect(data.error).toBe('User not found')
    })
  })

  describe('response format', () => {
    it('includes all expected user fields', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)

      const request = createAuthenticatedRequest(
        'https://example.com/api/auth/me',
        sessionToken
      )

      const context = createMockContext(db, request)
      const response = await onRequestGet(context)

      const data = await response.json()
      const userData = data.user

      // Verify all expected fields are present
      expect(userData).toHaveProperty('id')
      expect(userData).toHaveProperty('email')
      expect(userData).toHaveProperty('email_verified')
      expect(userData).toHaveProperty('rating')
      expect(userData).toHaveProperty('gamesPlayed')
      expect(userData).toHaveProperty('wins')
      expect(userData).toHaveProperty('losses')
      expect(userData).toHaveProperty('draws')
      expect(userData).toHaveProperty('createdAt')
      expect(userData).toHaveProperty('lastLogin')
      expect(userData).toHaveProperty('updatedAt')

      // Verify sensitive fields are NOT present
      expect(userData).not.toHaveProperty('password_hash')
      expect(userData).not.toHaveProperty('encrypted_dek')
      expect(userData).not.toHaveProperty('oauth_id')
    })
  })
})
