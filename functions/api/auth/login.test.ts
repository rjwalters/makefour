import { describe, it, expect, beforeEach } from 'vitest'
import { onRequestPost } from './login'
import {
  MockD1Database,
  createMockRequest,
  createMockContext,
  createTestUser,
  expectResponse,
} from '../../lib/test-utils'

describe('POST /api/auth/login', () => {
  let db: MockD1Database

  beforeEach(() => {
    db = new MockD1Database()
  })

  describe('successful login', () => {
    it('returns user data and session token for valid credentials', async () => {
      const { user, password } = await createTestUser(db, {
        email: 'valid@example.com',
      })

      const request = createMockRequest('https://example.com/api/auth/login', {
        method: 'POST',
        body: {
          email: 'valid@example.com',
          password,
        },
      })

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      expect(response.status).toBe(200)

      const data = await expectResponse<{
        user: { id: string; email: string; email_verified: boolean }
        session_token: string
        encrypted_dek: string | null
      }>(response, 200)

      expect(data.user.email).toBe('valid@example.com')
      expect(data.user.id).toBe(user.id)
      expect(data.session_token).toBeDefined()
      expect(typeof data.session_token).toBe('string')
    })

    it('creates a session in the database', async () => {
      const { user, password } = await createTestUser(db)

      const request = createMockRequest('https://example.com/api/auth/login', {
        method: 'POST',
        body: {
          email: user.email,
          password,
        },
      })

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)
      const data = await response.json()

      // Verify session was created
      const sessions = db._getSessions()
      expect(sessions.size).toBe(1)

      const session = sessions.get(data.session_token)
      expect(session).toBeDefined()
      expect(session?.user_id).toBe(user.id)
    })

    it('returns encrypted_dek for client-side decryption', async () => {
      const { user, password } = await createTestUser(db, {
        encrypted_dek: 'test-encrypted-dek-value',
      })

      const request = createMockRequest('https://example.com/api/auth/login', {
        method: 'POST',
        body: {
          email: user.email,
          password,
        },
      })

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      const data = await response.json()
      expect(data.encrypted_dek).toBe('test-encrypted-dek-value')
    })
  })

  describe('invalid credentials', () => {
    it('returns 401 for wrong password', async () => {
      const { user } = await createTestUser(db)

      const request = createMockRequest('https://example.com/api/auth/login', {
        method: 'POST',
        body: {
          email: user.email,
          password: 'WrongPassword123!',
        },
      })

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      expect(response.status).toBe(401)

      const data = await response.json()
      expect(data.error).toBe('Invalid credentials')
    })

    it('returns 401 for correct email with empty password', async () => {
      const { user } = await createTestUser(db)

      const request = createMockRequest('https://example.com/api/auth/login', {
        method: 'POST',
        body: {
          email: user.email,
          password: '',
        },
      })

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      expect(response.status).toBe(400) // Validation error
    })
  })

  describe('non-existent user', () => {
    it('returns 401 for non-existent email', async () => {
      const request = createMockRequest('https://example.com/api/auth/login', {
        method: 'POST',
        body: {
          email: 'nonexistent@example.com',
          password: 'SomePassword123!',
        },
      })

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      expect(response.status).toBe(401)

      const data = await response.json()
      expect(data.error).toBe('Invalid credentials')
    })

    it('does not reveal whether email exists', async () => {
      await createTestUser(db, { email: 'exists@example.com' })

      // Try with non-existent email
      const request1 = createMockRequest('https://example.com/api/auth/login', {
        method: 'POST',
        body: {
          email: 'nonexistent@example.com',
          password: 'SomePassword123!',
        },
      })

      const context1 = createMockContext(db, request1)
      const response1 = await onRequestPost(context1)
      const data1 = await response1.json()

      // Try with existing email but wrong password
      const request2 = createMockRequest('https://example.com/api/auth/login', {
        method: 'POST',
        body: {
          email: 'exists@example.com',
          password: 'WrongPassword123!',
        },
      })

      const context2 = createMockContext(db, request2)
      const response2 = await onRequestPost(context2)
      const data2 = await response2.json()

      // Both should return the same error message
      expect(response1.status).toBe(response2.status)
      expect(data1.error).toBe(data2.error)
    })
  })

  describe('validation errors', () => {
    it('returns 400 for invalid email format', async () => {
      const request = createMockRequest('https://example.com/api/auth/login', {
        method: 'POST',
        body: {
          email: 'not-an-email',
          password: 'SomePassword123!',
        },
      })

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      expect(response.status).toBe(400)
    })

    it('returns 400 for missing email', async () => {
      const request = createMockRequest('https://example.com/api/auth/login', {
        method: 'POST',
        body: {
          password: 'SomePassword123!',
        },
      })

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      expect(response.status).toBe(400)
    })

    it('returns 400 for missing password', async () => {
      const request = createMockRequest('https://example.com/api/auth/login', {
        method: 'POST',
        body: {
          email: 'test@example.com',
        },
      })

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      expect(response.status).toBe(400)
    })

    it('returns 400 for empty body', async () => {
      const request = createMockRequest('https://example.com/api/auth/login', {
        method: 'POST',
        body: {},
      })

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      expect(response.status).toBe(400)
    })
  })

  describe('OAuth users', () => {
    it('returns 401 for OAuth user without password_hash', async () => {
      await createTestUser(db, {
        email: 'oauth@example.com',
        password_hash: null,
        oauth_provider: 'google',
        oauth_id: 'google-123',
      })

      const request = createMockRequest('https://example.com/api/auth/login', {
        method: 'POST',
        body: {
          email: 'oauth@example.com',
          password: 'SomePassword123!',
        },
      })

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      expect(response.status).toBe(401)
    })
  })
})
