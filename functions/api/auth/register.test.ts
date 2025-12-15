import { describe, it, expect, beforeEach } from 'vitest'
import { onRequestPost } from './register'
import {
  MockD1Database,
  createMockRequest,
  createMockContext,
  createTestUser,
  expectResponse,
} from '../../lib/test-utils'

describe('POST /api/auth/register', () => {
  let db: MockD1Database

  beforeEach(() => {
    db = new MockD1Database()
  })

  describe('successful registration', () => {
    it('creates a new user with valid email and password', async () => {
      const request = createMockRequest('https://example.com/api/auth/register', {
        method: 'POST',
        body: {
          email: 'newuser@example.com',
          password: 'SecurePassword123!',
        },
      })

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      expect(response.status).toBe(201)

      const data = await expectResponse<{
        user: { id: string; email: string; email_verified: number }
        message: string
      }>(response, 201)

      expect(data.message).toBe('Registration successful')
      expect(data.user.email).toBe('newuser@example.com')
      expect(data.user.id).toBeDefined()
    })

    it('stores the user in the database', async () => {
      const request = createMockRequest('https://example.com/api/auth/register', {
        method: 'POST',
        body: {
          email: 'stored@example.com',
          password: 'SecurePassword123!',
        },
      })

      const context = createMockContext(db, request)
      await onRequestPost(context)

      // Verify user was stored
      const users = db._getUsers()
      expect(users.size).toBe(1)

      const user = Array.from(users.values())[0]
      expect(user.email).toBe('stored@example.com')
      expect(user.password_hash).toBeDefined()
      expect(user.encrypted_dek).toBeDefined()
    })
  })

  describe('duplicate email handling', () => {
    it('returns 409 for duplicate email', async () => {
      // Create existing user
      await createTestUser(db, { email: 'existing@example.com' })

      const request = createMockRequest('https://example.com/api/auth/register', {
        method: 'POST',
        body: {
          email: 'existing@example.com',
          password: 'SecurePassword123!',
        },
      })

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      expect(response.status).toBe(409)

      const data = await response.json()
      expect(data.error).toBe('User already exists')
    })
  })

  describe('invalid email format', () => {
    it('returns 400 for invalid email', async () => {
      const request = createMockRequest('https://example.com/api/auth/register', {
        method: 'POST',
        body: {
          email: 'not-an-email',
          password: 'SecurePassword123!',
        },
      })

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      expect(response.status).toBe(400)

      const data = await response.json()
      expect(data.error).toBe('Validation error')
    })

    it('returns 400 for empty email', async () => {
      const request = createMockRequest('https://example.com/api/auth/register', {
        method: 'POST',
        body: {
          email: '',
          password: 'SecurePassword123!',
        },
      })

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      expect(response.status).toBe(400)
    })
  })

  describe('weak password rejection', () => {
    it('returns 400 for password shorter than 8 characters', async () => {
      const request = createMockRequest('https://example.com/api/auth/register', {
        method: 'POST',
        body: {
          email: 'test@example.com',
          password: 'short',
        },
      })

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      expect(response.status).toBe(400)

      const data = await response.json()
      expect(data.error).toBe('Validation error')
      expect(data.details).toContain('8 characters')
    })

    it('returns 400 for password longer than 128 characters', async () => {
      const request = createMockRequest('https://example.com/api/auth/register', {
        method: 'POST',
        body: {
          email: 'test@example.com',
          password: 'a'.repeat(129),
        },
      })

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      expect(response.status).toBe(400)

      const data = await response.json()
      expect(data.error).toBe('Validation error')
    })

    it('returns 400 for empty password', async () => {
      const request = createMockRequest('https://example.com/api/auth/register', {
        method: 'POST',
        body: {
          email: 'test@example.com',
          password: '',
        },
      })

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      expect(response.status).toBe(400)
    })
  })

  describe('missing fields', () => {
    it('returns 400 when email is missing', async () => {
      const request = createMockRequest('https://example.com/api/auth/register', {
        method: 'POST',
        body: {
          password: 'SecurePassword123!',
        },
      })

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      expect(response.status).toBe(400)
    })

    it('returns 400 when password is missing', async () => {
      const request = createMockRequest('https://example.com/api/auth/register', {
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
      const request = createMockRequest('https://example.com/api/auth/register', {
        method: 'POST',
        body: {},
      })

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      expect(response.status).toBe(400)
    })
  })
})
