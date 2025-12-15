/**
 * Tests for POST /api/auth/verify-email
 */

import { describe, it, expect, beforeEach } from 'vitest'
import { onRequestPost } from './verify-email'
import {
  MockD1Database,
  createMockContext,
  createMockRequest,
  createTestUser,
  createUnverifiedTestUser,
  createTestVerificationToken,
  createExpiredVerificationToken,
  expectResponse,
} from '../../lib/test-utils'

describe('POST /api/auth/verify-email', () => {
  let db: MockD1Database

  beforeEach(() => {
    db = new MockD1Database()
  })

  it('should verify email with valid token', async () => {
    // Create unverified user
    const { user } = await createUnverifiedTestUser(db)
    expect(user.email_verified).toBe(0)

    // Create verification token
    const token = createTestVerificationToken(db, user.id)

    // Verify
    const request = createMockRequest('http://localhost/api/auth/verify-email', {
      method: 'POST',
      body: { token: token.id },
    })
    const context = createMockContext(db, request)
    const response = await onRequestPost(context)

    const data = await expectResponse<{ success: boolean; message: string; user: { email_verified: boolean } }>(response, 200)

    expect(data.success).toBe(true)
    expect(data.message).toBe('Email verified successfully')
    expect(data.user.email_verified).toBe(true)

    // Verify user is now verified in database
    const updatedUser = db._getUsers().get(user.id)
    expect(updatedUser?.email_verified).toBe(1)

    // Verify token is marked as used
    const updatedToken = db._getVerificationTokens().get(token.id)
    expect(updatedToken?.used).toBe(1)
  })

  it('should reject invalid token', async () => {
    const request = createMockRequest('http://localhost/api/auth/verify-email', {
      method: 'POST',
      body: { token: crypto.randomUUID() },
    })
    const context = createMockContext(db, request)
    const response = await onRequestPost(context)

    const data = await expectResponse<{ error: string }>(response, 400)
    expect(data.error).toBe('Invalid verification token')
  })

  it('should reject expired token', async () => {
    const { user } = await createUnverifiedTestUser(db)
    const token = createExpiredVerificationToken(db, user.id)

    const request = createMockRequest('http://localhost/api/auth/verify-email', {
      method: 'POST',
      body: { token: token.id },
    })
    const context = createMockContext(db, request)
    const response = await onRequestPost(context)

    const data = await expectResponse<{ error: string }>(response, 400)
    expect(data.error).toBe('Token has expired')

    // User should still be unverified
    const updatedUser = db._getUsers().get(user.id)
    expect(updatedUser?.email_verified).toBe(0)
  })

  it('should reject already used token', async () => {
    const { user } = await createUnverifiedTestUser(db)
    const token = createTestVerificationToken(db, user.id, { used: 1 })

    const request = createMockRequest('http://localhost/api/auth/verify-email', {
      method: 'POST',
      body: { token: token.id },
    })
    const context = createMockContext(db, request)
    const response = await onRequestPost(context)

    const data = await expectResponse<{ error: string }>(response, 400)
    expect(data.error).toBe('Token has already been used')
  })

  it('should reject invalid token format', async () => {
    const request = createMockRequest('http://localhost/api/auth/verify-email', {
      method: 'POST',
      body: { token: 'not-a-uuid' },
    })
    const context = createMockContext(db, request)
    const response = await onRequestPost(context)

    const data = await expectResponse<{ error: string }>(response, 400)
    expect(data.error).toBe('Invalid verification token format')
  })

  it('should reject missing token', async () => {
    const request = createMockRequest('http://localhost/api/auth/verify-email', {
      method: 'POST',
      body: {},
    })
    const context = createMockContext(db, request)
    const response = await onRequestPost(context)

    expect(response.status).toBe(400)
  })
})
