import { describe, it, expect, beforeEach } from 'vitest'
import { onRequestGet } from './[id]'
import {
  MockD1Database,
  createMockContext,
  createTestUser,
  createTestSession,
  createTestGame,
  createAuthenticatedRequest,
  createMockRequest,
} from '../../lib/test-utils'

describe('GET /api/games/:id', () => {
  let db: MockD1Database

  beforeEach(() => {
    db = new MockD1Database()
  })

  describe('returns correct game', () => {
    it('returns game data for valid ID', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)
      const game = createTestGame(db, user.id, {
        outcome: 'win',
        ai_difficulty: 'expert',
        moves: JSON.stringify([3, 2, 3, 2, 3, 2, 3]),
        move_count: 7,
        player_number: 1,
      })

      const request = createAuthenticatedRequest(
        `https://example.com/api/games/${game.id}`,
        sessionToken
      )

      const context = createMockContext(db, request, { id: game.id })
      const response = await onRequestGet(context)

      expect(response.status).toBe(200)

      const data = await response.json()
      expect(data.id).toBe(game.id)
      expect(data.outcome).toBe('win')
      expect(data.aiDifficulty).toBe('expert')
      expect(data.moves).toEqual([3, 2, 3, 2, 3, 2, 3])
      expect(data.moveCount).toBe(7)
      expect(data.playerNumber).toBe(1)
    })

    it('parses moves JSON correctly', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)
      const game = createTestGame(db, user.id, {
        moves: JSON.stringify([0, 1, 2, 3, 4, 5, 6]),
      })

      const request = createAuthenticatedRequest(
        `https://example.com/api/games/${game.id}`,
        sessionToken
      )

      const context = createMockContext(db, request, { id: game.id })
      const response = await onRequestGet(context)

      const data = await response.json()
      expect(Array.isArray(data.moves)).toBe(true)
      expect(data.moves).toEqual([0, 1, 2, 3, 4, 5, 6])
    })

    it('returns all expected fields', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)
      const game = createTestGame(db, user.id)

      const request = createAuthenticatedRequest(
        `https://example.com/api/games/${game.id}`,
        sessionToken
      )

      const context = createMockContext(db, request, { id: game.id })
      const response = await onRequestGet(context)

      const data = await response.json()
      expect(data).toHaveProperty('id')
      expect(data).toHaveProperty('outcome')
      expect(data).toHaveProperty('moves')
      expect(data).toHaveProperty('moveCount')
      expect(data).toHaveProperty('opponentType')
      expect(data).toHaveProperty('aiDifficulty')
      expect(data).toHaveProperty('playerNumber')
      expect(data).toHaveProperty('createdAt')
    })
  })

  describe('404 for non-existent game', () => {
    it('returns 404 for non-existent game ID', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)
      const nonExistentId = crypto.randomUUID()

      const request = createAuthenticatedRequest(
        `https://example.com/api/games/${nonExistentId}`,
        sessionToken
      )

      const context = createMockContext(db, request, { id: nonExistentId })
      const response = await onRequestGet(context)

      expect(response.status).toBe(404)

      const data = await response.json()
      expect(data.error).toBe('Game not found')
    })

    it('returns 404 for invalid UUID format', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)

      const request = createAuthenticatedRequest(
        'https://example.com/api/games/invalid-id',
        sessionToken
      )

      const context = createMockContext(db, request, { id: 'invalid-id' })
      const response = await onRequestGet(context)

      expect(response.status).toBe(404)
    })
  })

  describe('only owner can access', () => {
    it('returns 404 when accessing another users game', async () => {
      const { user: owner } = await createTestUser(db)
      const { user: otherUser } = await createTestUser(db)
      const sessionToken = createTestSession(db, otherUser.id)

      // Create game owned by another user
      const game = createTestGame(db, owner.id)

      const request = createAuthenticatedRequest(
        `https://example.com/api/games/${game.id}`,
        sessionToken
      )

      const context = createMockContext(db, request, { id: game.id })
      const response = await onRequestGet(context)

      // Should return 404 to not leak existence of game
      expect(response.status).toBe(404)
    })

    it('owner can access their own game', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)
      const game = createTestGame(db, user.id)

      const request = createAuthenticatedRequest(
        `https://example.com/api/games/${game.id}`,
        sessionToken
      )

      const context = createMockContext(db, request, { id: game.id })
      const response = await onRequestGet(context)

      expect(response.status).toBe(200)
    })
  })

  describe('authentication required', () => {
    it('returns 401 without session token', async () => {
      const { user } = await createTestUser(db)
      const game = createTestGame(db, user.id)

      const request = createMockRequest(
        `https://example.com/api/games/${game.id}`
      )

      const context = createMockContext(db, request, { id: game.id })
      const response = await onRequestGet(context)

      expect(response.status).toBe(401)
    })

    it('returns 401 with invalid session token', async () => {
      const { user } = await createTestUser(db)
      const game = createTestGame(db, user.id)

      const request = createAuthenticatedRequest(
        `https://example.com/api/games/${game.id}`,
        'invalid-token'
      )

      const context = createMockContext(db, request, { id: game.id })
      const response = await onRequestGet(context)

      expect(response.status).toBe(401)
    })
  })

  describe('game data integrity', () => {
    it('returns correct data for AI game', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)
      const game = createTestGame(db, user.id, {
        opponent_type: 'ai',
        ai_difficulty: 'perfect',
      })

      const request = createAuthenticatedRequest(
        `https://example.com/api/games/${game.id}`,
        sessionToken
      )

      const context = createMockContext(db, request, { id: game.id })
      const response = await onRequestGet(context)

      const data = await response.json()
      expect(data.opponentType).toBe('ai')
      expect(data.aiDifficulty).toBe('perfect')
    })

    it('returns correct data for human game', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)
      const game = createTestGame(db, user.id, {
        opponent_type: 'human',
        ai_difficulty: null,
      })

      const request = createAuthenticatedRequest(
        `https://example.com/api/games/${game.id}`,
        sessionToken
      )

      const context = createMockContext(db, request, { id: game.id })
      const response = await onRequestGet(context)

      const data = await response.json()
      expect(data.opponentType).toBe('human')
      expect(data.aiDifficulty).toBeNull()
    })

    it('returns correct player number', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)

      // Test player 1
      const game1 = createTestGame(db, user.id, { player_number: 1 })
      const request1 = createAuthenticatedRequest(
        `https://example.com/api/games/${game1.id}`,
        sessionToken
      )
      const context1 = createMockContext(db, request1, { id: game1.id })
      const response1 = await onRequestGet(context1)
      const data1 = await response1.json()
      expect(data1.playerNumber).toBe(1)

      // Test player 2
      const game2 = createTestGame(db, user.id, { player_number: 2 })
      const request2 = createAuthenticatedRequest(
        `https://example.com/api/games/${game2.id}`,
        sessionToken
      )
      const context2 = createMockContext(db, request2, { id: game2.id })
      const response2 = await onRequestGet(context2)
      const data2 = await response2.json()
      expect(data2.playerNumber).toBe(2)
    })
  })
})
