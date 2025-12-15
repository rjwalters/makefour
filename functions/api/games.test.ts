import { describe, it, expect, beforeEach } from 'vitest'
import { onRequestGet, onRequestPost } from './games'
import {
  MockD1Database,
  createMockContext,
  createTestUser,
  createTestSession,
  createTestGame,
  createAuthenticatedRequest,
  createMockRequest,
} from '../lib/test-utils'

describe('GET /api/games', () => {
  let db: MockD1Database

  beforeEach(() => {
    db = new MockD1Database()
  })

  describe('returns users games', () => {
    it('returns empty list for user with no games', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)

      const request = createAuthenticatedRequest(
        'https://example.com/api/games',
        sessionToken
      )

      const context = createMockContext(db, request)
      const response = await onRequestGet(context)

      expect(response.status).toBe(200)

      const data = await response.json()
      expect(data.games).toEqual([])
      expect(data.pagination.total).toBe(0)
    })

    it('returns all games for user', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)

      // Create some games
      createTestGame(db, user.id, { outcome: 'win' })
      createTestGame(db, user.id, { outcome: 'loss' })
      createTestGame(db, user.id, { outcome: 'draw' })

      const request = createAuthenticatedRequest(
        'https://example.com/api/games',
        sessionToken
      )

      const context = createMockContext(db, request)
      const response = await onRequestGet(context)

      expect(response.status).toBe(200)

      const data = await response.json()
      expect(data.games.length).toBe(3)
      expect(data.pagination.total).toBe(3)
    })

    it('does not return other users games', async () => {
      const { user: user1 } = await createTestUser(db)
      const { user: user2 } = await createTestUser(db)
      const sessionToken = createTestSession(db, user1.id)

      // Create games for both users
      createTestGame(db, user1.id)
      createTestGame(db, user2.id)
      createTestGame(db, user2.id)

      const request = createAuthenticatedRequest(
        'https://example.com/api/games',
        sessionToken
      )

      const context = createMockContext(db, request)
      const response = await onRequestGet(context)

      const data = await response.json()
      expect(data.games.length).toBe(1) // Only user1's game
    })

    it('returns games ordered by created_at descending', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)

      // Create games with different timestamps
      const now = Date.now()
      createTestGame(db, user.id, { created_at: now - 2000, outcome: 'win' })
      createTestGame(db, user.id, { created_at: now - 1000, outcome: 'loss' })
      createTestGame(db, user.id, { created_at: now, outcome: 'draw' })

      const request = createAuthenticatedRequest(
        'https://example.com/api/games',
        sessionToken
      )

      const context = createMockContext(db, request)
      const response = await onRequestGet(context)

      const data = await response.json()
      expect(data.games[0].outcome).toBe('draw') // Most recent
      expect(data.games[1].outcome).toBe('loss')
      expect(data.games[2].outcome).toBe('win') // Oldest
    })
  })

  describe('pagination works correctly', () => {
    it('respects limit parameter', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)

      // Create 5 games
      for (let i = 0; i < 5; i++) {
        createTestGame(db, user.id)
      }

      const request = createAuthenticatedRequest(
        'https://example.com/api/games?limit=2',
        sessionToken
      )

      const context = createMockContext(db, request)
      const response = await onRequestGet(context)

      const data = await response.json()
      expect(data.games.length).toBe(2)
      expect(data.pagination.total).toBe(5)
      expect(data.pagination.limit).toBe(2)
      expect(data.pagination.hasMore).toBe(true)
    })

    it('respects offset parameter', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)

      // Create 5 games with different timestamps
      const now = Date.now()
      for (let i = 0; i < 5; i++) {
        createTestGame(db, user.id, { created_at: now - i * 1000 })
      }

      const request = createAuthenticatedRequest(
        'https://example.com/api/games?offset=2&limit=2',
        sessionToken
      )

      const context = createMockContext(db, request)
      const response = await onRequestGet(context)

      const data = await response.json()
      expect(data.games.length).toBe(2)
      expect(data.pagination.offset).toBe(2)
    })

    it('caps limit at 100', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)

      const request = createAuthenticatedRequest(
        'https://example.com/api/games?limit=200',
        sessionToken
      )

      const context = createMockContext(db, request)
      const response = await onRequestGet(context)

      const data = await response.json()
      expect(data.pagination.limit).toBe(100)
    })

    it('uses default limit of 20', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)

      const request = createAuthenticatedRequest(
        'https://example.com/api/games',
        sessionToken
      )

      const context = createMockContext(db, request)
      const response = await onRequestGet(context)

      const data = await response.json()
      expect(data.pagination.limit).toBe(20)
    })

    it('returns correct hasMore flag', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)

      // Create 3 games
      for (let i = 0; i < 3; i++) {
        createTestGame(db, user.id)
      }

      // With limit=2, hasMore should be true
      const request1 = createAuthenticatedRequest(
        'https://example.com/api/games?limit=2',
        sessionToken
      )
      const context1 = createMockContext(db, request1)
      const response1 = await onRequestGet(context1)
      const data1 = await response1.json()
      expect(data1.pagination.hasMore).toBe(true)

      // With limit=5, hasMore should be false
      const request2 = createAuthenticatedRequest(
        'https://example.com/api/games?limit=5',
        sessionToken
      )
      const context2 = createMockContext(db, request2)
      const response2 = await onRequestGet(context2)
      const data2 = await response2.json()
      expect(data2.pagination.hasMore).toBe(false)
    })
  })

  describe('requires authentication', () => {
    it('returns 401 without session token', async () => {
      const request = createMockRequest('https://example.com/api/games')

      const context = createMockContext(db, request)
      const response = await onRequestGet(context)

      expect(response.status).toBe(401)
    })

    it('returns 401 with invalid session token', async () => {
      const request = createAuthenticatedRequest(
        'https://example.com/api/games',
        'invalid-token'
      )

      const context = createMockContext(db, request)
      const response = await onRequestGet(context)

      expect(response.status).toBe(401)
    })
  })
})

describe('POST /api/games', () => {
  let db: MockD1Database

  beforeEach(() => {
    db = new MockD1Database()
  })

  describe('saves game correctly', () => {
    it('creates a new game record', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)

      const request = createAuthenticatedRequest(
        'https://example.com/api/games',
        sessionToken,
        {
          method: 'POST',
          body: {
            outcome: 'win',
            moves: [3, 2, 3, 2, 3, 2, 3],
            opponentType: 'ai',
            aiDifficulty: 'intermediate',
            playerNumber: 1,
          },
        }
      )

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      expect(response.status).toBe(201)

      const data = await response.json()
      expect(data.id).toBeDefined()
      expect(data.outcome).toBe('win')
      expect(data.moves).toEqual([3, 2, 3, 2, 3, 2, 3])
      expect(data.moveCount).toBe(7)
    })

    it('stores game in database', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)

      const request = createAuthenticatedRequest(
        'https://example.com/api/games',
        sessionToken,
        {
          method: 'POST',
          body: {
            outcome: 'loss',
            moves: [0, 1, 2],
            opponentType: 'ai',
            aiDifficulty: 'beginner',
          },
        }
      )

      const context = createMockContext(db, request)
      await onRequestPost(context)

      const games = db._getGames()
      expect(games.size).toBe(1)

      const game = Array.from(games.values())[0]
      expect(game.user_id).toBe(user.id)
      expect(game.outcome).toBe('loss')
    })

    it('updates user rating for AI games', async () => {
      const { user } = await createTestUser(db, { rating: 1200 })
      const sessionToken = createTestSession(db, user.id)

      const request = createAuthenticatedRequest(
        'https://example.com/api/games',
        sessionToken,
        {
          method: 'POST',
          body: {
            outcome: 'win',
            moves: [3, 2, 3, 2, 3, 2, 3],
            opponentType: 'ai',
            aiDifficulty: 'intermediate',
          },
        }
      )

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      const data = await response.json()
      expect(data.ratingChange).toBeGreaterThan(0)
      expect(data.newRating).toBeGreaterThan(1200)
    })

    it('updates user stats', async () => {
      const { user } = await createTestUser(db, {
        games_played: 5,
        wins: 3,
        losses: 2,
        draws: 0,
      })
      const sessionToken = createTestSession(db, user.id)

      const request = createAuthenticatedRequest(
        'https://example.com/api/games',
        sessionToken,
        {
          method: 'POST',
          body: {
            outcome: 'win',
            moves: [3, 2, 3, 2, 3, 2, 3],
            opponentType: 'ai',
            aiDifficulty: 'beginner',
          },
        }
      )

      const context = createMockContext(db, request)
      await onRequestPost(context)

      const updatedUser = db._getUsers().get(user.id)
      expect(updatedUser?.games_played).toBe(6)
      expect(updatedUser?.wins).toBe(4)
    })
  })

  describe('validates move format', () => {
    it('rejects moves with invalid column numbers', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)

      const request = createAuthenticatedRequest(
        'https://example.com/api/games',
        sessionToken,
        {
          method: 'POST',
          body: {
            outcome: 'win',
            moves: [3, 7, 3], // 7 is invalid (0-6 valid)
            opponentType: 'ai',
            aiDifficulty: 'beginner',
          },
        }
      )

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      expect(response.status).toBe(400)
    })

    it('rejects negative column numbers', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)

      const request = createAuthenticatedRequest(
        'https://example.com/api/games',
        sessionToken,
        {
          method: 'POST',
          body: {
            outcome: 'win',
            moves: [-1, 3, 3],
            opponentType: 'ai',
            aiDifficulty: 'beginner',
          },
        }
      )

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      expect(response.status).toBe(400)
    })

    it('accepts valid move format', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)

      const request = createAuthenticatedRequest(
        'https://example.com/api/games',
        sessionToken,
        {
          method: 'POST',
          body: {
            outcome: 'draw',
            moves: [0, 1, 2, 3, 4, 5, 6], // All valid columns
            opponentType: 'ai',
            aiDifficulty: 'expert',
          },
        }
      )

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      expect(response.status).toBe(201)
    })
  })

  describe('rejects invalid outcomes', () => {
    it('rejects invalid outcome value', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)

      const request = createAuthenticatedRequest(
        'https://example.com/api/games',
        sessionToken,
        {
          method: 'POST',
          body: {
            outcome: 'invalid',
            moves: [3, 2, 3],
            opponentType: 'ai',
            aiDifficulty: 'beginner',
          },
        }
      )

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      expect(response.status).toBe(400)
    })

    it('accepts all valid outcomes', async () => {
      const { user } = await createTestUser(db)
      const sessionToken = createTestSession(db, user.id)

      for (const outcome of ['win', 'loss', 'draw']) {
        const request = createAuthenticatedRequest(
          'https://example.com/api/games',
          sessionToken,
          {
            method: 'POST',
            body: {
              outcome,
              moves: [3],
              opponentType: 'ai',
              aiDifficulty: 'beginner',
            },
          }
        )

        const context = createMockContext(db, request)
        const response = await onRequestPost(context)

        expect(response.status).toBe(201)
      }
    })
  })

  describe('requires authentication', () => {
    it('returns 401 without session token', async () => {
      const request = createMockRequest('https://example.com/api/games', {
        method: 'POST',
        body: {
          outcome: 'win',
          moves: [3],
          opponentType: 'ai',
          aiDifficulty: 'beginner',
        },
      })

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      expect(response.status).toBe(401)
    })
  })

  describe('AI difficulty ratings', () => {
    it('calculates correct rating change for different difficulties', async () => {
      const difficulties = ['beginner', 'intermediate', 'expert', 'perfect']

      for (const difficulty of difficulties) {
        const db = new MockD1Database()
        const { user } = await createTestUser(db, { rating: 1200 })
        const sessionToken = createTestSession(db, user.id)

        const request = createAuthenticatedRequest(
          'https://example.com/api/games',
          sessionToken,
          {
            method: 'POST',
            body: {
              outcome: 'win',
              moves: [3, 2, 3, 2, 3, 2, 3],
              opponentType: 'ai',
              aiDifficulty: difficulty,
            },
          }
        )

        const context = createMockContext(db, request)
        const response = await onRequestPost(context)

        expect(response.status).toBe(201)

        const data = await response.json()
        expect(data.ratingChange).toBeDefined()
        expect(typeof data.ratingChange).toBe('number')
      }
    })
  })

  describe('human games', () => {
    it('does not change rating for human games', async () => {
      const { user } = await createTestUser(db, { rating: 1200 })
      const sessionToken = createTestSession(db, user.id)

      const request = createAuthenticatedRequest(
        'https://example.com/api/games',
        sessionToken,
        {
          method: 'POST',
          body: {
            outcome: 'win',
            moves: [3, 2, 3, 2, 3, 2, 3],
            opponentType: 'human',
          },
        }
      )

      const context = createMockContext(db, request)
      const response = await onRequestPost(context)

      const data = await response.json()
      expect(data.ratingChange).toBe(0)
      expect(data.newRating).toBe(1200)
    })
  })
})
