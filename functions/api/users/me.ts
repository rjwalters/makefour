import { and, eq, or } from 'drizzle-orm'
import { createDb } from '../../../shared/db/client'
import { activeGames, users } from '../../../shared/db/schema'
import { errorResponse, jsonResponse, validateSession } from '../../lib/auth'

interface Env {
  DB: D1Database
}

/**
 * DELETE /api/users/me - Delete user account
 *
 * Permanently deletes the user's account and all associated data:
 * - User record
 * - Session tokens
 * - Games
 * - Rating history
 * - Matchmaking queue entries
 * - Active games (as forfeit)
 */
export async function onRequestDelete(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    // Validate session
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    const userId = session.userId
    const db = createDb(DB)

    // Check if user exists
    const user = await db.query.users.findFirst({
      where: eq(users.id, userId),
      columns: {
        id: true,
      },
    })

    if (!user) {
      return errorResponse('User not found', 404)
    }

    // Handle any active games (forfeit them)
    const activeGamesList = await db.query.activeGames.findMany({
      where: and(
        or(eq(activeGames.player1Id, userId), eq(activeGames.player2Id, userId)),
        eq(activeGames.status, 'active')
      ),
      columns: {
        id: true,
        player1Id: true,
        player2Id: true,
      },
    })

    for (const game of activeGamesList) {
      // Mark game as abandoned with the other player winning
      const winner = game.player1Id === userId ? '2' : '1'
      await db
        .update(activeGames)
        .set({
          status: 'abandoned',
          winner: winner,
          updatedAt: Date.now(),
        })
        .where(eq(activeGames.id, game.id))
    }

    // Delete user - CASCADE will handle related records
    // Due to ON DELETE CASCADE, this will also delete:
    // - session_tokens
    // - email_verification_tokens
    // - password_reset_tokens
    // - games
    // - rating_history
    // - matchmaking_queue entries
    await db.delete(users).where(eq(users.id, userId))

    return jsonResponse({ message: 'Account deleted successfully' })
  } catch (error) {
    console.error('Delete account error:', error)
    return errorResponse('Internal server error', 500)
  }
}
