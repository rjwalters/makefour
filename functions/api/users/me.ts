import { validateSession, errorResponse, jsonResponse } from '../../lib/auth'

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

    // Check if user exists
    const user = await DB.prepare('SELECT id FROM users WHERE id = ?')
      .bind(userId)
      .first()

    if (!user) {
      return errorResponse('User not found', 404)
    }

    // Handle any active games (forfeit them)
    const activeGamesResult = await DB.prepare(
      `SELECT id, player1_id, player2_id FROM active_games
       WHERE (player1_id = ? OR player2_id = ?) AND status = 'active'`
    )
      .bind(userId, userId)
      .all()

    const activeGames = activeGamesResult.results || []

    for (const game of activeGames) {
      // Mark game as abandoned with the other player winning
      const winner = game.player1_id === userId ? '2' : '1'
      await DB.prepare(
        `UPDATE active_games SET status = 'abandoned', winner = ?, updated_at = ? WHERE id = ?`
      )
        .bind(winner, Date.now(), game.id)
        .run()
    }

    // Delete user - CASCADE will handle related records
    // Due to ON DELETE CASCADE, this will also delete:
    // - session_tokens
    // - email_verification_tokens
    // - password_reset_tokens
    // - games
    // - rating_history
    // - matchmaking_queue entries
    await DB.prepare('DELETE FROM users WHERE id = ?')
      .bind(userId)
      .run()

    return jsonResponse({ message: 'Account deleted successfully' })
  } catch (error) {
    console.error('Delete account error:', error)
    return errorResponse('Internal server error', 500)
  }
}
