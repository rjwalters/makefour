import { validateSession, errorResponse, jsonResponse } from '../../../lib/auth'

interface Env {
  DB: D1Database
}

interface UserRow {
  id: string
  email: string
  email_verified: number
  oauth_provider: string | null
  rating: number
  games_played: number
  wins: number
  losses: number
  draws: number
  created_at: number
  last_login: number
  updated_at: number
}

interface GameRow {
  outcome: string
  player_number: number
  opponent_type: string
  move_count: number
  rating_change: number
  created_at: number
}

interface RatingHistoryRow {
  rating_after: number
  created_at: number
}

export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    // Validate session
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    const userId = session.userId

    // Get user data including oauth_provider
    const user = await DB.prepare(
      `SELECT id, email, email_verified, oauth_provider, rating, games_played, wins, losses, draws,
              created_at, last_login, updated_at
       FROM users WHERE id = ?`
    )
      .bind(userId)
      .first<UserRow>()

    if (!user) {
      return errorResponse('User not found', 404)
    }

    // Get all games for stats calculation
    const gamesResult = await DB.prepare(
      `SELECT outcome, player_number, opponent_type, move_count, rating_change, created_at
       FROM games WHERE user_id = ? ORDER BY created_at ASC`
    )
      .bind(userId)
      .all<GameRow>()

    const games = gamesResult.results || []

    // Get rating history
    const ratingHistoryResult = await DB.prepare(
      `SELECT rating_after, created_at
       FROM rating_history WHERE user_id = ? ORDER BY created_at ASC`
    )
      .bind(userId)
      .all<RatingHistoryRow>()

    const ratingHistory = ratingHistoryResult.results || []

    // Calculate advanced stats
    let peakRating = user.rating
    let lowestRating = user.rating
    let currentStreak = 0
    let longestWinStreak = 0
    let longestLossStreak = 0
    let tempWinStreak = 0
    let tempLossStreak = 0
    let gamesAsPlayer1 = 0
    let gamesAsPlayer2 = 0
    let aiGames = 0
    let humanGames = 0
    let totalMoveCount = 0
    let recentRatingChange = 0

    // Process rating history for peak/lowest
    for (const entry of ratingHistory) {
      if (entry.rating_after > peakRating) peakRating = entry.rating_after
      if (entry.rating_after < lowestRating) lowestRating = entry.rating_after
    }

    // If no history, use current rating
    if (ratingHistory.length === 0) {
      peakRating = 1200
      lowestRating = 1200
    }

    // Process games for other stats
    for (const game of games) {
      // Player number stats
      if (game.player_number === 1) gamesAsPlayer1++
      else gamesAsPlayer2++

      // Opponent type stats
      if (game.opponent_type === 'ai') aiGames++
      else humanGames++

      // Move count
      totalMoveCount += game.move_count

      // Streak calculation
      if (game.outcome === 'win') {
        tempWinStreak++
        tempLossStreak = 0
        if (tempWinStreak > longestWinStreak) longestWinStreak = tempWinStreak
      } else if (game.outcome === 'loss') {
        tempLossStreak++
        tempWinStreak = 0
        if (tempLossStreak > longestLossStreak) longestLossStreak = tempLossStreak
      } else {
        // Draw breaks streaks
        tempWinStreak = 0
        tempLossStreak = 0
      }
    }

    // Current streak (from most recent games)
    if (games.length > 0) {
      const recentGames = [...games].reverse()
      let streakType: 'win' | 'loss' | null = null
      currentStreak = 0

      for (const game of recentGames) {
        if (game.outcome === 'win') {
          if (streakType === null) streakType = 'win'
          if (streakType === 'win') currentStreak++
          else break
        } else if (game.outcome === 'loss') {
          if (streakType === null) streakType = 'loss'
          if (streakType === 'loss') currentStreak++
          else break
        } else {
          break // Draw breaks streak
        }
      }

      // Negative for loss streak
      if (streakType === 'loss') currentStreak = -currentStreak
    }

    // Recent rating change (last 10 games)
    const recentGames = games.slice(-10)
    recentRatingChange = recentGames.reduce((sum, game) => sum + (game.rating_change || 0), 0)

    // Determine rating trend
    let ratingTrend: 'improving' | 'declining' | 'stable' = 'stable'
    if (recentRatingChange > 20) ratingTrend = 'improving'
    else if (recentRatingChange < -20) ratingTrend = 'declining'

    // Calculate average move count
    const avgMoveCount = games.length > 0 ? totalMoveCount / games.length : 0

    // Format rating history for chart
    const formattedRatingHistory = ratingHistory.map(entry => ({
      rating: entry.rating_after,
      createdAt: entry.created_at,
    }))

    return jsonResponse({
      user: {
        id: user.id,
        email: user.email,
        email_verified: user.email_verified === 1,
        oauth_provider: user.oauth_provider,
        rating: user.rating,
        gamesPlayed: user.games_played,
        wins: user.wins,
        losses: user.losses,
        draws: user.draws,
        createdAt: user.created_at,
        lastLogin: user.last_login,
        updatedAt: user.updated_at,
      },
      stats: {
        peakRating,
        lowestRating,
        currentStreak,
        longestWinStreak,
        longestLossStreak,
        avgMoveCount,
        gamesAsPlayer1,
        gamesAsPlayer2,
        aiGames,
        humanGames,
        ratingTrend,
        recentRatingChange,
      },
      ratingHistory: formattedRatingHistory,
    })
  } catch (error) {
    console.error('Stats endpoint error:', error)
    return errorResponse('Internal server error', 500)
  }
}
