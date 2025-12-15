import { validateSession, errorResponse, jsonResponse } from '../../../../lib/auth'

interface Env {
  DB: D1Database
}

interface GameRow {
  id: string
  outcome: string
  player_number: number
  opponent_type: string
  ai_difficulty: string | null
  move_count: number
  moves: string
  rating_change: number
  created_at: number
}

interface DailyStats {
  date: string
  games: number
  wins: number
  losses: number
  draws: number
  ratingChange: number
  avgMoveCount: number
}

interface OpeningStats {
  column: number
  games: number
  wins: number
  losses: number
  draws: number
  winRate: number
}

interface RecentGame {
  id: string
  outcome: string
  opponentType: string
  aiDifficulty: string | null
  playerNumber: number
  moveCount: number
  ratingChange: number
  createdAt: number
  firstMove: number | null
}

export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    const userId = session.userId
    const url = new URL(context.request.url)

    // Parse date range filters (timestamps in milliseconds)
    const startDateParam = url.searchParams.get('start')
    const endDateParam = url.searchParams.get('end')

    // Default to last 30 days if no range specified
    const now = Date.now()
    const thirtyDaysAgo = now - 30 * 24 * 60 * 60 * 1000
    const startDate = startDateParam ? Number.parseInt(startDateParam, 10) : thirtyDaysAgo
    const endDate = endDateParam ? Number.parseInt(endDateParam, 10) : now

    // Get games within date range
    const gamesResult = await DB.prepare(
      `SELECT id, outcome, player_number, opponent_type, ai_difficulty, move_count, moves, rating_change, created_at
       FROM games
       WHERE user_id = ? AND created_at >= ? AND created_at <= ?
       ORDER BY created_at ASC`
    )
      .bind(userId, startDate, endDate)
      .all<GameRow>()

    const games = gamesResult.results || []

    // Calculate daily aggregated stats
    const dailyMap = new Map<string, DailyStats>()
    const openingMap = new Map<number, { wins: number; losses: number; draws: number; games: number }>()

    // Stats for wins vs losses analysis
    let totalWinMoves = 0
    let winCount = 0
    let totalLossMoves = 0
    let lossCount = 0

    // Player position stats
    let player1Wins = 0
    let player1Games = 0
    let player2Wins = 0
    let player2Games = 0

    for (const game of games) {
      // Daily aggregation
      const date = new Date(game.created_at).toISOString().split('T')[0]
      const existing = dailyMap.get(date) || {
        date,
        games: 0,
        wins: 0,
        losses: 0,
        draws: 0,
        ratingChange: 0,
        avgMoveCount: 0,
      }

      existing.games++
      existing.ratingChange += game.rating_change || 0
      existing.avgMoveCount += game.move_count

      if (game.outcome === 'win') existing.wins++
      else if (game.outcome === 'loss') existing.losses++
      else existing.draws++

      dailyMap.set(date, existing)

      // Opening move stats (first move in the moves array)
      try {
        const moves = JSON.parse(game.moves) as number[]
        if (moves.length > 0) {
          const firstMove = game.player_number === 1 ? moves[0] : moves[1]
          if (firstMove !== undefined) {
            const openingStats = openingMap.get(firstMove) || { wins: 0, losses: 0, draws: 0, games: 0 }
            openingStats.games++
            if (game.outcome === 'win') openingStats.wins++
            else if (game.outcome === 'loss') openingStats.losses++
            else openingStats.draws++
            openingMap.set(firstMove, openingStats)
          }
        }
      } catch {
        // Skip malformed moves
      }

      // Win/loss move count analysis
      if (game.outcome === 'win') {
        totalWinMoves += game.move_count
        winCount++
      } else if (game.outcome === 'loss') {
        totalLossMoves += game.move_count
        lossCount++
      }

      // Player position stats
      if (game.player_number === 1) {
        player1Games++
        if (game.outcome === 'win') player1Wins++
      } else {
        player2Games++
        if (game.outcome === 'win') player2Wins++
      }
    }

    // Finalize daily averages
    const dailyStats: DailyStats[] = []
    for (const [, stats] of dailyMap) {
      stats.avgMoveCount = stats.games > 0 ? Math.round((stats.avgMoveCount / stats.games) * 10) / 10 : 0
      dailyStats.push(stats)
    }
    dailyStats.sort((a, b) => a.date.localeCompare(b.date))

    // Calculate cumulative win rate over time
    let cumulativeWins = 0
    let cumulativeGames = 0
    const winRateOverTime = dailyStats.map((day) => {
      cumulativeWins += day.wins
      cumulativeGames += day.games
      return {
        date: day.date,
        winRate: cumulativeGames > 0 ? Math.round((cumulativeWins / cumulativeGames) * 1000) / 10 : 0,
        games: cumulativeGames,
      }
    })

    // Build opening stats sorted by games played
    const openingStats: OpeningStats[] = []
    for (const [column, stats] of openingMap) {
      openingStats.push({
        column,
        games: stats.games,
        wins: stats.wins,
        losses: stats.losses,
        draws: stats.draws,
        winRate: stats.games > 0 ? Math.round((stats.wins / stats.games) * 1000) / 10 : 0,
      })
    }
    openingStats.sort((a, b) => b.games - a.games)

    // Recent games (last 10)
    const recentGames: RecentGame[] = games.slice(-10).reverse().map((game) => {
      let firstMove: number | null = null
      try {
        const moves = JSON.parse(game.moves) as number[]
        if (moves.length > 0) {
          firstMove = game.player_number === 1 ? moves[0] : (moves[1] ?? null)
        }
      } catch {
        // Skip malformed moves
      }
      return {
        id: game.id,
        outcome: game.outcome,
        opponentType: game.opponent_type,
        aiDifficulty: game.ai_difficulty,
        playerNumber: game.player_number,
        moveCount: game.move_count,
        ratingChange: game.rating_change,
        createdAt: game.created_at,
        firstMove,
      }
    })

    // Calculate weekly aggregation
    const weeklyMap = new Map<string, { games: number; wins: number; losses: number; draws: number }>()
    for (const game of games) {
      const date = new Date(game.created_at)
      // Get ISO week number
      const startOfYear = new Date(date.getFullYear(), 0, 1)
      const weekNumber = Math.ceil(((date.getTime() - startOfYear.getTime()) / 86400000 + startOfYear.getDay() + 1) / 7)
      const weekKey = `${date.getFullYear()}-W${weekNumber.toString().padStart(2, '0')}`

      const existing = weeklyMap.get(weekKey) || { games: 0, wins: 0, losses: 0, draws: 0 }
      existing.games++
      if (game.outcome === 'win') existing.wins++
      else if (game.outcome === 'loss') existing.losses++
      else existing.draws++
      weeklyMap.set(weekKey, existing)
    }

    const weeklyStats = Array.from(weeklyMap.entries())
      .map(([week, stats]) => ({ week, ...stats }))
      .sort((a, b) => a.week.localeCompare(b.week))

    return jsonResponse({
      dateRange: {
        start: startDate,
        end: endDate,
      },
      summary: {
        totalGames: games.length,
        wins: games.filter((g) => g.outcome === 'win').length,
        losses: games.filter((g) => g.outcome === 'loss').length,
        draws: games.filter((g) => g.outcome === 'draw').length,
        avgMovesToWin: winCount > 0 ? Math.round((totalWinMoves / winCount) * 10) / 10 : 0,
        avgMovesToLoss: lossCount > 0 ? Math.round((totalLossMoves / lossCount) * 10) / 10 : 0,
        player1WinRate: player1Games > 0 ? Math.round((player1Wins / player1Games) * 1000) / 10 : 0,
        player2WinRate: player2Games > 0 ? Math.round((player2Wins / player2Games) * 1000) / 10 : 0,
      },
      dailyStats,
      weeklyStats,
      winRateOverTime,
      openingStats,
      recentGames,
    })
  } catch (error) {
    console.error('Stats history endpoint error:', error)
    return errorResponse('Internal server error', 500)
  }
}
