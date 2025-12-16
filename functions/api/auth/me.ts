interface Env {
  DB: D1Database
}

export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    // Get session token from Authorization header or query param
    const authHeader = context.request.headers.get('Authorization')
    const url = new URL(context.request.url)
    const session_token = authHeader?.replace('Bearer ', '') || url.searchParams.get('session_token')

    if (!session_token) {
      return new Response(
        JSON.stringify({
          error: 'Session token required',
        }),
        {
          status: 401,
          headers: { 'Content-Type': 'application/json' },
        }
      )
    }

    // Find session token and check if it's expired
    const session = await DB.prepare(
      'SELECT * FROM session_tokens WHERE id = ?'
    )
      .bind(session_token)
      .first()

    if (!session) {
      return new Response(
        JSON.stringify({
          error: 'Invalid session token',
        }),
        {
          status: 401,
          headers: { 'Content-Type': 'application/json' },
        }
      )
    }

    // Check if expired
    if (session.expires_at < Date.now()) {
      // Delete expired token
      await DB.prepare('DELETE FROM session_tokens WHERE id = ?').bind(session_token).run()

      return new Response(
        JSON.stringify({
          error: 'Session expired',
        }),
        {
          status: 401,
          headers: { 'Content-Type': 'application/json' },
        }
      )
    }

    // Get user with rating information
    const user = await DB.prepare(
      `SELECT id, email, email_verified, username, rating, games_played, wins, losses, draws,
              created_at, last_login, updated_at
       FROM users WHERE id = ?`
    )
      .bind(session.user_id)
      .first<{
        id: string
        email: string
        email_verified: number
        username: string | null
        rating: number
        games_played: number
        wins: number
        losses: number
        draws: number
        created_at: number
        last_login: number
        updated_at: number
      }>()

    if (!user) {
      return new Response(
        JSON.stringify({
          error: 'User not found',
        }),
        {
          status: 404,
          headers: { 'Content-Type': 'application/json' },
        }
      )
    }

    return new Response(
      JSON.stringify({
        user: {
          id: user.id,
          email: user.email,
          email_verified: user.email_verified === 1,
          username: user.username,
          displayName: user.username || user.email.split('@')[0],
          rating: user.rating,
          gamesPlayed: user.games_played,
          wins: user.wins,
          losses: user.losses,
          draws: user.draws,
          createdAt: user.created_at,
          lastLogin: user.last_login,
          updatedAt: user.updated_at,
        },
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }
    )
  } catch (error) {
    console.error('Me endpoint error:', error)

    return new Response(
      JSON.stringify({
        error: 'Internal server error',
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    )
  }
}
