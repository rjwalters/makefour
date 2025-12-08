interface Env {
  DB: D1Database
}

export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    // Get session token from Authorization header
    const authHeader = context.request.headers.get('Authorization')
    const session_token = authHeader?.replace('Bearer ', '')

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

    // Delete session token
    await DB.prepare('DELETE FROM session_tokens WHERE id = ?').bind(session_token).run()

    return new Response(
      JSON.stringify({
        message: 'Logged out successfully',
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }
    )
  } catch (error) {
    console.error('Logout error:', error)

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
