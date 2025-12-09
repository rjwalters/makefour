/**
 * Google OAuth callback endpoint
 * Handles the redirect from Google after user authorization
 */

interface Env {
  DB: D1Database
  GOOGLE_CLIENT_ID: string
  GOOGLE_CLIENT_SECRET: string
}

interface GoogleTokenResponse {
  access_token: string
  expires_in: number
  token_type: string
  scope: string
  id_token?: string
}

interface GoogleUserInfo {
  id: string
  email: string
  verified_email: boolean
  name?: string
  given_name?: string
  family_name?: string
  picture?: string
}

const SESSION_DURATION_MS = 30 * 24 * 60 * 60 * 1000 // 30 days

export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET } = context.env
  const url = new URL(context.request.url)

  // Get authorization code and state from query params
  const code = url.searchParams.get('code')
  const state = url.searchParams.get('state')
  const error = url.searchParams.get('error')

  // Handle OAuth errors
  if (error) {
    return redirectWithError(`OAuth error: ${error}`)
  }

  if (!code || !state) {
    return redirectWithError('Missing authorization code or state')
  }

  // Verify state from cookie (CSRF protection)
  const cookies = parseCookies(context.request.headers.get('Cookie') || '')
  const storedState = cookies['oauth_state']

  if (!storedState || storedState !== state) {
    return redirectWithError('Invalid state parameter')
  }

  try {
    // Exchange authorization code for access token
    const redirectUri = `${url.origin}/api/auth/google/callback`
    const tokenResponse = await fetch('https://oauth2.googleapis.com/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        code,
        client_id: GOOGLE_CLIENT_ID,
        client_secret: GOOGLE_CLIENT_SECRET,
        redirect_uri: redirectUri,
        grant_type: 'authorization_code',
      }),
    })

    if (!tokenResponse.ok) {
      const errorData = await tokenResponse.text()
      console.error('Token exchange failed:', errorData)
      return redirectWithError('Failed to exchange authorization code')
    }

    const tokenData: GoogleTokenResponse = await tokenResponse.json()

    // Fetch user info from Google
    const userInfoResponse = await fetch('https://www.googleapis.com/oauth2/v2/userinfo', {
      headers: { 'Authorization': `Bearer ${tokenData.access_token}` },
    })

    if (!userInfoResponse.ok) {
      return redirectWithError('Failed to fetch user info from Google')
    }

    const googleUser: GoogleUserInfo = await userInfoResponse.json()

    if (!googleUser.email) {
      return redirectWithError('No email provided by Google')
    }

    // Find or create user in database
    const now = Date.now()
    let user = await DB.prepare(
      'SELECT * FROM users WHERE oauth_provider = ? AND oauth_id = ?'
    ).bind('google', googleUser.id).first()

    if (!user) {
      // Check if email already exists (might have registered with password)
      const existingUser = await DB.prepare(
        'SELECT * FROM users WHERE email = ?'
      ).bind(googleUser.email).first()

      if (existingUser) {
        // Link Google account to existing user
        await DB.prepare(
          'UPDATE users SET oauth_provider = ?, oauth_id = ?, email_verified = 1, updated_at = ? WHERE id = ?'
        ).bind('google', googleUser.id, now, existingUser.id).run()
        user = existingUser
      } else {
        // Create new user
        const userId = crypto.randomUUID()
        await DB.prepare(
          `INSERT INTO users (id, email, email_verified, oauth_provider, oauth_id, created_at, last_login, updated_at)
           VALUES (?, ?, 1, ?, ?, ?, ?, ?)`
        ).bind(userId, googleUser.email, 'google', googleUser.id, now, now, now).run()

        user = {
          id: userId,
          email: googleUser.email,
          email_verified: 1,
          oauth_provider: 'google',
          oauth_id: googleUser.id,
          created_at: now,
          last_login: now,
          updated_at: now,
        }
      }
    } else {
      // Update last login
      await DB.prepare(
        'UPDATE users SET last_login = ?, updated_at = ? WHERE id = ?'
      ).bind(now, now, user.id).run()
    }

    // Create session token
    const sessionTokenId = crypto.randomUUID()
    const expiresAt = now + SESSION_DURATION_MS

    await DB.prepare(
      'INSERT INTO session_tokens (id, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)'
    ).bind(sessionTokenId, user.id, expiresAt, now).run()

    // Redirect to frontend with session token
    // The frontend will store this and complete the auth flow
    const successUrl = new URL('/', url.origin)
    successUrl.searchParams.set('oauth_token', sessionTokenId)
    successUrl.searchParams.set('oauth_provider', 'google')

    // Clear the state cookie
    const clearStateCookie = 'oauth_state=; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=0'

    return new Response(null, {
      status: 302,
      headers: {
        'Location': successUrl.toString(),
        'Set-Cookie': clearStateCookie,
      },
    })
  } catch (err) {
    console.error('OAuth callback error:', err)
    return redirectWithError('An unexpected error occurred')
  }
}

function redirectWithError(message: string): Response {
  const errorUrl = new URL('/login')
  errorUrl.searchParams.set('error', message)

  return new Response(null, {
    status: 302,
    headers: {
      'Location': errorUrl.toString(),
      'Set-Cookie': 'oauth_state=; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=0',
    },
  })
}

function parseCookies(cookieHeader: string): Record<string, string> {
  const cookies: Record<string, string> = {}
  if (!cookieHeader) return cookies

  for (const cookie of cookieHeader.split(';')) {
    const [name, ...rest] = cookie.trim().split('=')
    if (name && rest.length > 0) {
      cookies[name] = rest.join('=')
    }
  }
  return cookies
}
