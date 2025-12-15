/**
 * Email service using Resend API
 * https://developers.cloudflare.com/workers/tutorials/send-emails-with-resend/
 */

export interface EmailConfig {
  apiKey: string
  fromAddress?: string
  baseUrl?: string
}

export interface SendEmailParams {
  to: string
  subject: string
  html: string
  text?: string
}

export interface SendEmailResult {
  success: boolean
  id?: string
  error?: string
}

const DEFAULT_FROM_ADDRESS = 'MakeFour <onboarding@resend.dev>'

/**
 * Send an email using the Resend API
 */
export async function sendEmail(
  config: EmailConfig,
  params: SendEmailParams
): Promise<SendEmailResult> {
  const { apiKey, fromAddress = DEFAULT_FROM_ADDRESS, baseUrl } = config
  const { to, subject, html, text } = params

  try {
    const response = await fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from: fromAddress,
        to: [to],
        subject,
        html,
        text,
      }),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      console.error('Resend API error:', response.status, errorData)
      return {
        success: false,
        error: errorData.message || `Email send failed with status ${response.status}`,
      }
    }

    const data = await response.json()
    return {
      success: true,
      id: data.id,
    }
  } catch (error) {
    console.error('Email send error:', error)
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown email error',
    }
  }
}

/**
 * Generate the verification email HTML
 */
export function generateVerificationEmailHtml(
  verifyUrl: string,
  expiresInHours: number = 24
): string {
  return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Verify your MakeFour account</title>
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
  <div style="text-align: center; margin-bottom: 30px;">
    <h1 style="color: #e53e3e; margin: 0;">MakeFour</h1>
    <p style="color: #666; margin: 5px 0 0 0;">Four-in-a-Row Strategy Game</p>
  </div>

  <div style="background: #f7f7f7; border-radius: 8px; padding: 30px; margin-bottom: 20px;">
    <h2 style="margin-top: 0; color: #333;">Verify your email address</h2>
    <p>Thanks for signing up for MakeFour! Please click the button below to verify your email address and unlock all features.</p>

    <div style="text-align: center; margin: 30px 0;">
      <a href="${verifyUrl}" style="display: inline-block; background: #e53e3e; color: white; text-decoration: none; padding: 14px 30px; border-radius: 6px; font-weight: 600;">
        Verify Email Address
      </a>
    </div>

    <p style="color: #666; font-size: 14px;">
      This link will expire in ${expiresInHours} hours. If you didn't create a MakeFour account, you can safely ignore this email.
    </p>
  </div>

  <div style="text-align: center; color: #999; font-size: 12px;">
    <p>If the button doesn't work, copy and paste this link into your browser:</p>
    <p style="word-break: break-all; color: #666;">${verifyUrl}</p>
  </div>
</body>
</html>
`.trim()
}

/**
 * Generate plain text version of verification email
 */
export function generateVerificationEmailText(
  verifyUrl: string,
  expiresInHours: number = 24
): string {
  return `
Verify your MakeFour account

Thanks for signing up for MakeFour! Please click the link below to verify your email address and unlock all features.

Verify your email: ${verifyUrl}

This link will expire in ${expiresInHours} hours.

If you didn't create a MakeFour account, you can safely ignore this email.

---
MakeFour - Four-in-a-Row Strategy Game
`.trim()
}

/**
 * Send a verification email to a user
 */
export async function sendVerificationEmail(
  config: EmailConfig,
  to: string,
  token: string
): Promise<SendEmailResult> {
  const baseUrl = config.baseUrl || 'https://makefour.pages.dev'
  const verifyUrl = `${baseUrl}/verify-email?token=${token}`

  return sendEmail(config, {
    to,
    subject: 'Verify your MakeFour account',
    html: generateVerificationEmailHtml(verifyUrl),
    text: generateVerificationEmailText(verifyUrl),
  })
}

/**
 * Create a verification token and store it in the database
 * Returns the token ID (to be used in the verification URL)
 */
export async function createVerificationToken(
  db: D1Database,
  userId: string,
  expiresInMs: number = 24 * 60 * 60 * 1000 // 24 hours
): Promise<string> {
  const tokenId = crypto.randomUUID()
  const now = Date.now()
  const expiresAt = now + expiresInMs

  await db.prepare(`
    INSERT INTO email_verification_tokens (id, user_id, expires_at, used, created_at)
    VALUES (?, ?, ?, 0, ?)
  `).bind(tokenId, userId, expiresAt, now).run()

  return tokenId
}

/**
 * Validate and consume a verification token
 * Returns the user ID if valid, null if invalid/expired/used
 */
export async function validateVerificationToken(
  db: D1Database,
  tokenId: string
): Promise<{ valid: true; userId: string } | { valid: false; error: string }> {
  const token = await db.prepare(`
    SELECT id, user_id, expires_at, used
    FROM email_verification_tokens
    WHERE id = ?
  `).bind(tokenId).first<{
    id: string
    user_id: string
    expires_at: number
    used: number
  }>()

  if (!token) {
    return { valid: false, error: 'Invalid verification token' }
  }

  if (token.used === 1) {
    return { valid: false, error: 'Token has already been used' }
  }

  if (token.expires_at < Date.now()) {
    return { valid: false, error: 'Token has expired' }
  }

  // Mark token as used
  await db.prepare(`
    UPDATE email_verification_tokens
    SET used = 1
    WHERE id = ?
  `).bind(tokenId).run()

  return { valid: true, userId: token.user_id }
}

/**
 * Mark a user's email as verified
 */
export async function markEmailVerified(
  db: D1Database,
  userId: string
): Promise<void> {
  await db.prepare(`
    UPDATE users
    SET email_verified = 1, updated_at = ?
    WHERE id = ?
  `).bind(Date.now(), userId).run()
}

/**
 * Check if a user's email is verified
 */
export async function isEmailVerified(
  db: D1Database,
  userId: string
): Promise<boolean> {
  const user = await db.prepare(`
    SELECT email_verified FROM users WHERE id = ?
  `).bind(userId).first<{ email_verified: number }>()

  return user?.email_verified === 1
}

/**
 * Delete existing unused verification tokens for a user (for resend)
 */
export async function deleteUnusedTokens(
  db: D1Database,
  userId: string
): Promise<void> {
  await db.prepare(`
    DELETE FROM email_verification_tokens
    WHERE user_id = ? AND used = 0
  `).bind(userId).run()
}
