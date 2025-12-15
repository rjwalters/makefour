/**
 * Email service for Cloudflare Workers
 *
 * Supports multiple providers through environment configuration.
 * Set EMAIL_PROVIDER to 'resend', 'sendgrid', 'mailgun', or 'console' (for development)
 */

interface EmailOptions {
  to: string
  subject: string
  html: string
  text?: string
}

interface EmailEnv {
  EMAIL_PROVIDER?: string
  EMAIL_API_KEY?: string
  EMAIL_FROM?: string
  EMAIL_DOMAIN?: string // For Mailgun
}

// Legacy interface for backward compatibility with verification email code
export interface EmailConfig {
  apiKey: string
  fromAddress?: string
  baseUrl?: string
}

export interface SendEmailResult {
  success: boolean
  id?: string
  error?: string
}

/**
 * Send an email using the configured provider
 */
export async function sendEmail(options: EmailOptions, env: EmailEnv): Promise<boolean>
export async function sendEmail(config: EmailConfig, params: EmailOptions): Promise<SendEmailResult>
export async function sendEmail(
  configOrOptions: EmailConfig | EmailOptions,
  envOrParams: EmailEnv | EmailOptions
): Promise<boolean | SendEmailResult> {
  // Detect which overload is being used
  if ('apiKey' in configOrOptions) {
    // Legacy verification email path using EmailConfig
    const config = configOrOptions as EmailConfig
    const params = envOrParams as EmailOptions
    const from = config.fromAddress || 'MakeFour <onboarding@resend.dev>'

    try {
      const response = await fetch('https://api.resend.com/emails', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${config.apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          from,
          to: [params.to],
          subject: params.subject,
          html: params.html,
          text: params.text,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        console.error('Resend API error:', response.status, errorData)
        return {
          success: false,
          error: (errorData as { message?: string }).message || `Email send failed with status ${response.status}`,
        }
      }

      const data = await response.json()
      return {
        success: true,
        id: (data as { id?: string }).id,
      }
    } catch (error) {
      console.error('Email send error:', error)
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown email error',
      }
    }
  }

  // Multi-provider path using EmailEnv
  const options = configOrOptions as EmailOptions
  const env = envOrParams as EmailEnv
  const provider = env.EMAIL_PROVIDER || 'console'
  const from = env.EMAIL_FROM || 'noreply@makefour.com'

  try {
    switch (provider) {
      case 'resend':
        return await sendWithResend(options, from, env.EMAIL_API_KEY!)

      case 'sendgrid':
        return await sendWithSendGrid(options, from, env.EMAIL_API_KEY!)

      case 'mailgun':
        return await sendWithMailgun(options, from, env.EMAIL_API_KEY!, env.EMAIL_DOMAIN!)

      case 'console':
      default:
        // Development mode - log to console
        console.log('=== EMAIL (Development Mode) ===')
        console.log(`To: ${options.to}`)
        console.log(`From: ${from}`)
        console.log(`Subject: ${options.subject}`)
        console.log(`Body: ${options.text || options.html}`)
        console.log('================================')
        return true
    }
  } catch (error) {
    console.error('Email send error:', error)
    return false
  }
}

async function sendWithResend(options: EmailOptions, from: string, apiKey: string): Promise<boolean> {
  const response = await fetch('https://api.resend.com/emails', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      from,
      to: options.to,
      subject: options.subject,
      html: options.html,
      text: options.text,
    }),
  })

  if (!response.ok) {
    const error = await response.text()
    console.error('Resend error:', error)
    return false
  }

  return true
}

async function sendWithSendGrid(options: EmailOptions, from: string, apiKey: string): Promise<boolean> {
  const response = await fetch('https://api.sendgrid.com/v3/mail/send', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      personalizations: [{ to: [{ email: options.to }] }],
      from: { email: from },
      subject: options.subject,
      content: [
        { type: 'text/plain', value: options.text || options.html },
        { type: 'text/html', value: options.html },
      ],
    }),
  })

  if (!response.ok) {
    const error = await response.text()
    console.error('SendGrid error:', error)
    return false
  }

  return true
}

async function sendWithMailgun(
  options: EmailOptions,
  from: string,
  apiKey: string,
  domain: string
): Promise<boolean> {
  const formData = new FormData()
  formData.append('from', from)
  formData.append('to', options.to)
  formData.append('subject', options.subject)
  formData.append('html', options.html)
  if (options.text) {
    formData.append('text', options.text)
  }

  const response = await fetch(`https://api.mailgun.net/v3/${domain}/messages`, {
    method: 'POST',
    headers: {
      'Authorization': `Basic ${btoa(`api:${apiKey}`)}`,
    },
    body: formData,
  })

  if (!response.ok) {
    const error = await response.text()
    console.error('Mailgun error:', error)
    return false
  }

  return true
}

/**
 * Generate password reset email HTML
 */
export function generatePasswordResetEmail(resetUrl: string, expiresInMinutes: number = 60): {
  subject: string
  html: string
  text: string
} {
  const subject = 'Reset Your Password - MakeFour'

  const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Reset Your Password</title>
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
  <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 8px 8px 0 0;">
    <h1 style="color: white; margin: 0; font-size: 24px;">Reset Your Password</h1>
  </div>

  <div style="background: #f9fafb; padding: 30px; border: 1px solid #e5e7eb; border-top: none; border-radius: 0 0 8px 8px;">
    <p style="margin-top: 0;">We received a request to reset your password. Click the button below to create a new password:</p>

    <div style="text-align: center; margin: 30px 0;">
      <a href="${resetUrl}" style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 14px 28px; text-decoration: none; border-radius: 6px; font-weight: 600; font-size: 16px;">Reset Password</a>
    </div>

    <p style="color: #6b7280; font-size: 14px;">This link will expire in <strong>${expiresInMinutes} minutes</strong>.</p>

    <p style="color: #6b7280; font-size: 14px;">If you can't click the button, copy and paste this link into your browser:</p>
    <p style="word-break: break-all; color: #667eea; font-size: 14px;">${resetUrl}</p>

    <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 24px 0;">

    <p style="color: #9ca3af; font-size: 13px; margin-bottom: 0;">
      <strong>Didn't request this?</strong><br>
      If you didn't request a password reset, you can safely ignore this email. Your password will remain unchanged.
    </p>
  </div>

  <div style="text-align: center; padding: 20px; color: #9ca3af; font-size: 12px;">
    <p style="margin: 0;">© ${new Date().getFullYear()} MakeFour. All rights reserved.</p>
  </div>
</body>
</html>
`

  const text = `
Reset Your Password

We received a request to reset your password. Click the link below to create a new password:

${resetUrl}

This link will expire in ${expiresInMinutes} minutes.

Didn't request this?
If you didn't request a password reset, you can safely ignore this email. Your password will remain unchanged.

© ${new Date().getFullYear()} MakeFour. All rights reserved.
`

  return { subject, html, text: text.trim() }
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
