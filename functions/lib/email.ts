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

/**
 * Send an email using the configured provider
 */
export async function sendEmail(options: EmailOptions, env: EmailEnv): Promise<boolean> {
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
