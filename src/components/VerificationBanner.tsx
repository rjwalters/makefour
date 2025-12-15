import { useState } from 'react'
import { useAuth } from '../contexts/AuthContext'

interface VerificationBannerProps {
  className?: string
}

export function VerificationBanner({ className = '' }: VerificationBannerProps) {
  const { user, resendVerification } = useAuth()
  const [isResending, setIsResending] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [dismissed, setDismissed] = useState(false)

  // Don't show if user is verified, not logged in, or banner was dismissed
  if (!user || user.email_verified || dismissed) {
    return null
  }

  const handleResend = async () => {
    setIsResending(true)
    setMessage(null)

    const result = await resendVerification()

    if (result.success) {
      setMessage({ type: 'success', text: 'Verification email sent! Check your inbox.' })
    } else {
      setMessage({ type: 'error', text: result.error || 'Failed to send email. Try again later.' })
    }

    setIsResending(false)
  }

  return (
    <div
      className={`bg-yellow-50 dark:bg-yellow-900/20 border-b border-yellow-200 dark:border-yellow-800 ${className}`}
    >
      <div className="max-w-7xl mx-auto px-4 py-3 sm:px-6 lg:px-8">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <svg
              className="h-5 w-5 text-yellow-600 dark:text-yellow-500 flex-shrink-0"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fillRule="evenodd"
                d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z"
                clipRule="evenodd"
              />
            </svg>
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <span className="font-medium">Verify your email</span> to unlock online matchmaking
              and appear on the leaderboard.
            </p>
          </div>

          <div className="flex items-center gap-3">
            {message && (
              <span
                className={`text-sm ${
                  message.type === 'success'
                    ? 'text-green-600 dark:text-green-400'
                    : 'text-red-600 dark:text-red-400'
                }`}
              >
                {message.text}
              </span>
            )}

            <button
              onClick={handleResend}
              disabled={isResending}
              className="text-sm font-medium text-yellow-800 dark:text-yellow-200 hover:text-yellow-900 dark:hover:text-yellow-100 underline disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isResending ? 'Sending...' : 'Resend email'}
            </button>

            <button
              onClick={() => setDismissed(true)}
              className="text-yellow-600 dark:text-yellow-500 hover:text-yellow-800 dark:hover:text-yellow-300"
              aria-label="Dismiss"
            >
              <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
