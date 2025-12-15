/**
 * Bot Avatar Component
 *
 * Displays a bot's avatar (emoji or fallback icon).
 * Supports multiple sizes and consistent styling across the app.
 */

import { cn } from '../lib/utils'

interface BotAvatarProps {
  avatarUrl: string | null
  name: string
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl'
  className?: string
}

const SIZE_CLASSES = {
  xs: 'w-5 h-5 text-sm',
  sm: 'w-8 h-8 text-lg',
  md: 'w-10 h-10 text-xl',
  lg: 'w-12 h-12 text-2xl',
  xl: 'w-16 h-16 text-3xl',
}

/**
 * Check if a string is an emoji avatar (vs a URL)
 */
function isEmoji(str: string): boolean {
  // Emoji strings are typically 1-4 characters and contain emoji codepoints
  return str.length <= 8 && /\p{Emoji}/u.test(str)
}

export default function BotAvatar({
  avatarUrl,
  name,
  size = 'md',
  className,
}: BotAvatarProps) {
  const sizeClass = SIZE_CLASSES[size]

  // If we have an emoji avatar, display it directly
  if (avatarUrl && isEmoji(avatarUrl)) {
    return (
      <div
        className={cn(
          'flex items-center justify-center rounded-full bg-purple-100 dark:bg-purple-900/50',
          sizeClass,
          className
        )}
        title={name}
        role="img"
        aria-label={`${name} avatar`}
      >
        <span className="leading-none">{avatarUrl}</span>
      </div>
    )
  }

  // If we have a URL avatar, display as image
  if (avatarUrl) {
    return (
      <img
        src={avatarUrl}
        alt={`${name} avatar`}
        className={cn('rounded-full object-cover', sizeClass, className)}
      />
    )
  }

  // Fallback: robot icon
  return (
    <div
      className={cn(
        'flex items-center justify-center rounded-full bg-purple-100 dark:bg-purple-900/50 text-purple-700 dark:text-purple-300',
        sizeClass,
        className
      )}
      title={name}
      role="img"
      aria-label={`${name} avatar`}
    >
      <svg
        className="w-1/2 h-1/2"
        viewBox="0 0 24 24"
        fill="currentColor"
      >
        <path d="M12 2a2 2 0 012 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 017 7h1a1 1 0 011 1v3a1 1 0 01-1 1h-1v1a2 2 0 01-2 2H5a2 2 0 01-2-2v-1H2a1 1 0 01-1-1v-3a1 1 0 011-1h1a7 7 0 017-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 012-2M7.5 13A1.5 1.5 0 006 14.5 1.5 1.5 0 007.5 16 1.5 1.5 0 009 14.5 1.5 1.5 0 007.5 13m9 0a1.5 1.5 0 00-1.5 1.5 1.5 1.5 0 001.5 1.5 1.5 1.5 0 001.5-1.5 1.5 1.5 0 00-1.5-1.5z" />
      </svg>
    </div>
  )
}
