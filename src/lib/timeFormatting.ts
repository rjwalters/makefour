/**
 * Shared time formatting utilities
 *
 * Centralizes time formatting logic used across multiple components
 * for consistent display of durations and timestamps.
 */

/**
 * Formats milliseconds as M:SS (e.g., "5:03", "0:45")
 * Optionally shows tenths of seconds for precision timing.
 *
 * @param ms - Time in milliseconds
 * @param showTenths - If true, shows tenths of seconds (e.g., "0:05.3")
 * @returns Formatted time string
 */
export function formatTimeMs(ms: number, showTenths: boolean = false): string {
  const totalSeconds = Math.floor(ms / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60

  const base = `${minutes}:${seconds.toString().padStart(2, '0')}`

  if (showTenths) {
    const tenths = Math.floor((ms % 1000) / 100)
    return `${base}.${tenths}`
  }

  return base
}

/**
 * Formats seconds as M:SS (e.g., "5:03", "0:45")
 *
 * @param seconds - Time in seconds
 * @returns Formatted time string
 */
export function formatTimeSeconds(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

/**
 * Formats a timestamp as HH:MM (locale time)
 * Uses the browser's locale settings.
 *
 * @param timestamp - Unix timestamp in milliseconds
 * @returns Formatted time of day string
 */
export function formatTimeOfDay(timestamp: number): string {
  const date = new Date(timestamp)
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}
