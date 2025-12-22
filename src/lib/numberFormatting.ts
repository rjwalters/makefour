/**
 * Shared number formatting utilities
 *
 * Centralizes number formatting logic used across multiple components
 * for consistent display of percentages, rates, and statistics.
 */

/**
 * Calculates win rate as a percentage from wins and total games.
 *
 * @param wins - Number of wins
 * @param totalGames - Total number of games played
 * @returns Win rate as a rounded integer percentage (0-100), or 0 if no games
 *
 * @example
 * calculateWinRate(7, 10) // 70
 * calculateWinRate(3, 7)  // 43
 * calculateWinRate(0, 0)  // 0
 */
export function calculateWinRate(wins: number, totalGames: number): number {
  if (totalGames === 0) return 0
  return Math.round((wins / totalGames) * 100)
}

/**
 * Formats a decimal value as a percentage string.
 *
 * @param value - Decimal value (e.g., 0.75 for 75%)
 * @param decimals - Number of decimal places (default: 0)
 * @returns Formatted percentage string with % suffix
 *
 * @example
 * formatPercent(0.753)    // "75%"
 * formatPercent(0.753, 1) // "75.3%"
 * formatPercent(0.5)      // "50%"
 */
export function formatPercent(value: number, decimals: number = 0): string {
  return `${(value * 100).toFixed(decimals)}%`
}

/**
 * Formats a ratio as a percentage string.
 *
 * @param numerator - The numerator of the ratio
 * @param denominator - The denominator of the ratio
 * @param decimals - Number of decimal places (default: 0)
 * @returns Formatted percentage string, or "0%" if denominator is 0
 *
 * @example
 * formatRatioAsPercent(7, 10)    // "70%"
 * formatRatioAsPercent(3, 7, 1)  // "42.9%"
 * formatRatioAsPercent(0, 0)     // "0%"
 */
export function formatRatioAsPercent(
  numerator: number,
  denominator: number,
  decimals: number = 0
): string {
  if (denominator === 0) return '0%'
  return formatPercent(numerator / denominator, decimals)
}

/**
 * Formats a number with a fixed number of decimal places.
 *
 * @param value - The number to format
 * @param decimals - Number of decimal places (default: 1)
 * @returns Formatted number string
 *
 * @example
 * formatDecimal(3.14159)    // "3.1"
 * formatDecimal(3.14159, 2) // "3.14"
 * formatDecimal(5)          // "5.0"
 */
export function formatDecimal(value: number, decimals: number = 1): string {
  return value.toFixed(decimals)
}

/**
 * Returns a color class based on win rate thresholds.
 * Useful for displaying win rates with color indicators.
 *
 * @param winRate - Win rate as a percentage (0-100)
 * @param variant - 'text' for text colors, 'bg' for background colors
 * @returns Tailwind color class string
 *
 * @example
 * getWinRateColor(70) // "text-green-600 dark:text-green-400"
 * getWinRateColor(45) // "text-yellow-600 dark:text-yellow-400"
 * getWinRateColor(30) // "text-red-600 dark:text-red-400"
 * getWinRateColor(70, 'bg') // "bg-green-500"
 */
export function getWinRateColor(
  winRate: number,
  variant: 'text' | 'bg' = 'text'
): string {
  if (variant === 'bg') {
    if (winRate >= 60) return 'bg-green-500'
    if (winRate >= 40) return 'bg-yellow-500'
    return 'bg-red-500'
  }

  if (winRate >= 60) return 'text-green-600 dark:text-green-400'
  if (winRate >= 40) return 'text-yellow-600 dark:text-yellow-400'
  return 'text-red-600 dark:text-red-400'
}

/**
 * Formats a large number with K/M suffixes for readability.
 *
 * @param num - The number to format
 * @param decimals - Number of decimal places for suffixed numbers (default: 1)
 * @returns Formatted string with appropriate suffix
 *
 * @example
 * formatCompactNumber(999)      // "999"
 * formatCompactNumber(1234)     // "1.2K"
 * formatCompactNumber(1234567)  // "1.2M"
 */
export function formatCompactNumber(num: number, decimals: number = 1): string {
  if (num >= 1_000_000) {
    return `${(num / 1_000_000).toFixed(decimals)}M`
  }
  if (num >= 1_000) {
    return `${(num / 1_000).toFixed(decimals)}K`
  }
  return num.toString()
}
