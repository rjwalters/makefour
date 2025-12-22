/**
 * Error handling utilities
 *
 * Provides consistent error message extraction and handling across the application.
 */

/**
 * Extract a user-friendly error message from an unknown error value.
 * Handles Error objects, strings, and objects with message/error properties.
 *
 * @param err - The error to extract a message from
 * @param fallback - Fallback message if no message can be extracted (default: 'An error occurred')
 * @returns A string error message
 *
 * @example
 * try {
 *   await someOperation()
 * } catch (err) {
 *   setError(getErrorMessage(err))
 * }
 */
export function getErrorMessage(err: unknown, fallback = 'An error occurred'): string {
  if (err instanceof Error) {
    return err.message
  }

  if (typeof err === 'string') {
    return err
  }

  if (err && typeof err === 'object') {
    // Handle API error responses with { error: string } or { message: string }
    if ('error' in err && typeof (err as { error: unknown }).error === 'string') {
      return (err as { error: string }).error
    }
    if ('message' in err && typeof (err as { message: unknown }).message === 'string') {
      return (err as { message: string }).message
    }
  }

  return fallback
}

/**
 * Check if an error is a network-related error (fetch failed, no connection, etc.)
 *
 * @param err - The error to check
 * @returns True if the error appears to be network-related
 */
export function isNetworkError(err: unknown): boolean {
  if (err instanceof TypeError) {
    // TypeError: Failed to fetch is a common network error
    const message = err.message.toLowerCase()
    return (
      message.includes('failed to fetch') ||
      message.includes('network') ||
      message.includes('connection')
    )
  }

  const message = getErrorMessage(err, '').toLowerCase()
  return (
    message.includes('network') ||
    message.includes('offline') ||
    message.includes('connection') ||
    message.includes('timeout')
  )
}

/**
 * Check if an error indicates an authentication problem (401, expired session, etc.)
 *
 * @param err - The error to check
 * @returns True if the error appears to be auth-related
 */
export function isAuthError(err: unknown): boolean {
  const message = getErrorMessage(err, '').toLowerCase()
  return (
    message.includes('unauthorized') ||
    message.includes('authentication') ||
    message.includes('not authenticated') ||
    message.includes('session expired') ||
    message.includes('invalid token') ||
    message.includes('401')
  )
}

/**
 * Check if an error indicates a "not found" condition (404)
 *
 * @param err - The error to check
 * @returns True if the error appears to be a not-found error
 */
export function isNotFoundError(err: unknown): boolean {
  const message = getErrorMessage(err, '').toLowerCase()
  return (
    message.includes('not found') ||
    message.includes('404') ||
    message.includes('does not exist')
  )
}

/**
 * Check if an error is a validation error (usually 400 Bad Request)
 *
 * @param err - The error to check
 * @returns True if the error appears to be a validation error
 */
export function isValidationError(err: unknown): boolean {
  const message = getErrorMessage(err, '').toLowerCase()
  return (
    message.includes('invalid') ||
    message.includes('validation') ||
    message.includes('required') ||
    message.includes('must be') ||
    message.includes('cannot be')
  )
}

/**
 * Log an error with context for debugging.
 * In production, this could be extended to send to an error tracking service.
 *
 * @param context - A description of where/what the error occurred
 * @param err - The error to log
 */
export function logError(context: string, err: unknown): void {
  const message = getErrorMessage(err)
  console.error(`[${context}]`, message, err)
}

/**
 * Create a standardized error response object for API-like responses.
 *
 * @param err - The error to convert
 * @param defaultMessage - Default message if error cannot be extracted
 * @returns An object with success: false and an error message
 */
export function toErrorResponse(
  err: unknown,
  defaultMessage = 'An error occurred'
): { success: false; error: string } {
  return {
    success: false,
    error: getErrorMessage(err, defaultMessage),
  }
}
