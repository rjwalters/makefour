/**
 * Input validation utilities
 *
 * Centralizes validation logic for usernames, passwords, and other inputs.
 * Provides consistent validation rules and error messages across the app.
 */

// ============================================================================
// VALIDATION CONSTANTS
// ============================================================================

/** Minimum username length */
export const MIN_USERNAME_LENGTH = 3

/** Maximum username length */
export const MAX_USERNAME_LENGTH = 20

/** Minimum password length */
export const MIN_PASSWORD_LENGTH = 8

/** Regex pattern for valid usernames */
export const USERNAME_PATTERN = /^[a-zA-Z][a-zA-Z0-9_]*$/

/** HTML pattern attribute for username inputs */
export const USERNAME_HTML_PATTERN = '^[a-zA-Z][a-zA-Z0-9_]{2,19}$'

// ============================================================================
// VALIDATION RESULT TYPES
// ============================================================================

/**
 * Result of a validation check.
 * If valid, error is null. If invalid, error contains the error message.
 */
export interface ValidationResult {
  isValid: boolean
  error: string | null
}

// ============================================================================
// USERNAME VALIDATION
// ============================================================================

/**
 * Validates a username against all rules.
 *
 * Rules:
 * - Must be between 3-20 characters
 * - Must start with a letter
 * - Can only contain letters, numbers, and underscores
 *
 * @param username - The username to validate
 * @returns Validation result with isValid flag and error message if invalid
 *
 * @example
 * validateUsername('alice')     // { isValid: true, error: null }
 * validateUsername('ab')        // { isValid: false, error: 'Username must be...' }
 * validateUsername('123abc')    // { isValid: false, error: 'Username must start...' }
 */
export function validateUsername(username: string): ValidationResult {
  if (!username) {
    return { isValid: false, error: 'Username is required' }
  }

  if (username.length < MIN_USERNAME_LENGTH) {
    return {
      isValid: false,
      error: `Username must be at least ${MIN_USERNAME_LENGTH} characters`,
    }
  }

  if (username.length > MAX_USERNAME_LENGTH) {
    return {
      isValid: false,
      error: `Username must be at most ${MAX_USERNAME_LENGTH} characters`,
    }
  }

  if (!USERNAME_PATTERN.test(username)) {
    return {
      isValid: false,
      error: 'Username must start with a letter and contain only letters, numbers, and underscores',
    }
  }

  return { isValid: true, error: null }
}

/**
 * Quick check if a username is valid without returning an error message.
 *
 * @param username - The username to validate
 * @returns True if valid, false otherwise
 */
export function isValidUsername(username: string): boolean {
  return validateUsername(username).isValid
}

// ============================================================================
// PASSWORD VALIDATION
// ============================================================================

/**
 * Validates a password against minimum requirements.
 *
 * Rules:
 * - Must be at least 8 characters
 *
 * @param password - The password to validate
 * @returns Validation result with isValid flag and error message if invalid
 *
 * @example
 * validatePassword('secure123')  // { isValid: true, error: null }
 * validatePassword('short')      // { isValid: false, error: 'Password must be...' }
 */
export function validatePassword(password: string): ValidationResult {
  if (!password) {
    return { isValid: false, error: 'Password is required' }
  }

  if (password.length < MIN_PASSWORD_LENGTH) {
    return {
      isValid: false,
      error: `Password must be at least ${MIN_PASSWORD_LENGTH} characters`,
    }
  }

  return { isValid: true, error: null }
}

/**
 * Validates that two passwords match.
 *
 * @param password - The password
 * @param confirmPassword - The confirmation password
 * @returns Validation result
 */
export function validatePasswordMatch(
  password: string,
  confirmPassword: string
): ValidationResult {
  if (password !== confirmPassword) {
    return { isValid: false, error: 'Passwords do not match' }
  }
  return { isValid: true, error: null }
}

/**
 * Quick check if a password is valid without returning an error message.
 *
 * @param password - The password to validate
 * @returns True if valid, false otherwise
 */
export function isValidPassword(password: string): boolean {
  return validatePassword(password).isValid
}

// ============================================================================
// EMAIL VALIDATION
// ============================================================================

/** Basic email regex pattern (for client-side validation) */
const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

/**
 * Validates an email address format.
 * Note: This is basic client-side validation. Server-side validation
 * should be the source of truth for email validity.
 *
 * @param email - The email to validate
 * @returns Validation result
 */
export function validateEmail(email: string): ValidationResult {
  if (!email) {
    return { isValid: false, error: 'Email is required' }
  }

  if (!EMAIL_PATTERN.test(email)) {
    return { isValid: false, error: 'Please enter a valid email address' }
  }

  return { isValid: true, error: null }
}

/**
 * Quick check if an email is valid without returning an error message.
 *
 * @param email - The email to validate
 * @returns True if valid, false otherwise
 */
export function isValidEmail(email: string): boolean {
  return validateEmail(email).isValid
}
