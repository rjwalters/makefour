import { test as base, expect, type Page } from '@playwright/test'
import { randomBytes } from 'node:crypto'
import { generateTestUser, selectors, type TestUser, timeouts } from './test-data'

/**
 * Extended test fixtures for MakeFour E2E tests
 */
export const test = base.extend<{
  testUser: TestUser
  authenticatedPage: Page
}>({
  // Model independent clients while keeping local rate limits active.
  // Leave configured headers unchanged when targeting a remote environment.
  context: async ({ context, baseURL, extraHTTPHeaders }, use) => {
    if (baseURL && ['localhost', '127.0.0.1'].includes(new URL(baseURL).hostname)) {
      const client = randomBytes(8).toString('hex').replace(/(.{4})(?=.)/g, '$1:')
      await context.setExtraHTTPHeaders({ ...extraHTTPHeaders, 'CF-Connecting-IP': `2001:db8:${client}::1` })
    }
    await use(context)
  },
  // Generate a unique test user for each test
  // biome-ignore lint/correctness/noEmptyPattern: Playwright fixture pattern
  testUser: async ({}, use) => {
    const user = generateTestUser()
    await use(user)
  },

  // Provide an authenticated page
  authenticatedPage: async ({ page, testUser }, use) => {
    await registerAndLogin(page, testUser)
    await use(page)
  },
})

export { expect }
export type { TestUser }

/**
 * Register a new user account
 */
export async function registerUser(page: Page, user: TestUser): Promise<void> {
  await page.goto('/login')

  // Switch to registration mode
  await page.click('button:has-text("New user?")')

  // Fill registration form
  await page.fill(selectors.loginForm.username, user.username)
  await page.fill(selectors.loginForm.email, user.email)
  await page.fill(selectors.loginForm.password, user.password)

  // Submit and wait for navigation
  await Promise.all([
    page.waitForURL('/play?mode=training', { timeout: timeouts.navigation }),
    page.click(selectors.loginForm.submitButton),
  ])
}

/**
 * Log in with existing credentials
 */
export async function loginUser(page: Page, user: TestUser): Promise<void> {
  await page.goto('/login')

  // Fill login form
  await page.fill(selectors.loginForm.email, user.email)
  await page.fill(selectors.loginForm.password, user.password)

  // Submit and wait for navigation
  await Promise.all([
    page.waitForURL('/play?mode=training', { timeout: timeouts.navigation }),
    page.click(selectors.loginForm.submitButton),
  ])
}

/**
 * Register and log in (for fresh test users)
 */
export async function registerAndLogin(page: Page, user: TestUser): Promise<void> {
  await registerUser(page, user)
}

/**
 * Log out the current user
 */
export async function logoutUser(page: Page): Promise<void> {
  // Look for logout button in navigation or profile dropdown
  const logoutButton = page.locator('button:has-text("Logout"), button:has-text("Log out")')
  if (await logoutButton.isVisible()) {
    await logoutButton.click()
    await page.waitForURL('/login', { timeout: timeouts.navigation })
  }
}

/**
 * Clear authentication state (localStorage)
 */
export async function clearAuthState(page: Page): Promise<void> {
  await page.evaluate(() => {
    localStorage.removeItem('makefour_session_token')
    localStorage.removeItem('makefour_dek')
  })
}

/**
 * Check if user is authenticated
 */
export async function isAuthenticated(page: Page): Promise<boolean> {
  return await page.evaluate(() => {
    return !!localStorage.getItem('makefour_session_token')
  })
}

/**
 * Navigate to a protected route (assumes authenticated)
 */
export async function navigateTo(page: Page, route: string): Promise<void> {
  await page.goto(route)
  await page.waitForLoadState('networkidle')
}
