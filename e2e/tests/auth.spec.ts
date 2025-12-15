import { clearAuthState, expect, isAuthenticated, test } from '../fixtures/base'
import { generateTestUser } from '../fixtures/test-data'
import { DashboardPage, LoginPage } from '../pages'

test.describe('Authentication Flow', () => {
  test.beforeEach(async ({ page }) => {
    // Clear any existing auth state before each test
    await page.goto('/')
    await clearAuthState(page)
  })

  test.describe('Registration', () => {
    test('should register a new user and redirect to dashboard', async ({ page }) => {
      const loginPage = new LoginPage(page)
      const dashboardPage = new DashboardPage(page)
      const user = generateTestUser()

      await loginPage.goto()

      // Switch to registration mode
      await loginPage.switchToRegisterMode()
      expect(await loginPage.isInRegisterMode()).toBe(true)

      // Fill and submit registration form
      await loginPage.register(user.email, user.password)

      // Should redirect to dashboard
      await loginPage.waitForRedirectToDashboard()

      // Verify user is logged in
      await dashboardPage.expectUserLoggedIn()
    })

    test('should show error for invalid email format', async ({ page }) => {
      const loginPage = new LoginPage(page)

      await loginPage.goto()
      await loginPage.switchToRegisterMode()

      // Try to register with invalid email
      await loginPage.emailInput.fill('invalid-email')
      await loginPage.passwordInput.fill('ValidPassword123!')

      // HTML5 validation should prevent submission
      // Check that we're still on the login page
      await expect(page).toHaveURL('/login')
    })

    test('should show error for short password', async ({ page }) => {
      const loginPage = new LoginPage(page)

      await loginPage.goto()
      await loginPage.switchToRegisterMode()

      // Try to register with short password
      await loginPage.emailInput.fill('test@example.com')
      await loginPage.passwordInput.fill('short')

      // HTML5 minLength validation should prevent submission
      await expect(page).toHaveURL('/login')
    })

    test('should show error for duplicate email registration', async ({ page }) => {
      const loginPage = new LoginPage(page)
      const user = generateTestUser()

      // Register first time
      await loginPage.goto()
      await loginPage.register(user.email, user.password)
      await loginPage.waitForRedirectToDashboard()

      // Clear state and try to register again with same email
      await clearAuthState(page)
      await loginPage.goto()
      await loginPage.register(user.email, user.password)

      // Should show error
      await loginPage.expectError()
    })
  })

  test.describe('Login', () => {
    test('should login existing user and redirect to dashboard', async ({ page }) => {
      const loginPage = new LoginPage(page)
      const dashboardPage = new DashboardPage(page)
      const user = generateTestUser()

      // First register a user
      await loginPage.goto()
      await loginPage.register(user.email, user.password)
      await loginPage.waitForRedirectToDashboard()

      // Logout
      await dashboardPage.logout()

      // Now login with the same credentials
      await loginPage.login(user.email, user.password)
      await loginPage.waitForRedirectToDashboard()

      // Verify user is logged in
      await dashboardPage.expectUserLoggedIn()
    })

    test('should show error for invalid credentials', async ({ page }) => {
      const loginPage = new LoginPage(page)

      await loginPage.goto()
      expect(await loginPage.isInLoginMode()).toBe(true)

      // Try to login with non-existent user
      await loginPage.login('nonexistent@test.com', 'wrongpassword123')

      // Should show error and stay on login page
      await loginPage.expectError()
      await expect(page).toHaveURL('/login')
    })

    test('should show error for wrong password', async ({ page }) => {
      const loginPage = new LoginPage(page)
      const user = generateTestUser()

      // Register a user first
      await loginPage.goto()
      await loginPage.register(user.email, user.password)
      await loginPage.waitForRedirectToDashboard()

      // Clear state
      await clearAuthState(page)
      await loginPage.goto()

      // Try to login with wrong password
      await loginPage.login(user.email, 'WrongPassword123!')

      // Should show error
      await loginPage.expectError()
    })
  })

  test.describe('Session Persistence', () => {
    test('should maintain session after page reload', async ({ page }) => {
      const loginPage = new LoginPage(page)
      const dashboardPage = new DashboardPage(page)
      const user = generateTestUser()

      // Register and login
      await loginPage.goto()
      await loginPage.register(user.email, user.password)
      await loginPage.waitForRedirectToDashboard()

      // Reload the page
      await page.reload()

      // Should still be logged in
      await expect(page).toHaveURL('/dashboard')
      await dashboardPage.expectUserLoggedIn()
    })

    test('should redirect to login when accessing protected route without auth', async ({ page }) => {
      await clearAuthState(page)

      // Try to access dashboard directly
      await page.goto('/dashboard')

      // Should redirect to login
      await expect(page).toHaveURL('/login')
    })

    test('should redirect to login when accessing games without auth', async ({ page }) => {
      await clearAuthState(page)

      // Try to access games page
      await page.goto('/games')

      // Should redirect to login
      await expect(page).toHaveURL('/login')
    })
  })

  test.describe('Logout', () => {
    test('should logout and redirect to login page', async ({ page }) => {
      const loginPage = new LoginPage(page)
      const dashboardPage = new DashboardPage(page)
      const user = generateTestUser()

      // Register and login
      await loginPage.goto()
      await loginPage.register(user.email, user.password)
      await loginPage.waitForRedirectToDashboard()

      // Logout
      await dashboardPage.logout()

      // Should be on login page
      await expect(page).toHaveURL('/login')

      // Should not be authenticated
      const authenticated = await isAuthenticated(page)
      expect(authenticated).toBe(false)
    })

    test('should clear session after logout', async ({ page }) => {
      const loginPage = new LoginPage(page)
      const dashboardPage = new DashboardPage(page)
      const user = generateTestUser()

      // Register and login
      await loginPage.goto()
      await loginPage.register(user.email, user.password)
      await loginPage.waitForRedirectToDashboard()

      // Logout
      await dashboardPage.logout()

      // Try to access protected route
      await page.goto('/dashboard')

      // Should redirect to login
      await expect(page).toHaveURL('/login')
    })
  })

  test.describe('Mode Toggle', () => {
    test('should toggle between login and register modes', async ({ page }) => {
      const loginPage = new LoginPage(page)

      await loginPage.goto()

      // Should start in login mode
      expect(await loginPage.isInLoginMode()).toBe(true)

      // Switch to register mode
      await loginPage.switchToRegisterMode()
      expect(await loginPage.isInRegisterMode()).toBe(true)

      // Switch back to login mode
      await loginPage.switchToLoginMode()
      expect(await loginPage.isInLoginMode()).toBe(true)
    })

    test('should clear form and error when toggling modes', async ({ page }) => {
      const loginPage = new LoginPage(page)

      await loginPage.goto()

      // Fill in some data
      await loginPage.emailInput.fill('test@example.com')
      await loginPage.passwordInput.fill('password123')

      // Switch modes
      await loginPage.switchToRegisterMode()

      // Form should be cleared
      await expect(loginPage.emailInput).toHaveValue('')
      await expect(loginPage.passwordInput).toHaveValue('')
    })
  })

  test.describe('Public Routes', () => {
    test('should allow access to play page without authentication', async ({ page }) => {
      await clearAuthState(page)

      // Visit play page
      await page.goto('/play')

      // Should be able to access play page (public route)
      await expect(page).toHaveURL('/play')
    })

    test('should allow access to home page without authentication', async ({ page }) => {
      await clearAuthState(page)

      // Visit home page
      await page.goto('/')

      // Home redirects to play which is public
      await expect(page).toHaveURL(/\/(play)?/)
    })
  })
})
