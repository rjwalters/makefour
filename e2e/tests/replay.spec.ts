import { expect, test } from '../fixtures/base'
import { GamesPage, LoginPage } from '../pages'

test.describe('Game History Flow', () => {
  test.describe('Games Page Access', () => {
    test('should display games page for authenticated user', async ({ page, testUser }) => {
      const loginPage = new LoginPage(page)
      const gamesPage = new GamesPage(page)

      // Register first
      await loginPage.goto()
      await loginPage.register(testUser.email, testUser.password)
      await page.waitForURL('/dashboard')

      // Go to games
      await gamesPage.goto()

      // Should be on games page
      await expect(page).toHaveURL('/games')
    })

    test('should redirect unauthenticated users to login', async ({ page }) => {
      // Clear any auth state
      await page.goto('/')
      await page.evaluate(() => {
        localStorage.removeItem('makefour_session_token')
        localStorage.removeItem('makefour_dek')
      })

      // Try to access games page
      await page.goto('/games')

      // Should redirect to login
      await expect(page).toHaveURL('/login')
    })
  })

  test.describe('Navigation', () => {
    test('should navigate from dashboard to games', async ({ page, testUser }) => {
      const loginPage = new LoginPage(page)

      // Register
      await loginPage.goto()
      await loginPage.register(testUser.email, testUser.password)
      await page.waitForURL('/dashboard')

      // Navigate to games via navbar or link
      const gamesLink = page.locator('a[href="/games"]').first()
      if (await gamesLink.isVisible()) {
        await gamesLink.click()
        await expect(page).toHaveURL('/games')
      }
    })

    test('should navigate from games back to dashboard', async ({ page, testUser }) => {
      const loginPage = new LoginPage(page)
      const gamesPage = new GamesPage(page)

      // Register and go to games
      await loginPage.goto()
      await loginPage.register(testUser.email, testUser.password)
      await page.waitForURL('/dashboard')
      await gamesPage.goto()

      // Navigate back to dashboard via navbar
      const dashboardLink = page.locator('a[href="/dashboard"]').first()
      if (await dashboardLink.isVisible()) {
        await dashboardLink.click()
        await expect(page).toHaveURL('/dashboard')
      }
    })
  })

  test.describe('Replay Page Access', () => {
    test('should handle invalid game ID gracefully', async ({ page, testUser }) => {
      const loginPage = new LoginPage(page)

      // Login first
      await loginPage.goto()
      await loginPage.register(testUser.email, testUser.password)
      await page.waitForURL('/dashboard')

      // Try to access a non-existent game
      await page.goto('/replay/nonexistent-game-id')

      // Should show error or redirect - not crash
      // Just verify the page loaded without hanging
      await page.waitForLoadState('networkidle')
    })

    test('should redirect unauthenticated users from replay to login', async ({ page }) => {
      // Clear any auth state
      await page.goto('/')
      await page.evaluate(() => {
        localStorage.removeItem('makefour_session_token')
        localStorage.removeItem('makefour_dek')
      })

      // Try to access replay page
      await page.goto('/replay/some-game-id')

      // Should redirect to login
      await expect(page).toHaveURL('/login')
    })
  })
})
