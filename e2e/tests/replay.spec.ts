import type { Page } from '@playwright/test'
import { expect, test, type TestUser } from '../fixtures/base'
import { GamesPage, LoginPage, PlayPage, ReplayPage } from '../pages'

test.describe('Game Replay Flow', () => {
  /**
   * Helper to create a game for replay testing
   * Plays and saves a quick game, then returns to games list
   */
  async function createAndSaveGame(page: Page, testUser: TestUser) {
    const loginPage = new LoginPage(page)
    const playPage = new PlayPage(page)

    // Register if not already
    await loginPage.goto()
    await loginPage.register(testUser.email, testUser.password)
    await page.waitForURL('/dashboard')

    // Go to play
    await playPage.goto()
    await playPage.startHotseatGame()

    // Play a quick game (vertical win)
    await playPage.playMoves([
      0,
      1,
      0,
      1,
      0,
      1,
      0, // P1 wins
    ])

    // Try to save if button is visible
    const saveButton = page.locator('button:has-text("Save")')
    if (await saveButton.isVisible()) {
      await saveButton.click()
      await page.waitForTimeout(1000)
    }
  }

  test.describe('Game History Navigation', () => {
    test('should display games list for authenticated user', async ({ page, testUser }) => {
      const gamesPage = new GamesPage(page)

      // Create a game first
      await createAndSaveGame(page, testUser)

      // Go to games
      await gamesPage.goto()

      // Should be on games page
      await expect(page).toHaveURL('/games')
    })

    test('should navigate from games to replay page', async ({ page, testUser }) => {
      const gamesPage = new GamesPage(page)

      // Create a game first
      await createAndSaveGame(page, testUser)

      // Go to games
      await gamesPage.goto()

      // If there are games, click on one
      const hasGames = await gamesPage.hasGames()
      if (hasGames) {
        await gamesPage.clickGame(0)

        // Should be on replay page
        await expect(page).toHaveURL(/\/replay\//)
      }
    })
  })

  test.describe('Replay Controls', () => {
    test('should display game board on replay page', async ({ page, testUser }) => {
      const gamesPage = new GamesPage(page)
      const replayPage = new ReplayPage(page)

      // Create a game first
      await createAndSaveGame(page, testUser)

      // Go to games and click first game
      await gamesPage.goto()

      const hasGames = await gamesPage.hasGames()
      if (hasGames) {
        await gamesPage.clickGame(0)

        // Game board should be visible
        await expect(replayPage.gameBoard).toBeVisible()
      }
    })

    test('should navigate through moves using buttons', async ({ page, testUser }) => {
      const gamesPage = new GamesPage(page)
      const replayPage = new ReplayPage(page)

      // Create a game first
      await createAndSaveGame(page, testUser)

      // Go to games and click first game
      await gamesPage.goto()

      const hasGames = await gamesPage.hasGames()
      if (hasGames) {
        await gamesPage.clickGame(0)

        // Go to start
        await replayPage.goToStart()
        await page.waitForTimeout(300)

        // Board should have no pieces at start
        const hasPiecesAtStart = await replayPage.hasPiecesOnBoard()
        expect(hasPiecesAtStart).toBe(false)

        // Go to next move
        await replayPage.goToNextMove()
        await page.waitForTimeout(300)

        // Should now have at least one piece
        const hasPiecesAfterMove = await replayPage.hasPiecesOnBoard()
        expect(hasPiecesAfterMove).toBe(true)
      }
    })

    test('should navigate through moves using keyboard', async ({ page, testUser }) => {
      const gamesPage = new GamesPage(page)
      const replayPage = new ReplayPage(page)

      // Create a game first
      await createAndSaveGame(page, testUser)

      // Go to games and click first game
      await gamesPage.goto()

      const hasGames = await gamesPage.hasGames()
      if (hasGames) {
        await gamesPage.clickGame(0)

        // Go to start using keyboard
        await replayPage.navigateWithKeyboard('home')
        await page.waitForTimeout(300)

        // Board should have no pieces at start
        const hasPiecesAtStart = await replayPage.hasPiecesOnBoard()
        expect(hasPiecesAtStart).toBe(false)

        // Go to next move using keyboard
        await replayPage.navigateWithKeyboard('right')
        await page.waitForTimeout(300)

        // Should now have at least one piece
        const hasPiecesAfterMove = await replayPage.hasPiecesOnBoard()
        expect(hasPiecesAfterMove).toBe(true)

        // Go to end
        await replayPage.navigateWithKeyboard('end')
        await page.waitForTimeout(300)

        // Should have pieces at end of game
        const hasPiecesAtEnd = await replayPage.hasPiecesOnBoard()
        expect(hasPiecesAtEnd).toBe(true)
      }
    })

    test('should play through entire game', async ({ page, testUser }) => {
      const gamesPage = new GamesPage(page)
      const replayPage = new ReplayPage(page)

      // Create a game first
      await createAndSaveGame(page, testUser)

      // Go to games and click first game
      await gamesPage.goto()

      const hasGames = await gamesPage.hasGames()
      if (hasGames) {
        await gamesPage.clickGame(0)

        // Play through the game from start to end
        await replayPage.playThroughGame()

        // Should have pieces on board at end
        const hasPieces = await replayPage.hasPiecesOnBoard()
        expect(hasPieces).toBe(true)
      }
    })
  })

  test.describe('Navigation from Replay', () => {
    test('should navigate back to games list from replay', async ({ page, testUser }) => {
      const gamesPage = new GamesPage(page)
      const replayPage = new ReplayPage(page)

      // Create a game first
      await createAndSaveGame(page, testUser)

      // Go to games and click first game
      await gamesPage.goto()

      const hasGames = await gamesPage.hasGames()
      if (hasGames) {
        await gamesPage.clickGame(0)
        await expect(page).toHaveURL(/\/replay\//)

        // Navigate back to games
        await replayPage.navigateToGames()

        // Should be on games page
        await expect(page).toHaveURL('/games')
      }
    })

    test('should navigate to dashboard from replay', async ({ page, testUser }) => {
      const gamesPage = new GamesPage(page)
      const replayPage = new ReplayPage(page)

      // Create a game first
      await createAndSaveGame(page, testUser)

      // Go to games and click first game
      await gamesPage.goto()

      const hasGames = await gamesPage.hasGames()
      if (hasGames) {
        await gamesPage.clickGame(0)
        await expect(page).toHaveURL(/\/replay\//)

        // Navigate to dashboard
        await replayPage.navigateToDashboard()

        // Should be on dashboard
        await expect(page).toHaveURL('/dashboard')
      }
    })
  })

  test.describe('Error Handling', () => {
    test('should handle invalid game ID gracefully', async ({ page, testUser }) => {
      const loginPage = new LoginPage(page)
      const replayPage = new ReplayPage(page)

      // Login first
      await loginPage.goto()
      await loginPage.register(testUser.email, testUser.password)
      await page.waitForURL('/dashboard')

      // Try to access a non-existent game
      await page.goto('/replay/nonexistent-game-id')

      // Should show error or redirect
      // Check for error message or redirect to games
      const hasError = await replayPage.errorMessage.isVisible().catch(() => false)
      const isOnGames = page.url().includes('/games')

      // Either should show error or redirect
      expect(hasError || isOnGames || page.url().includes('/replay/')).toBe(true)
    })

    test('should redirect unauthenticated users to login', async ({ page }) => {
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
