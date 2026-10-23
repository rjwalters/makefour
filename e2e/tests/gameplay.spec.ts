import { expect, test } from '../fixtures/base'
import { DashboardPage, LoginPage, PlayPage } from '../pages'

test.describe('Gameplay Flow', () => {
  test.describe('Game Setup', () => {
    test('should start a training game from setup', async ({ page, testUser }) => {
      const loginPage = new LoginPage(page)
      const playPage = new PlayPage(page)

      // Register and navigate to play
      await loginPage.goto()
      await loginPage.register(testUser.email, testUser.password)
      await page.waitForURL('/play?mode=training')
      await playPage.goto()

      // Start training game (vs AI)
      await playPage.startHotseatGame()

      // Board should be visible
      await expect(playPage.gameBoard).toBeVisible()
    })

    test('should allow access to play page without login (guest mode)', async ({ page }) => {
      const playPage = new PlayPage(page)

      await playPage.goto()

      // Should be able to see the play page with training mode
      await expect(page).toHaveURL(/\/play/)
    })

    test('should show training mode setup options', async ({ page, testUser }) => {
      const loginPage = new LoginPage(page)
      const playPage = new PlayPage(page)

      await loginPage.goto()
      await loginPage.register(testUser.email, testUser.password)
      await page.waitForURL('/play?mode=training')
      await playPage.goto()

      // Should see Training Mode title
      await expect(page.locator('text=Training Mode')).toBeVisible()

      // Should see difficulty options
      await expect(page.locator('text=Difficulty')).toBeVisible()

      // Should see start button
      await expect(page.locator('button:has-text("Start Training")')).toBeVisible()
    })
  })

  test.describe('Making Moves', () => {
    test('should drop a piece when clicking a column', async ({ page, testUser }) => {
      const loginPage = new LoginPage(page)
      const playPage = new PlayPage(page)

      // Setup
      await loginPage.goto()
      await loginPage.register(testUser.email, testUser.password)
      await page.waitForURL('/play?mode=training')
      await playPage.goto()
      await playPage.startHotseatGame()

      // Make a move
      await playPage.dropPieceInColumn(3) // Middle column

      // Board should now have a piece
      const hasPieces = await playPage.hasPieces()
      expect(hasPieces).toBe(true)
    })

    test('should show player pieces after moves', async ({ page, testUser }) => {
      const loginPage = new LoginPage(page)
      const playPage = new PlayPage(page)

      // Setup
      await loginPage.goto()
      await loginPage.register(testUser.email, testUser.password)
      await page.waitForURL('/play?mode=training')
      await playPage.goto()
      await playPage.startHotseatGame()

      // Make a move
      await playPage.dropPieceInColumn(3)

      // Wait for AI to respond
      await page.waitForTimeout(1000)

      // Should have at least one red piece (player's move)
      const redPieces = page.locator('.bg-red-500')
      expect(await redPieces.count()).toBeGreaterThan(0)
    })
  })

  test.describe('Game UI Elements', () => {
    test('should show game status during play', async ({ page, testUser }) => {
      const loginPage = new LoginPage(page)
      const playPage = new PlayPage(page)

      // Setup
      await loginPage.goto()
      await loginPage.register(testUser.email, testUser.password)
      await page.waitForURL('/play?mode=training')
      await playPage.goto()
      await playPage.startHotseatGame()

      // Should show player vs AI header
      await expect(page.getByText('You', { exact: true })).toBeVisible()
      await expect(page.locator('text=AI')).toBeVisible()
    })

    test('should have New Game button during play', async ({ page, testUser }) => {
      const loginPage = new LoginPage(page)
      const playPage = new PlayPage(page)

      // Setup
      await loginPage.goto()
      await loginPage.register(testUser.email, testUser.password)
      await page.waitForURL('/play?mode=training')
      await playPage.goto()
      await playPage.startHotseatGame()

      // Make a move to enable the New Game button
      await playPage.dropPieceInColumn(3)
      await page.waitForTimeout(500)

      // Should show New Game button
      await expect(page.locator('button:has-text("New Game")')).toBeVisible()
    })
  })

  test.describe('New Game', () => {
    test('should return to setup when clicking New Game', async ({ page, testUser }) => {
      const loginPage = new LoginPage(page)
      const playPage = new PlayPage(page)

      // Setup and play some moves
      await loginPage.goto()
      await loginPage.register(testUser.email, testUser.password)
      await page.waitForURL('/play?mode=training')
      await playPage.goto()
      await playPage.startHotseatGame()

      // Make a move
      await playPage.dropPieceInColumn(0)
      await page.waitForTimeout(500)

      // Click New Game
      await playPage.clickNewGame()
      await page.waitForTimeout(500)

      // Should see setup screen again
      const setupVisible = await page.locator('text=Training Mode').isVisible()
      const startVisible = await page.locator('button:has-text("Start Training")').isVisible()
      expect(setupVisible || startVisible).toBe(true)
    })
  })

  test.describe('Navigation', () => {
    test('should navigate from play to profile', async ({ page, testUser }) => {
      const loginPage = new LoginPage(page)
      const playPage = new PlayPage(page)

      // Setup
      await loginPage.goto()
      await loginPage.register(testUser.email, testUser.password)
      await page.waitForURL('/play?mode=training')
      await playPage.goto()

      // Navigate to the protected account page through the user menu
      await new DashboardPage(page).navigateToProfile()

      // Should be on profile
      await expect(page).toHaveURL('/profile')
    })

    test('should navigate from profile to play', async ({ page, testUser }) => {
      const loginPage = new LoginPage(page)
      const dashboardPage = new DashboardPage(page)

      // Setup
      await loginPage.goto()
      await loginPage.register(testUser.email, testUser.password)
      await page.waitForURL('/play?mode=training')

      // Navigate from the protected account page to training
      await dashboardPage.navigateToProfile()
      await dashboardPage.navigateToPlay()

      // Should be on play page
      await expect(page).toHaveURL(/\/play/)
    })
  })
})
