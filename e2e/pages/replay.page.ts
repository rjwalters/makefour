import type { Locator, Page } from '@playwright/test'
import { timeouts } from '../fixtures/test-data'

/**
 * Page object for the Replay page
 */
export class ReplayPage {
  readonly page: Page
  readonly pageTitle: Locator
  readonly gameBoard: Locator
  readonly moveCounter: Locator
  readonly loadingIndicator: Locator
  readonly errorMessage: Locator

  // Navigation controls
  readonly prevButton: Locator
  readonly nextButton: Locator
  readonly startButton: Locator
  readonly endButton: Locator

  // Game info
  readonly outcomeIndicator: Locator
  readonly moveQualityIndicator: Locator
  readonly analysisSection: Locator

  // Navigation
  readonly backToGamesLink: Locator
  readonly dashboardLink: Locator
  readonly mobileMenuButton: Locator

  constructor(page: Page) {
    this.page = page
    this.pageTitle = page.locator('text=Game Replay, text=Replay').first()
    this.gameBoard = page.locator('.bg-blue-600, .bg-blue-800').first()
    this.moveCounter = page.locator('text=/Move \\d+|\\d+ \\/ \\d+/i')
    this.loadingIndicator = page.locator('text=/Loading/i')
    this.errorMessage = page.locator('text=/Error|Failed|not found/i')

    // Navigation controls - look for navigation buttons
    this.prevButton = page.locator(
      'button[aria-label*="Previous"], button[aria-label*="Back"], button:has-text("←"), button:has-text("Prev")'
    )
    this.nextButton = page.locator(
      'button[aria-label*="Next"], button[aria-label*="Forward"], button:has-text("→"), button:has-text("Next")'
    )
    this.startButton = page.locator('button[aria-label*="Start"], button:has-text("⏮"), button:has-text("Start")')
    this.endButton = page.locator('button[aria-label*="End"], button:has-text("⏭"), button:has-text("End")')

    // Game info
    this.outcomeIndicator = page.locator('text=/Win|Loss|Draw/i')
    this.moveQualityIndicator = page.locator('[class*="quality"], text=/Excellent|Good|Inaccuracy|Mistake|Blunder/i')
    this.analysisSection = page.locator('text=/Analysis|Quality|Score/i')

    // Navigation
    this.backToGamesLink = page.locator('a[href="/games"]').first()
    this.dashboardLink = page.locator('a[href="/dashboard"]').first()
    this.mobileMenuButton = page.locator('button[aria-label="Toggle menu"]')
  }

  async goto(gameId: string) {
    await this.page.goto(`/replay/${gameId}`)
    await this.page.waitForLoadState('networkidle')
  }

  /**
   * Go to the previous move
   */
  async goToPreviousMove() {
    if (await this.prevButton.first().isEnabled()) {
      await this.prevButton.first().click()
      await this.page.waitForTimeout(100)
    }
  }

  /**
   * Go to the next move
   */
  async goToNextMove() {
    if (await this.nextButton.first().isEnabled()) {
      await this.nextButton.first().click()
      await this.page.waitForTimeout(100)
    }
  }

  /**
   * Go to the start of the game
   */
  async goToStart() {
    if (await this.startButton.first().isVisible()) {
      await this.startButton.first().click()
      await this.page.waitForTimeout(100)
    } else {
      // Navigate using keyboard
      await this.page.keyboard.press('Home')
      await this.page.waitForTimeout(100)
    }
  }

  /**
   * Go to the end of the game
   */
  async goToEnd() {
    if (await this.endButton.first().isVisible()) {
      await this.endButton.first().click()
      await this.page.waitForTimeout(100)
    } else {
      // Navigate using keyboard
      await this.page.keyboard.press('End')
      await this.page.waitForTimeout(100)
    }
  }

  /**
   * Navigate through all moves from start to end
   */
  async playThroughGame() {
    await this.goToStart()

    // Keep clicking next until we can't anymore
    let attempts = 0
    const maxAttempts = 50 // Max moves in a game

    while (attempts < maxAttempts) {
      const nextEnabled = await this.nextButton
        .first()
        .isEnabled()
        .catch(() => false)
      if (!nextEnabled) break

      await this.goToNextMove()
      attempts++
    }
  }

  /**
   * Use keyboard to navigate
   */
  async navigateWithKeyboard(direction: 'left' | 'right' | 'home' | 'end') {
    const keyMap = {
      left: 'ArrowLeft',
      right: 'ArrowRight',
      home: 'Home',
      end: 'End',
    }
    await this.page.keyboard.press(keyMap[direction])
    await this.page.waitForTimeout(100)
  }

  /**
   * Get the current move number
   */
  async getCurrentMoveNumber(): Promise<number | null> {
    const text = await this.moveCounter.textContent().catch(() => null)
    if (!text) return null

    // Try to parse "Move X" or "X / Y" format
    const match = text.match(/(\d+)/)
    return match ? parseInt(match[1], 10) : null
  }

  /**
   * Check if game has pieces on the board
   */
  async hasPiecesOnBoard(): Promise<boolean> {
    const pieces = this.page.locator('.bg-red-500, .bg-yellow-400')
    const count = await pieces.count()
    return count > 0
  }

  /**
   * Navigate back to games list
   */
  async navigateToGames() {
    const isMobileMenuVisible = await this.mobileMenuButton.isVisible()
    if (isMobileMenuVisible) {
      await this.mobileMenuButton.click()
    }

    if (await this.backToGamesLink.isVisible()) {
      await this.backToGamesLink.click()
    } else {
      await this.page.goto('/games')
    }

    await this.page.waitForURL('/games', { timeout: timeouts.navigation })
  }

  /**
   * Navigate to dashboard
   */
  async navigateToDashboard() {
    const isMobileMenuVisible = await this.mobileMenuButton.isVisible()
    if (isMobileMenuVisible) {
      await this.mobileMenuButton.click()
    }
    await this.dashboardLink.click()
    await this.page.waitForURL('/dashboard', { timeout: timeouts.navigation })
  }
}
