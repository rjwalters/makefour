import type { Locator, Page } from '@playwright/test'
import { timeouts } from '../fixtures/test-data'

/**
 * Page object for the Games (history) page
 */
export class GamesPage {
  readonly page: Page
  readonly pageTitle: Locator
  readonly gamesList: Locator
  readonly loadMoreButton: Locator
  readonly emptyMessage: Locator
  readonly loadingIndicator: Locator
  readonly errorMessage: Locator
  readonly logoutButton: Locator
  readonly dashboardLink: Locator
  readonly mobileMenuButton: Locator

  constructor(page: Page) {
    this.page = page
    this.pageTitle = page.locator('text=My Games, text=Game History').first()
    this.gamesList = page.locator('.space-y-4, [class*="grid"]').first()
    this.loadMoreButton = page.locator('button:has-text("Load More")')
    this.emptyMessage = page.locator('text=/No games|no games played yet/i')
    this.loadingIndicator = page.locator('text=/Loading/i')
    this.errorMessage = page.locator('text=/Error|Failed/i')
    this.logoutButton = page.locator('button:has-text("Logout")')
    this.dashboardLink = page.locator('a[href="/dashboard"]').first()
    this.mobileMenuButton = page.locator('button[aria-label="Toggle menu"]')
  }

  async goto() {
    await this.page.goto('/games')
    await this.page.waitForLoadState('networkidle')
  }

  /**
   * Get all game cards
   */
  getGameCards(): Locator {
    return this.page.locator('[class*="Card"], .rounded-lg.border').filter({ hasText: /Win|Loss|Draw/i })
  }

  /**
   * Get a specific game card by index (0-based)
   */
  getGameCard(index: number): Locator {
    return this.getGameCards().nth(index)
  }

  /**
   * Click on a game to view replay
   */
  async clickGame(index: number) {
    const gameCard = this.getGameCard(index)
    const replayButton = gameCard.locator('a:has-text("Replay"), button:has-text("Replay"), a:has-text("View")')
    if (await replayButton.isVisible()) {
      await replayButton.click()
    } else {
      // Click the card itself
      await gameCard.click()
    }
    await this.page.waitForURL(/\/replay\//, { timeout: timeouts.navigation })
  }

  /**
   * Get the number of games displayed
   */
  async getGameCount(): Promise<number> {
    const cards = this.getGameCards()
    return await cards.count()
  }

  /**
   * Check if there are any games
   */
  async hasGames(): Promise<boolean> {
    const count = await this.getGameCount()
    return count > 0
  }

  /**
   * Load more games (pagination)
   */
  async loadMore() {
    if (await this.loadMoreButton.isVisible()) {
      await this.loadMoreButton.click()
      await this.page.waitForTimeout(1000)
    }
  }

  /**
   * Navigate back to dashboard
   */
  async navigateToDashboard() {
    const isMobileMenuVisible = await this.mobileMenuButton.isVisible()
    if (isMobileMenuVisible) {
      await this.mobileMenuButton.click()
    }
    await this.dashboardLink.click()
    await this.page.waitForURL('/dashboard', { timeout: timeouts.navigation })
  }

  /**
   * Get game outcome text
   */
  async getGameOutcome(index: number): Promise<string | null> {
    const gameCard = this.getGameCard(index)
    const outcomeText = await gameCard.locator('text=/Win|Loss|Draw/i').first().textContent()
    return outcomeText
  }
}
