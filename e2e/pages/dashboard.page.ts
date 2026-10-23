import { expect, type Locator, type Page } from '@playwright/test'
import { timeouts } from '../fixtures/test-data'

/**
 * Page object for the Dashboard page
 */
export class DashboardPage {
  readonly page: Page
  readonly pageTitle: Locator
  readonly userEmail: Locator
  readonly logoutButton: Locator
  readonly playLink: Locator
  readonly gamesLink: Locator
  readonly leaderboardLink: Locator
  readonly profileLink: Locator
  readonly statsLink: Locator
  readonly rating: Locator
  readonly gamesPlayed: Locator
  readonly wins: Locator
  readonly losses: Locator
  readonly mobileMenuButton: Locator
  readonly userMenuButton: Locator

  constructor(page: Page) {
    this.page = page
    this.pageTitle = page.locator('text=MakeFour').first()
    this.userEmail = page.locator('.text-muted-foreground:has-text("@")')
    this.logoutButton = page.locator('button:has-text("Logout")')
    this.playLink = page.locator('a[href="/play?mode=training"]').first()
    this.gamesLink = page.locator('a[href="/games"]').first()
    this.leaderboardLink = page.locator('a[href="/leaderboard"]').first()
    this.profileLink = page.locator('a[href="/profile"]').first()
    this.statsLink = page.locator('a[href="/stats"]').first()
    this.rating = page.locator('text=Rating').locator('..').locator('.text-3xl')
    this.gamesPlayed = page.locator('text=Games').locator('..').locator('.text-xl').first()
    this.wins = page.locator('text=Wins').locator('..').locator('.text-xl')
    this.losses = page.locator('text=Losses').locator('..').locator('.text-xl')
    this.mobileMenuButton = page.locator('button[aria-label="Toggle menu"]')
    this.userMenuButton = page.locator('header button:has(span.truncate)')
  }

  async goto() {
    await this.page.goto('/profile')
    await this.page.waitForLoadState('networkidle')
  }

  async logout() {
    // Logout from the protected account page so the route must return to login.
    await this.navigateToProfile()
    await this.openUserMenu()
    await this.logoutButton.first().click()
    await this.page.waitForURL('/login', { timeout: timeouts.navigation })
  }

  async openUserMenu() {
    if (await this.profileLink.isVisible()) return
    // Check if mobile menu is needed
    const isMobileMenuVisible = await this.mobileMenuButton.isVisible()
    if (isMobileMenuVisible) {
      await this.mobileMenuButton.click()
    } else {
      await this.userMenuButton.click()
    }
  }

  async navigateToPlay() {
    const isMobileMenuVisible = await this.mobileMenuButton.isVisible()
    if (isMobileMenuVisible) {
      await this.mobileMenuButton.click()
    }
    await this.playLink.click()
    await this.page.waitForURL('/play?mode=training', { timeout: timeouts.navigation })
  }

  async navigateToGames() {
    const isMobileMenuVisible = await this.mobileMenuButton.isVisible()
    if (isMobileMenuVisible) {
      await this.mobileMenuButton.click()
    }
    await this.gamesLink.click()
    await this.page.waitForURL('/games', { timeout: timeouts.navigation })
  }

  async navigateToLeaderboard() {
    const isMobileMenuVisible = await this.mobileMenuButton.isVisible()
    if (isMobileMenuVisible) {
      await this.mobileMenuButton.click()
    }
    await this.leaderboardLink.click()
    await this.page.waitForURL('/leaderboard', { timeout: timeouts.navigation })
  }

  async navigateToProfile() {
    await this.openUserMenu()
    await this.profileLink.click()
    await this.page.waitForURL('/profile', { timeout: timeouts.navigation })
  }

  async expectUserLoggedIn(email?: string) {
    await expect(this.pageTitle).toBeVisible()
    await this.openUserMenu()
    await expect(this.profileLink).toBeVisible()
    if (email) {
      await expect(this.userEmail).toContainText(email)
    }
  }

  async getUserStats(): Promise<{ rating: string; games: string; wins: string; losses: string }> {
    return {
      rating: (await this.rating.textContent()) || '0',
      games: (await this.gamesPlayed.textContent()) || '0',
      wins: (await this.wins.textContent()) || '0',
      losses: (await this.losses.textContent()) || '0',
    }
  }
}
