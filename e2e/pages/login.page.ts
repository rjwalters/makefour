import { expect, type Locator, type Page } from '@playwright/test'
import { timeouts } from '../fixtures/test-data'

/**
 * Page object for the Login/Register page
 */
export class LoginPage {
  readonly page: Page
  readonly emailInput: Locator
  readonly passwordInput: Locator
  readonly submitButton: Locator
  readonly toggleModeButton: Locator
  readonly googleButton: Locator
  readonly errorMessage: Locator
  readonly pageTitle: Locator

  constructor(page: Page) {
    this.page = page
    this.emailInput = page.locator('input#email')
    this.passwordInput = page.locator('input#password')
    this.submitButton = page.locator('button[type="submit"]')
    this.toggleModeButton = page.locator('button:has-text("New user?"), button:has-text("Already have an account?")')
    this.googleButton = page.locator('button:has-text("Continue with Google")')
    this.errorMessage = page.locator('.text-destructive')
    this.pageTitle = page.locator('text=Welcome to MakeFour')
  }

  async goto() {
    await this.page.goto('/login')
    await this.pageTitle.waitFor({ state: 'visible' })
  }

  async login(email: string, password: string) {
    await this.emailInput.fill(email)
    await this.passwordInput.fill(password)
    await this.submitButton.click()
  }

  async register(email: string, password: string) {
    // Switch to registration mode
    await this.toggleModeButton.click()
    await this.emailInput.fill(email)
    await this.passwordInput.fill(password)
    await this.submitButton.click()
  }

  async switchToRegisterMode() {
    const button = this.page.locator('button:has-text("New user?")')
    if (await button.isVisible()) {
      await button.click()
    }
  }

  async switchToLoginMode() {
    const button = this.page.locator('button:has-text("Already have an account?")')
    if (await button.isVisible()) {
      await button.click()
    }
  }

  async waitForRedirectToDashboard() {
    await this.page.waitForURL('/dashboard', { timeout: timeouts.navigation })
  }

  async expectError(errorText?: string) {
    await expect(this.errorMessage).toBeVisible()
    if (errorText) {
      await expect(this.errorMessage).toContainText(errorText)
    }
  }

  async isInLoginMode(): Promise<boolean> {
    return await this.submitButton.textContent().then((text) => text?.includes('Login') ?? false)
  }

  async isInRegisterMode(): Promise<boolean> {
    return await this.submitButton.textContent().then((text) => text?.includes('Create Account') ?? false)
  }
}
