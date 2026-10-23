import { expect, test } from '../fixtures/base'
import { LoginPage } from '../pages'

function gate() {
  let release = () => {}
  const ready = new Promise<void>(resolve => { release = resolve })
  return { ready, release }
}

test.describe('Protected page session restoration', () => {
  test('waits for a valid saved session before showing games', async ({ page, testUser }) => {
    const loginPage = new LoginPage(page)
    await loginPage.goto()
    await loginPage.register(testUser.email, testUser.password)
    await loginPage.waitForRedirectToTraining()

    const requested = gate()
    const response = gate()
    await page.route('**/api/auth/me', async route => {
      requested.release()
      await response.ready
      await route.continue()
    })

    await page.goto('/games')
    await requested.ready
    try {
      await expect(page.getByRole('status')).toHaveText('Loading session...')
      await expect(page).toHaveURL('/games')
      await expect(page.getByRole('heading', { name: 'My Games' })).not.toBeVisible()
    } finally {
      response.release()
    }

    await expect(page.getByRole('heading', { name: 'My Games' })).toBeVisible()
    await expect(page).toHaveURL('/games')
  })

  for (const failure of ['invalid', 'unreachable'] as const) {
    test(`denies access after an ${failure} saved session check`, async ({ page }) => {
      await page.addInitScript(() => {
        localStorage.setItem('makefour_session_token', 'invalid-e2e-session')
      })
      const requested = gate()
      const response = gate()
      await page.route('**/api/auth/me', async route => {
        requested.release()
        await response.ready
        if (failure === 'invalid') {
          await route.fulfill({ status: 401, json: { error: 'Invalid session' } })
        } else {
          await route.abort('failed')
        }
      })

      await page.goto('/games')
      await requested.ready
      try {
        await expect(page.getByRole('status')).toHaveText('Loading session...')
        await expect(page).toHaveURL('/games')
        await expect(page.getByRole('heading', { name: 'My Games' })).not.toBeVisible()
      } finally {
        response.release()
      }

      await expect(page).toHaveURL('/login')
      await expect(page.getByRole('heading', { name: 'My Games' })).not.toBeVisible()
      expect(await page.evaluate(() => localStorage.getItem('makefour_session_token'))).toBeNull()
    })
  }
})
