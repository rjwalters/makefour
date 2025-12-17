/**
 * Playwright test to validate chat functionality
 *
 * Tests:
 * 1. Login flow
 * 2. Game creation
 * 3. Chat connection
 * 4. Message sending and receiving
 * 5. Duplicate detection
 * 6. Multiple message handling
 * 7. Bot reactions
 */

import { chromium } from 'playwright'

const BASE_URL = process.env.TEST_URL || 'https://dc4d76ba.makefour.pages.dev'

const TESTS = {
  passed: 0,
  failed: 0,
  results: []
}

function test(name, passed, details = '') {
  if (passed) {
    TESTS.passed++
    console.log(`  ✓ ${name}`)
  } else {
    TESTS.failed++
    console.log(`  ✗ ${name}${details ? ': ' + details : ''}`)
  }
  TESTS.results.push({ name, passed, details })
}

async function validateChat() {
  console.log(`\n🧪 Chat Validation Test\n`)
  console.log(`URL: ${BASE_URL}/debug/chat\n`)

  const browser = await chromium.launch({ headless: true })
  const context = await browser.newContext()
  const page = await context.newPage()

  try {
    // =========================================================================
    // 1. PAGE LOAD
    // =========================================================================
    console.log('1. Page Load')

    await page.goto(`${BASE_URL}/debug/chat`, { waitUntil: 'networkidle', timeout: 30000 })

    test('Page loads without redirect', page.url().includes('/debug/chat'))

    const pageRendered = await page.locator('[data-testid="debug-chat-page"]').count() > 0
    test('Debug chat page renders', pageRendered)

    // =========================================================================
    // 2. AUTHENTICATION
    // =========================================================================
    console.log('\n2. Authentication')

    const loginBtn = page.locator('[data-testid="debug-login-btn"]')
    const needsLogin = await loginBtn.count() > 0

    if (needsLogin) {
      await loginBtn.click()
      await page.waitForLoadState('networkidle')
      await page.waitForSelector('[data-testid="debug-chat-page"]', { timeout: 10000 })
    }

    // Wait for auth state
    await page.waitForFunction(() => {
      const el = document.querySelector('[data-testid="auth-status"]')
      return el && el.textContent.includes('YES')
    }, { timeout: 10000 })

    const authStatus = await page.locator('[data-testid="auth-status"]').textContent()
    test('User authenticated', authStatus.includes('YES'))
    test('User has username', authStatus.includes('DebugTester'))

    // =========================================================================
    // 3. GAME CREATION
    // =========================================================================
    console.log('\n3. Game Creation')

    await page.locator('[data-testid="create-game-btn"]').click()

    await page.waitForFunction(() => {
      const input = document.querySelector('[data-testid="game-id-input"]')
      return input && input.value && input.value.length > 30
    }, { timeout: 10000 })

    const gameId = await page.locator('[data-testid="game-id-input"]').inputValue()
    test('Game ID generated', gameId.length === 36, gameId) // UUID length

    // =========================================================================
    // 4. CHAT CONNECTION
    // =========================================================================
    console.log('\n4. Chat Connection')

    await page.locator('[data-testid="connect-btn"]').click()
    await page.waitForTimeout(500)

    const connectionStatus = await page.locator('[data-testid="connection-status"]').textContent()
    test('Chat connected', connectionStatus.includes('ACTIVE'))
    test('Connected to correct game', connectionStatus.includes(gameId.slice(0, 8)))

    // =========================================================================
    // 5. SEND SINGLE MESSAGE
    // =========================================================================
    console.log('\n5. Single Message')

    const testMessage1 = `Test message ${Date.now()}`
    await page.locator('[data-testid="message-input"]').fill(testMessage1)
    await page.locator('[data-testid="send-btn"]').click()

    // Wait for message to appear
    await page.waitForFunction((msg) => {
      const el = document.querySelector('#messages-json')
      if (!el) return false
      try {
        const messages = JSON.parse(el.value)
        return messages.some(m => m.content === msg)
      } catch { return false }
    }, testMessage1, { timeout: 5000 })

    let messages = JSON.parse(await page.locator('#messages-json').inputValue())
    test('Message sent and received', messages.some(m => m.content === testMessage1))
    test('Message has correct sender type', messages.find(m => m.content === testMessage1)?.sender_type === 'human')
    test('Message has unique ID', messages.find(m => m.content === testMessage1)?.id?.length > 0)

    // =========================================================================
    // 6. SEND MULTIPLE MESSAGES
    // =========================================================================
    console.log('\n6. Multiple Messages')

    const testMessages = [
      `Multi test A ${Date.now()}`,
      `Multi test B ${Date.now()}`,
      `Multi test C ${Date.now()}`,
    ]

    for (const msg of testMessages) {
      await page.locator('[data-testid="message-input"]').fill(msg)
      await page.locator('[data-testid="send-btn"]').click()
      await page.waitForTimeout(500) // Small delay between messages
    }

    // Wait for all messages
    await page.waitForTimeout(2000)

    messages = JSON.parse(await page.locator('#messages-json').inputValue())
    const allMessagesReceived = testMessages.every(msg =>
      messages.some(m => m.content === msg)
    )
    test('All messages received', allMessagesReceived, `Got ${messages.length} messages`)

    // =========================================================================
    // 7. DUPLICATE DETECTION
    // =========================================================================
    console.log('\n7. Duplicate Detection')

    const ids = messages.map(m => m.id)
    const uniqueIds = new Set(ids)
    const hasDuplicates = ids.length !== uniqueIds.size

    test('No duplicate message IDs', !hasDuplicates,
      hasDuplicates ? `Found ${ids.length - uniqueIds.size} duplicates` : '')

    // Check debug log for duplicate warnings
    const debugLog = await page.locator('#debug-log').inputValue()
    const hasDuplicateWarning = debugLog.includes('DUPLICATE')
    test('No duplicate warnings in log', !hasDuplicateWarning)

    // =========================================================================
    // 8. MESSAGE ORDER
    // =========================================================================
    console.log('\n8. Message Order')

    const timestamps = messages.map(m => m.created_at)
    const isSorted = timestamps.every((t, i) => i === 0 || t >= timestamps[i - 1])
    test('Messages in chronological order', isSorted)

    // =========================================================================
    // 9. BOT REACTION
    // =========================================================================
    console.log('\n9. Bot Reaction')

    const messageCountBefore = messages.length
    await page.locator('[data-testid="trigger-bot-btn"]').click()
    await page.waitForTimeout(3000) // Wait for bot response

    const messagesAfter = JSON.parse(await page.locator('#messages-json').inputValue())
    const botMessages = messagesAfter.filter(m => m.sender_type === 'bot')

    // Bot may or may not respond depending on game state
    test('Bot reaction triggered without error', !debugLog.includes('[ERROR] Chat error'))
    console.log(`     (Bot messages: ${botMessages.length})`)

    // =========================================================================
    // 10. QUICK REACTIONS
    // =========================================================================
    console.log('\n10. Quick Reactions')

    await page.locator('[data-testid="quick-gg"]').click()
    const inputValue = await page.locator('[data-testid="message-input"]').inputValue()
    test('Quick reaction fills input', inputValue === 'Good game!')

    // =========================================================================
    // 11. MUTE TOGGLE
    // =========================================================================
    console.log('\n11. Mute Toggle')

    await page.locator('[data-testid="mute-btn"]').click()
    await page.waitForTimeout(500)

    const chatStatus = await page.locator('[data-testid="chat-status"]').textContent()
    test('Mute toggles correctly', chatStatus.includes('muted=true'))

    // Unmute
    await page.locator('[data-testid="mute-btn"]').click()

    // =========================================================================
    // RESULTS
    // =========================================================================
    console.log('\n' + '='.repeat(50))
    console.log(`\n📊 Results: ${TESTS.passed} passed, ${TESTS.failed} failed\n`)

    if (TESTS.failed > 0) {
      console.log('Failed tests:')
      TESTS.results.filter(r => !r.passed).forEach(r => {
        console.log(`  - ${r.name}${r.details ? ': ' + r.details : ''}`)
      })
      console.log('')
    }

    // Screenshot
    await page.screenshot({ path: '/tmp/chat-validation.png', fullPage: true })
    console.log('Screenshot saved to /tmp/chat-validation.png\n')

    process.exit(TESTS.failed > 0 ? 1 : 0)

  } catch (error) {
    console.error('\n❌ Test Error:', error.message)

    try {
      const debugLog = await page.locator('#debug-log').inputValue()
      console.log('\nDebug log:')
      console.log(debugLog)
    } catch {}

    await page.screenshot({ path: '/tmp/chat-validation-error.png', fullPage: true })
    console.log('\nError screenshot saved to /tmp/chat-validation-error.png')
    process.exit(1)
  } finally {
    await browser.close()
  }
}

validateChat()
