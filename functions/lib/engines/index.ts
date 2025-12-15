/**
 * AI Engines Index
 *
 * Registers all available AI engines with the global registry.
 * Import this module to ensure engines are registered before use.
 */

import { engineRegistry } from '../ai-engine'
import { minimaxEngine } from './minimax-engine'
import { aggressiveMinimaxEngine } from './aggressive-minimax-engine'
import { deepMinimaxEngine } from './deep-minimax-engine'

// Register all minimax variants
engineRegistry.register(minimaxEngine, true)
engineRegistry.register(aggressiveMinimaxEngine, true)
engineRegistry.register(deepMinimaxEngine, true)

// Set standard minimax as the default engine
engineRegistry.setDefault('minimax')

// Re-export engines for direct access if needed
export { minimaxEngine, aggressiveMinimaxEngine, deepMinimaxEngine }

// Re-export registry utilities
export { engineRegistry }
