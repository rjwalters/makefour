/**
 * AI Engines Index
 *
 * Registers all available AI engines with the global registry.
 * Import this module to ensure engines are registered before use.
 */

import { engineRegistry } from '../ai-engine'
import { minimaxEngine } from './minimax-engine'

// Register the minimax engine
engineRegistry.register(minimaxEngine, true)

// Set minimax as the default engine
engineRegistry.setDefault('minimax')

// Re-export engines for direct access if needed
export { minimaxEngine }

// Re-export registry utilities
export { engineRegistry }
