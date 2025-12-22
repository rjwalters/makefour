/**
 * Neural Engine Tests
 */

import { describe, it, expect, beforeEach } from 'vitest'
import {
  NeuralEngine,
  DEFAULT_NEURAL_CONFIG,
  MODEL_REGISTRY,
  getModelMetadata,
  listModels,
  clearModelCache,
  createNeuralEngine,
} from './neural-engine'
import { createEmptyBoard, applyMove, type Board, type Player } from '../game'

describe('NeuralEngine', () => {
  let engine: NeuralEngine

  beforeEach(() => {
    engine = new NeuralEngine()
  })

  describe('initialization', () => {
    it('should create engine with default config', () => {
      expect(engine.name).toBe('neural')
      expect(engine.description).toBe('Neural network inference with optional hybrid search')
    })

    it('should be ready in simulated mode', () => {
      expect(engine.isReady()).toBe(true)
    })

    it('should accept custom config', () => {
      const customEngine = new NeuralEngine({
        temperature: 0.8,
        useHybridSearch: false,
        hybridDepth: 5,
      })
      expect(customEngine.isReady()).toBe(true)
    })
  })

  describe('selectMove', () => {
    it('should select a valid move on empty board', async () => {
      const board = createEmptyBoard()
      const result = await engine.selectMove(board, 1, { searchDepth: 3, errorRate: 0 }, 1000)

      expect(result.column).toBeGreaterThanOrEqual(0)
      expect(result.column).toBeLessThan(7)
      expect(board[0][result.column]).toBeNull() // Column should have been valid
    })

    it('should return the only valid move when one column left', async () => {
      // Create board with only column 3 open
      const board = createEmptyBoard()
      for (let col = 0; col < 7; col++) {
        if (col !== 3) {
          for (let row = 0; row < 6; row++) {
            board[row][col] = (row + col) % 2 === 0 ? 1 : 2
          }
        }
      }

      const result = await engine.selectMove(board, 1, { searchDepth: 3, errorRate: 0 }, 1000)
      expect(result.column).toBe(3)
      expect(result.confidence).toBe(1)
    })

    it('should find winning moves', async () => {
      const board = createEmptyBoard()
      // Set up a winning position: Player 1 has 3 in a row at bottom
      board[5][0] = 1
      board[5][1] = 1
      board[5][2] = 1
      // Player 2 has some pieces
      board[5][5] = 2
      board[5][6] = 2

      // Use greedy mode (temperature 0) to ensure we pick the best move
      const greedyEngine = new NeuralEngine({ temperature: 0, useHybridSearch: true })
      const result = await greedyEngine.selectMove(board, 1, { searchDepth: 3, errorRate: 0 }, 1000)

      // Should play column 3 to complete the four
      expect(result.column).toBe(3)
    })

    it('should block opponent winning moves', async () => {
      const board = createEmptyBoard()
      // Set up opponent (Player 2) about to win
      board[5][0] = 2
      board[5][1] = 2
      board[5][2] = 2
      // Player 1 has some pieces elsewhere
      board[5][5] = 1
      board[5][6] = 1

      const greedyEngine = new NeuralEngine({ temperature: 0, useHybridSearch: true })
      const result = await greedyEngine.selectMove(board, 1, { searchDepth: 3, errorRate: 0 }, 1000)

      // Should block at column 3
      expect(result.column).toBe(3)
    })

    it('should respect time budget', async () => {
      const board = createEmptyBoard()
      const startTime = Date.now()
      const timeBudget = 100 // 100ms

      await engine.selectMove(board, 1, { searchDepth: 10, errorRate: 0 }, timeBudget)

      const elapsed = Date.now() - startTime
      // Should complete within reasonable time (allow some buffer)
      expect(elapsed).toBeLessThan(timeBudget + 50)
    })

    it('should apply error rate', async () => {
      const board = createEmptyBoard()
      // Set up a clear winning position
      board[5][0] = 1
      board[5][1] = 1
      board[5][2] = 1

      // With 100% error rate, should pick random moves
      const results = new Set<number>()
      for (let i = 0; i < 20; i++) {
        const result = await engine.selectMove(
          board,
          1,
          { searchDepth: 3, errorRate: 1.0 },
          100
        )
        results.add(result.column)
      }

      // With 100% error rate and random selection, should see variety
      expect(results.size).toBeGreaterThan(1)
    })

    it('should return search info', async () => {
      const board = createEmptyBoard()
      const result = await engine.selectMove(board, 1, { searchDepth: 3, errorRate: 0 }, 1000)

      expect(result.searchInfo).toBeDefined()
      expect(result.searchInfo?.timeUsed).toBeGreaterThanOrEqual(0)
    })
  })

  describe('hybrid search mode', () => {
    it('should use hybrid search when enabled', async () => {
      const board = createEmptyBoard()
      const hybridEngine = new NeuralEngine({
        useHybridSearch: true,
        hybridDepth: 3,
        temperature: 0,
      })

      const result = await hybridEngine.selectMove(
        board,
        1,
        { searchDepth: 3, errorRate: 0 },
        1000
      )

      // Hybrid mode should report depth in search info
      expect(result.searchInfo?.depth).toBe(3)
    })

    it('should work in pure policy mode', async () => {
      const board = createEmptyBoard()
      const policyEngine = new NeuralEngine({
        useHybridSearch: false,
        temperature: 0,
      })

      const result = await policyEngine.selectMove(
        board,
        1,
        { searchDepth: 1, errorRate: 0 },
        1000
      )

      // Pure policy mode should report depth 0
      expect(result.searchInfo?.depth).toBe(0)
    })
  })

  describe('temperature control', () => {
    it('should be deterministic with temperature 0', async () => {
      const board = createEmptyBoard()
      const deterministicEngine = new NeuralEngine({
        temperature: 0,
        useHybridSearch: false,
      })

      const results = []
      for (let i = 0; i < 5; i++) {
        const result = await deterministicEngine.selectMove(
          board,
          1,
          { searchDepth: 1, errorRate: 0 },
          100
        )
        results.push(result.column)
      }

      // All results should be the same with temperature 0
      expect(new Set(results).size).toBe(1)
    })

    it('should add variety with higher temperature', async () => {
      const board = createEmptyBoard()
      const randomEngine = new NeuralEngine({
        temperature: 2.0, // High temperature for more randomness
        useHybridSearch: false,
      })

      const results = new Set<number>()
      for (let i = 0; i < 50; i++) {
        const result = await randomEngine.selectMove(
          board,
          1,
          { searchDepth: 1, errorRate: 0 },
          100
        )
        results.add(result.column)
      }

      // With high temperature, should see some variety
      expect(results.size).toBeGreaterThan(1)
    })
  })

  describe('evaluatePosition', () => {
    it('should return positive score for advantageous position', () => {
      const board = createEmptyBoard()
      // Player 1 controls center with 3-in-a-row
      board[5][2] = 1
      board[5][3] = 1
      board[5][4] = 1
      // Player 2 has pieces on edges
      board[5][0] = 2
      board[5][6] = 2

      const score = engine.evaluatePosition(board, 1)
      expect(score).toBeGreaterThan(0)
    })

    it('should return negative score for disadvantageous position', () => {
      const board = createEmptyBoard()
      // Player 2 has a strong center position
      board[5][2] = 2
      board[5][3] = 2
      board[5][4] = 2
      // Player 1 has weak position
      board[5][0] = 1
      board[5][6] = 1

      const score = engine.evaluatePosition(board, 1)
      expect(score).toBeLessThan(0)
    })
  })

  describe('explainMove', () => {
    it('should explain winning move', () => {
      const board = createEmptyBoard()
      board[5][0] = 1
      board[5][1] = 1
      board[5][2] = 1

      const explanation = engine.explainMove(board, 3, 1)
      expect(explanation).toContain('winning')
    })

    it('should explain blocking move', () => {
      const board = createEmptyBoard()
      board[5][0] = 2
      board[5][1] = 2
      board[5][2] = 2

      const explanation = engine.explainMove(board, 3, 1)
      expect(explanation).toContain('blocking')
    })

    it('should explain center move', () => {
      const board = createEmptyBoard()

      const explanation = engine.explainMove(board, 3, 1)
      expect(explanation).toContain('center')
    })
  })

  describe('DEFAULT_NEURAL_CONFIG', () => {
    it('should have sensible defaults', () => {
      expect(DEFAULT_NEURAL_CONFIG.modelPath).toBeNull()
      expect(DEFAULT_NEURAL_CONFIG.modelId).toBeNull()
      expect(DEFAULT_NEURAL_CONFIG.temperature).toBeGreaterThan(0)
      expect(DEFAULT_NEURAL_CONFIG.temperature).toBeLessThan(2)
      expect(DEFAULT_NEURAL_CONFIG.useHybridSearch).toBe(true)
      expect(DEFAULT_NEURAL_CONFIG.hybridDepth).toBeGreaterThanOrEqual(1)
    })
  })

  describe('modelId configuration', () => {
    it('should create engine with modelId from registry', () => {
      const engine = new NeuralEngine({ modelId: 'heuristic-v1' })
      expect(engine.isReady()).toBe(true)
      expect(engine.getModelMetadata()).not.toBeNull()
      expect(engine.getModelMetadata()?.id).toBe('heuristic-v1')
    })

    it('should fallback to simulated inference for unknown modelId', () => {
      const engine = new NeuralEngine({ modelId: 'unknown-model-xyz' })
      expect(engine.isReady()).toBe(true)
      expect(engine.getModelMetadata()).toBeNull()
    })

    it('should prefer modelId over deprecated modelPath', () => {
      const engine = new NeuralEngine({
        modelId: 'heuristic-v1',
        modelPath: '/some/path.onnx',
      })
      expect(engine.getModelMetadata()?.id).toBe('heuristic-v1')
    })
  })
})

describe('Model Registry', () => {
  describe('MODEL_REGISTRY', () => {
    it('should contain at least the heuristic baseline model', () => {
      expect(MODEL_REGISTRY.length).toBeGreaterThanOrEqual(1)
      const heuristic = MODEL_REGISTRY.find((m) => m.id === 'heuristic-v1')
      expect(heuristic).toBeDefined()
      expect(heuristic?.name).toBe('Heuristic Baseline')
    })

    it('should have valid model metadata for all models', () => {
      for (const model of MODEL_REGISTRY) {
        expect(model.id).toBeTruthy()
        expect(model.name).toBeTruthy()
        expect(model.architecture).toMatch(/^(mlp|cnn|resnet|transformer)$/)
        expect(model.expectedElo).toBeGreaterThan(0)
        expect(model.version).toBeTruthy()
        expect(model.encoding).toMatch(/^(onehot-6x7x3|bitboard|flat-binary)$/)
      }
    })
  })

  describe('getModelMetadata', () => {
    it('should return metadata for existing model', () => {
      const metadata = getModelMetadata('heuristic-v1')
      expect(metadata).not.toBeNull()
      expect(metadata?.id).toBe('heuristic-v1')
    })

    it('should return null for non-existing model', () => {
      const metadata = getModelMetadata('non-existent-model')
      expect(metadata).toBeNull()
    })
  })

  describe('listModels', () => {
    it('should return a copy of the registry', () => {
      const models1 = listModels()
      const models2 = listModels()
      expect(models1).toEqual(models2)
      expect(models1).not.toBe(models2) // Different references
    })
  })

  describe('clearModelCache', () => {
    it('should clear without error', () => {
      expect(() => clearModelCache()).not.toThrow()
    })
  })
})

describe('createNeuralEngine factory', () => {
  it('should create engine with specified model', async () => {
    const engine = await createNeuralEngine('heuristic-v1')
    expect(engine.isReady()).toBe(true)
    expect(engine.getModelMetadata()?.id).toBe('heuristic-v1')
  })

  it('should select valid moves', async () => {
    const engine = await createNeuralEngine('heuristic-v1')
    const board = createEmptyBoard()
    const result = await engine.selectMove(board, 1, { searchDepth: 3, errorRate: 0 }, 1000)

    expect(result.column).toBeGreaterThanOrEqual(0)
    expect(result.column).toBeLessThan(7)
  })
})
