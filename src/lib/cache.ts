/**
 * Cache utilities
 *
 * Provides reusable cache implementations with consistent behavior.
 */

/**
 * A simple LRU (Least Recently Used) cache implementation.
 *
 * When the cache reaches its maximum size, the least recently used
 * entries are evicted to make room for new ones.
 *
 * @example
 * const cache = new LRUCache<string, number>(100)
 * cache.set('key', 42)
 * const value = cache.get('key') // 42
 */
export class LRUCache<K, V> {
  private cache: Map<K, V>
  private maxSize: number

  /**
   * Creates a new LRU cache.
   *
   * @param maxSize - Maximum number of entries to store (default: 1000)
   */
  constructor(maxSize = 1000) {
    this.cache = new Map()
    this.maxSize = maxSize
  }

  /**
   * Gets a value from the cache.
   * Accessing a key moves it to the "most recently used" position.
   *
   * @param key - The key to look up
   * @returns The cached value, or undefined if not found
   */
  get(key: K): V | undefined {
    if (!this.cache.has(key)) {
      return undefined
    }

    // Move to end (most recently used) by re-inserting
    const value = this.cache.get(key)!
    this.cache.delete(key)
    this.cache.set(key, value)
    return value
  }

  /**
   * Sets a value in the cache.
   * If the cache is at capacity, evicts the least recently used entry.
   *
   * @param key - The key to store
   * @param value - The value to store
   */
  set(key: K, value: V): void {
    // If key exists, delete it first to update its position
    if (this.cache.has(key)) {
      this.cache.delete(key)
    }
    // Evict oldest entry if at capacity
    else if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value
      if (firstKey !== undefined) {
        this.cache.delete(firstKey)
      }
    }

    this.cache.set(key, value)
  }

  /**
   * Checks if a key exists in the cache.
   * Does NOT affect the LRU ordering.
   *
   * @param key - The key to check
   * @returns True if the key exists
   */
  has(key: K): boolean {
    return this.cache.has(key)
  }

  /**
   * Removes a key from the cache.
   *
   * @param key - The key to remove
   * @returns True if the key was removed, false if it didn't exist
   */
  delete(key: K): boolean {
    return this.cache.delete(key)
  }

  /**
   * Clears all entries from the cache.
   */
  clear(): void {
    this.cache.clear()
  }

  /**
   * Returns the current number of entries in the cache.
   */
  get size(): number {
    return this.cache.size
  }

  /**
   * Returns the maximum capacity of the cache.
   */
  get capacity(): number {
    return this.maxSize
  }

  /**
   * Returns all keys in the cache (from oldest to newest).
   */
  keys(): IterableIterator<K> {
    return this.cache.keys()
  }

  /**
   * Returns all values in the cache (from oldest to newest).
   */
  values(): IterableIterator<V> {
    return this.cache.values()
  }

  /**
   * Returns all entries in the cache (from oldest to newest).
   */
  entries(): IterableIterator<[K, V]> {
    return this.cache.entries()
  }

  /**
   * Iterates over all entries in the cache.
   */
  forEach(callback: (value: V, key: K, cache: LRUCache<K, V>) => void): void {
    this.cache.forEach((value, key) => callback(value, key, this))
  }
}

/**
 * A simple cache that can optionally dispose of values when evicted or cleared.
 * Useful for caching objects that need cleanup (like ONNX sessions).
 *
 * @example
 * const cache = new DisposableCache<string, NeuralAgent>(10, (agent) => agent.dispose())
 */
export class DisposableCache<K, V> extends LRUCache<K, V> {
  private disposeCallback?: (value: V) => void

  /**
   * Creates a new disposable cache.
   *
   * @param maxSize - Maximum number of entries to store
   * @param onDispose - Optional callback to run when a value is evicted or removed
   */
  constructor(maxSize = 1000, onDispose?: (value: V) => void) {
    super(maxSize)
    this.disposeCallback = onDispose
  }

  /**
   * Sets a value, disposing of any evicted entry.
   */
  override set(key: K, value: V): void {
    // If we need to evict, dispose of the evicted value
    if (!this.has(key) && this.size >= this.capacity && this.disposeCallback) {
      const firstKey = this.keys().next().value
      if (firstKey !== undefined) {
        const evictedValue = this.get(firstKey)
        if (evictedValue !== undefined) {
          this.disposeCallback(evictedValue)
        }
      }
    }

    super.set(key, value)
  }

  /**
   * Removes a key and disposes of its value.
   */
  override delete(key: K): boolean {
    if (this.disposeCallback && this.has(key)) {
      const value = this.get(key)
      if (value !== undefined) {
        this.disposeCallback(value)
      }
    }
    return super.delete(key)
  }

  /**
   * Clears all entries and disposes of all values.
   */
  override clear(): void {
    if (this.disposeCallback) {
      this.forEach((value) => this.disposeCallback!(value))
    }
    super.clear()
  }
}

/**
 * Creates a memoized function with LRU caching.
 * The function's result is cached based on the stringified arguments.
 *
 * @param fn - The function to memoize
 * @param maxSize - Maximum number of results to cache (default: 100)
 * @returns A memoized version of the function
 *
 * @example
 * const expensiveCalculation = memoize((x: number) => {
 *   console.log('calculating...')
 *   return x * 2
 * })
 *
 * expensiveCalculation(5) // logs 'calculating...', returns 10
 * expensiveCalculation(5) // returns 10 (cached, no log)
 */
export function memoize<Args extends unknown[], R>(
  fn: (...args: Args) => R,
  maxSize = 100
): (...args: Args) => R {
  const cache = new LRUCache<string, R>(maxSize)

  return (...args: Args): R => {
    const key = JSON.stringify(args)
    const cached = cache.get(key)

    if (cached !== undefined) {
      return cached
    }

    const result = fn(...args)
    cache.set(key, result)
    return result
  }
}
