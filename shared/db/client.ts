import { drizzle } from 'drizzle-orm/d1'
import * as schema from './schema'

// D1Database type from Cloudflare Workers
type D1DatabaseType = {
  prepare(query: string): D1PreparedStatement
  dump(): Promise<ArrayBuffer>
  batch<T = unknown>(statements: D1PreparedStatement[]): Promise<D1Result<T>[]>
  exec(query: string): Promise<D1ExecResult>
}

interface D1PreparedStatement {
  bind(...values: unknown[]): D1PreparedStatement
  first<T = unknown>(colName?: string): Promise<T | null>
  run<T = unknown>(): Promise<D1Result<T>>
  all<T = unknown>(): Promise<D1Result<T>>
  raw<T = unknown>(): Promise<T[]>
}

interface D1Result<T = unknown> {
  results?: T[]
  success: boolean
  error?: string
  meta: {
    duration: number
    changes: number
    last_row_id: number
    served_by: string
    internal_stats: unknown
  }
}

interface D1ExecResult {
  count: number
  duration: number
}

/**
 * Create a Drizzle database client from a D1 database instance.
 * Use this in Cloudflare Workers/Pages Functions.
 *
 * @example
 * ```typescript
 * export async function onRequestGet(context: EventContext<Env, any, any>) {
 *   const db = createDb(context.env.DB)
 *   const user = await db.query.users.findFirst({
 *     where: eq(users.id, userId)
 *   })
 * }
 * ```
 */
export function createDb(d1: D1DatabaseType) {
  return drizzle(d1 as Parameters<typeof drizzle>[0], { schema })
}

export type Database = ReturnType<typeof createDb>
