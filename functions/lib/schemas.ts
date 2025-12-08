import { z } from 'zod'

// User schemas
export const userSchema = z.object({
  id: z.string().uuid(),
  email: z.string().email(),
  email_verified: z.number().int().min(0).max(1), // SQLite boolean
  password_hash: z.string().nullable(),
  oauth_provider: z.enum(['google']).nullable(),
  oauth_id: z.string().nullable(),
  encrypted_dek: z.string().nullable(),
  created_at: z.number(),
  last_login: z.number(),
  updated_at: z.number(),
})

export type User = z.infer<typeof userSchema>

export const publicUserSchema = userSchema.omit({
  password_hash: true,
  oauth_id: true,
  encrypted_dek: true,
})

export type PublicUser = z.infer<typeof publicUserSchema>

// Registration schemas
export const registerRequestSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z
    .string()
    .min(8, 'Password must be at least 8 characters')
    .max(128, 'Password must be less than 128 characters'),
})

export type RegisterRequest = z.infer<typeof registerRequestSchema>

export const registerResponseSchema = z.object({
  user: z.object({
    id: z.string(),
    email: z.string(),
    email_verified: z.union([z.number(), z.boolean()]),
    created_at: z.number(),
    last_login: z.number(),
    updated_at: z.number(),
  }),
  message: z.string(),
})

export type RegisterResponse = z.infer<typeof registerResponseSchema>

// Login schemas
export const loginRequestSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string().min(1, 'Password is required'),
})

export type LoginRequest = z.infer<typeof loginRequestSchema>

export const loginResponseSchema = z.object({
  user: z.object({
    id: z.string(),
    email: z.string(),
    email_verified: z.boolean(),
    created_at: z.number(),
    last_login: z.number(),
    updated_at: z.number(),
  }),
  session_token: z.string().uuid(),
  encrypted_dek: z.string().nullable(),
})

export type LoginResponse = z.infer<typeof loginResponseSchema>

// Session token schema
export const sessionTokenSchema = z.object({
  id: z.string().uuid(),
  user_id: z.string().uuid(),
  expires_at: z.number(),
  created_at: z.number(),
})

export type SessionToken = z.infer<typeof sessionTokenSchema>

// Error response schema
export const errorResponseSchema = z.object({
  error: z.string(),
  details: z.string().optional(),
})

export type ErrorResponse = z.infer<typeof errorResponseSchema>

// Validation helpers
export function validateRegisterRequest(data: unknown): RegisterRequest {
  return registerRequestSchema.parse(data)
}

export function validateLoginRequest(data: unknown): LoginRequest {
  return loginRequestSchema.parse(data)
}

// Helper to format validation errors
export function formatZodError(error: z.ZodError): ErrorResponse {
  return {
    error: 'Validation error',
    details: error.errors[0].message,
  }
}

// AI usage schemas
export const aiUsageSchema = z.object({
  id: z.string().uuid(),
  user_id: z.string().uuid(),
  model: z.string(),
  prompt_tokens: z.number().int().min(0),
  completion_tokens: z.number().int().min(0),
  total_tokens: z.number().int().min(0),
  endpoint: z.string(),
  created_at: z.number(),
})

export type AIUsage = z.infer<typeof aiUsageSchema>

export const createAIUsageSchema = z.object({
  model: z.string(),
  prompt_tokens: z.number().int().min(0),
  completion_tokens: z.number().int().min(0),
  total_tokens: z.number().int().min(0),
  endpoint: z.string(),
})

export type CreateAIUsage = z.infer<typeof createAIUsageSchema>

export const aiUsageStatsSchema = z.object({
  total_requests: z.number().int(),
  total_prompt_tokens: z.number().int(),
  total_completion_tokens: z.number().int(),
  total_tokens: z.number().int(),
  by_model: z.array(z.object({
    model: z.string(),
    requests: z.number().int(),
    prompt_tokens: z.number().int(),
    completion_tokens: z.number().int(),
    total_tokens: z.number().int(),
  })),
  by_endpoint: z.array(z.object({
    endpoint: z.string(),
    requests: z.number().int(),
    prompt_tokens: z.number().int(),
    completion_tokens: z.number().int(),
    total_tokens: z.number().int(),
  })),
})

export type AIUsageStats = z.infer<typeof aiUsageStatsSchema>
