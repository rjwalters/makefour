// Backend crypto utilities for Cloudflare Workers
// Uses Web Crypto API (available in Workers)

/**
 * Generate a random Data Encryption Key (DEK)
 * This key will be used to encrypt/decrypt user data
 */
export async function generateDEK(): Promise<CryptoKey> {
  return crypto.subtle.generateKey(
    { name: 'AES-GCM', length: 256 },
    true, // extractable
    ['encrypt', 'decrypt']
  )
}

/**
 * Derive a Key Encryption Key (KEK) from a password using PBKDF2
 * The KEK is used to encrypt/decrypt the DEK
 */
export async function deriveKEK(password: string, salt: Uint8Array): Promise<CryptoKey> {
  const encoder = new TextEncoder()
  const passwordKey = await crypto.subtle.importKey(
    'raw',
    encoder.encode(password),
    'PBKDF2',
    false,
    ['deriveKey']
  )

  return crypto.subtle.deriveKey(
    {
      name: 'PBKDF2',
      salt: salt,
      iterations: 100000,
      hash: 'SHA-256',
    },
    passwordKey,
    { name: 'AES-GCM', length: 256 },
    false, // not extractable for security
    ['encrypt', 'decrypt']
  )
}

/**
 * Encrypt a DEK with a password-derived KEK
 * Returns: base64-encoded JSON with salt, iv, and encrypted DEK
 */
export async function encryptDEK(dek: CryptoKey, password: string): Promise<string> {
  // Generate random salt for KEK derivation
  const salt = crypto.getRandomValues(new Uint8Array(16))

  // Derive KEK from password
  const kek = await deriveKEK(password, salt)

  // Export DEK as raw bytes
  const dekBytes = await crypto.subtle.exportKey('raw', dek)

  // Generate random IV for AES-GCM
  const iv = crypto.getRandomValues(new Uint8Array(12))

  // Encrypt DEK with KEK
  const encryptedDEK = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv },
    kek,
    dekBytes
  )

  // Package everything together
  const result = {
    salt: Array.from(salt),
    iv: Array.from(iv),
    encryptedDEK: Array.from(new Uint8Array(encryptedDEK))
  }

  // Encode as base64 JSON string
  return btoa(JSON.stringify(result))
}

/**
 * Decrypt a DEK using a password-derived KEK
 * Takes base64-encoded JSON with salt, iv, and encrypted DEK
 * Returns the decrypted DEK as a CryptoKey
 */
export async function decryptDEK(encryptedDEKString: string, password: string): Promise<CryptoKey> {
  // Decode from base64 and parse JSON
  const { salt, iv, encryptedDEK } = JSON.parse(atob(encryptedDEKString))

  // Convert arrays back to Uint8Array
  const saltBytes = new Uint8Array(salt)
  const ivBytes = new Uint8Array(iv)
  const encryptedDEKBytes = new Uint8Array(encryptedDEK)

  // Derive KEK from password using stored salt
  const kek = await deriveKEK(password, saltBytes)

  // Decrypt DEK
  const dekBytes = await crypto.subtle.decrypt(
    { name: 'AES-GCM', iv: ivBytes },
    kek,
    encryptedDEKBytes
  )

  // Import decrypted DEK as a CryptoKey
  return crypto.subtle.importKey(
    'raw',
    dekBytes,
    { name: 'AES-GCM', length: 256 },
    true,
    ['encrypt', 'decrypt']
  )
}
