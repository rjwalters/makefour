// =============================================================================
// Two-Tier Encryption System
// =============================================================================
// DEK (Data Encryption Key): Random 256-bit key used to encrypt user data
// KEK (Key Encryption Key): Derived from user password, used to encrypt DEK
// This allows password changes without re-encrypting all user data

import { STORAGE_KEY_COACH_SALT } from './storageKeys'

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

// =============================================================================
// Legacy Functions (Deprecated - kept for migration)
// =============================================================================

/**
 * @deprecated Use generateDEK() instead
 * Derive a deterministic user ID from a passphrase
 */
export async function deriveUserId(passphrase: string): Promise<string> {
  const encoder = new TextEncoder()
  const data = encoder.encode(passphrase)

  // Use a fixed salt for user ID derivation (deterministic)
  const fixedSalt = encoder.encode('coach-user-id-salt-v1')

  const keyMaterial = await crypto.subtle.importKey(
    'raw',
    data,
    'PBKDF2',
    false,
    ['deriveBits']
  )

  const derivedBits = await crypto.subtle.deriveBits(
    {
      name: 'PBKDF2',
      salt: fixedSalt,
      iterations: 100000,
      hash: 'SHA-256',
    },
    keyMaterial,
    256
  )

  // Convert to hex string
  const hashArray = Array.from(new Uint8Array(derivedBits))
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('')
}

/**
 * @deprecated Use deriveKEK() instead
 * Derive a cryptographic key from a passphrase using PBKDF2
 */
export async function deriveKey(passphrase: string, salt?: Uint8Array): Promise<CryptoKey> {
  const encoder = new TextEncoder()
  const passphraseKey = await crypto.subtle.importKey(
    'raw',
    encoder.encode(passphrase),
    'PBKDF2',
    false,
    ['deriveBits', 'deriveKey']
  )

  // Use provided salt or get from localStorage, or generate new one
  let actualSalt = salt
  if (!actualSalt) {
    const storedSalt = localStorage.getItem(STORAGE_KEY_COACH_SALT)
    if (storedSalt) {
      actualSalt = new Uint8Array(JSON.parse(storedSalt))
    } else {
      actualSalt = crypto.getRandomValues(new Uint8Array(16))
      localStorage.setItem(STORAGE_KEY_COACH_SALT, JSON.stringify(Array.from(actualSalt)))
    }
  }

  return crypto.subtle.deriveKey(
    {
      name: 'PBKDF2',
      salt: actualSalt,
      iterations: 100000,
      hash: 'SHA-256',
    },
    passphraseKey,
    { name: 'AES-GCM', length: 256 },
    true,
    ['encrypt', 'decrypt']
  )
}

// =============================================================================
// Data Encryption Functions (used with DEK)
// =============================================================================

// Encrypt data with AES-GCM
export async function encryptData(data: string, key: CryptoKey): Promise<string> {
  const encoder = new TextEncoder()
  const iv = crypto.getRandomValues(new Uint8Array(12))

  const encryptedData = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv },
    key,
    encoder.encode(data)
  )

  // Combine IV and encrypted data
  const combined = new Uint8Array(iv.length + encryptedData.byteLength)
  combined.set(iv, 0)
  combined.set(new Uint8Array(encryptedData), iv.length)

  // Convert to base64
  return btoa(String.fromCharCode(...combined))
}

// Decrypt data with AES-GCM
export async function decryptData(encryptedData: string, key: CryptoKey): Promise<string> {
  // Convert from base64
  const combined = Uint8Array.from(atob(encryptedData), c => c.charCodeAt(0))

  // Extract IV and encrypted data
  const iv = combined.slice(0, 12)
  const data = combined.slice(12)

  const decryptedData = await crypto.subtle.decrypt(
    { name: 'AES-GCM', iv },
    key,
    data
  )

  const decoder = new TextDecoder()
  return decoder.decode(decryptedData)
}
