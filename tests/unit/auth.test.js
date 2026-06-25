import { describe, test, expect, beforeEach } from 'vitest'
import { safeCompare, hashPassword } from '../../src/auth.js'

describe('safeCompare', () => {
  test('returns true for identical strings', () => {
    expect(safeCompare('hello', 'hello')).toBe(true)
  })

  test('returns false for different strings', () => {
    expect(safeCompare('hello', 'world')).toBe(false)
  })

  test('returns false for different lengths', () => {
    expect(safeCompare('hello', 'hell')).toBe(false)
  })

  test('returns false for non-string inputs', () => {
    expect(safeCompare(123, 'hello')).toBe(false)
    expect(safeCompare('hello', null)).toBe(false)
    expect(safeCompare(undefined, undefined)).toBe(false)
  })

  test('returns true for empty strings', () => {
    expect(safeCompare('', '')).toBe(true)
  })
})

describe('hashPassword', () => {
  test('returns consistent hash for same input', async () => {
    const hash1 = await hashPassword('test-password')
    const hash2 = await hashPassword('test-password')
    expect(hash1).toBe(hash2)
  })

  test('returns different hash for different input', async () => {
    const hash1 = await hashPassword('password1')
    const hash2 = await hashPassword('password2')
    expect(hash1).not.toBe(hash2)
  })

  test('returns hex string', async () => {
    const hash = await hashPassword('test')
    expect(hash).toMatch(/^[0-9a-f]{64}$/)
  })
})
