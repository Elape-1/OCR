import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
const supabasePublishableKey = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY || import.meta.env.VITE_SUPABASE_ANON_KEY
export const authRequired = ['1', 'true', 'yes', 'on'].includes(
  String(import.meta.env.VITE_AUTH_REQUIRED || 'false').trim().toLowerCase(),
)

export const supabase = supabaseUrl && supabasePublishableKey
  ? createClient(supabaseUrl, supabasePublishableKey)
  : null

const DEVICE_ID_STORAGE_KEY = 'ocr-device-id'

function getDeviceId() {
  let deviceId = window.localStorage.getItem(DEVICE_ID_STORAGE_KEY)
  if (!deviceId) {
    deviceId = typeof crypto?.randomUUID === 'function'
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(36).slice(2)}`
    window.localStorage.setItem(DEVICE_ID_STORAGE_KEY, deviceId)
  }
  return deviceId
}

export async function getAccessToken() {
  if (!supabase) return null
  const { data } = await supabase.auth.getSession()
  return data.session?.access_token || null
}

export async function authenticatedFetch(input, init = {}) {
  const token = await getAccessToken()
  if (authRequired && supabase && !token) {
    throw new Error('Please sign in to continue')
  }
  const headers = new Headers(init.headers || {})
  headers.set('X-Device-ID', getDeviceId())
  if (token) headers.set('Authorization', `Bearer ${token}`)
  return window.fetch(input, { ...init, headers })
}

export { getDeviceId }
