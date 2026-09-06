import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
const supabasePublishableKey = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY || import.meta.env.VITE_SUPABASE_ANON_KEY
export const authRequired = ['1', 'true', 'yes', 'on'].includes(
  String(import.meta.env.VITE_AUTH_REQUIRED || 'false').trim().toLowerCase(),
)

export const supabase = supabaseUrl && supabasePublishableKey
  ? createClient(supabaseUrl, supabasePublishableKey)
  : null

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
  if (token) headers.set('Authorization', `Bearer ${token}`)
  return window.fetch(input, { ...init, headers })
}
