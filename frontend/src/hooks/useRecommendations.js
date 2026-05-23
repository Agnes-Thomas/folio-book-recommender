import { useState, useCallback, useRef } from 'react'

const API = import.meta.env.VITE_API_URL || ''

export function useRecommendations() {
  const [seed, setSeed] = useState(null)
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [engineReady, setEngineReady] = useState(null)
  const abortRef = useRef(null)

  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API}/health`)
      const data = await res.json()
      setEngineReady(data.ready)
      return data.ready
    } catch {
      setEngineReady(false)
      return false
    }
  }, [])

  const search = useCallback(async (query, userId = 1, topN = 5) => {
    if (abortRef.current) abortRef.current.abort()
    abortRef.current = new AbortController()

    setLoading(true)
    setError(null)
    setSeed(null)
    setResults([])

    try {
      const res = await fetch(`${API}/api/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, user_id: userId, top_n: topN }),
        signal: abortRef.current.signal,
      })
      if (res.status === 503) {
        setError('The recommendation engine is still warming up. Please try again in a moment.')
        return
      }
      if (!res.ok) {
        const err = await res.json()
        setError(err.detail || 'Something went wrong.')
        return
      }
      const data = await res.json()
      setSeed(data.seed)
      setResults(data.recommendations)
    } catch (e) {
      if (e.name !== 'AbortError') setError('Could not reach the server. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }, [])

  const suggest = useCallback(async (q) => {
    if (!q || q.length < 2) return []
    try {
      const res = await fetch(`${API}/api/search?q=${encodeURIComponent(q)}&limit=6`)
      if (!res.ok) return []
      return await res.json()
    } catch {
      return []
    }
  }, [])

  return { seed, results, loading, error, engineReady, checkHealth, search, suggest }
}
