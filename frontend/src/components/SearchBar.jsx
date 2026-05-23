import { useState, useEffect, useRef } from 'react'
import { Search, User, BookOpen, Loader2 } from 'lucide-react'

export default function SearchBar({ onSearch, onSuggest, loading }) {
  const [query, setQuery] = useState('')
  const [userId, setUserId] = useState(1)
  const [topN, setTopN] = useState(5)
  const [suggestions, setSuggestions] = useState([])
  const [showSugg, setShowSugg] = useState(false)
  const debounceRef = useRef(null)
  const wrapRef = useRef(null)

  useEffect(() => {
    function handleClick(e) {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setShowSugg(false)
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  function handleQueryChange(e) {
    const v = e.target.value
    setQuery(v)
    clearTimeout(debounceRef.current)
    if (v.length < 2) { setSuggestions([]); setShowSugg(false); return }
    debounceRef.current = setTimeout(async () => {
      const s = await onSuggest(v)
      setSuggestions(s)
      setShowSugg(s.length > 0)
    }, 280)
  }

  function handleSubmit(e) {
    e.preventDefault()
    if (!query.trim()) return
    setShowSugg(false)
    onSearch(query, userId, topN)
  }

  function pickSuggestion(s) {
    setQuery(s.title)
    setShowSugg(false)
    onSearch(s.title, userId, topN)
  }

  return (
    <div style={{ width: '100%', maxWidth: 680, margin: '0 auto' }} ref={wrapRef}>
      <form onSubmit={handleSubmit}>
        {/* Main search row */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 0,
          background: '#fff',
          border: '1.5px solid var(--border)',
          borderRadius: 4,
          boxShadow: 'var(--shadow)',
          overflow: 'hidden',
        }}>
          <BookOpen size={18} style={{ marginLeft: 16, color: 'var(--gold)', flexShrink: 0 }} />
          <input
            value={query}
            onChange={handleQueryChange}
            onFocus={() => suggestions.length && setShowSugg(true)}
            placeholder="Search by title, author, or genre…"
            style={{
              flex: 1, border: 'none', outline: 'none', padding: '14px 12px',
              fontSize: '1rem', fontFamily: "'DM Sans', sans-serif",
              color: 'var(--ink)', background: 'transparent',
            }}
          />
          <button type="submit" disabled={loading || !query.trim()} style={{
            background: 'var(--gold)', border: 'none', cursor: loading ? 'wait' : 'pointer',
            padding: '0 20px', height: 50, display: 'flex', alignItems: 'center', gap: 8,
            color: 'var(--ink)', fontWeight: 500, fontSize: '0.9rem',
            transition: 'background 0.2s',
            opacity: !query.trim() ? 0.5 : 1,
          }}
          onMouseEnter={e => e.currentTarget.style.background = 'var(--gold-light)'}
          onMouseLeave={e => e.currentTarget.style.background = 'var(--gold)'}
          >
            {loading
              ? <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} />
              : <Search size={16} />}
            {loading ? 'Searching' : 'Discover'}
          </button>
        </div>

        {/* Autocomplete */}
        {showSugg && (
          <div style={{
            background: '#fff', border: '1.5px solid var(--border)',
            borderTop: 'none', borderRadius: '0 0 4px 4px',
            boxShadow: 'var(--shadow)', zIndex: 10, position: 'relative',
          }}>
            {suggestions.map((s, i) => (
              <div key={i} onClick={() => pickSuggestion(s)} style={{
                padding: '10px 16px', cursor: 'pointer',
                borderBottom: i < suggestions.length - 1 ? '1px solid var(--border)' : 'none',
                transition: 'background 0.15s',
              }}
              onMouseEnter={e => e.currentTarget.style.background = 'var(--parchment)'}
              onMouseLeave={e => e.currentTarget.style.background = '#fff'}
              >
                <span style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: '0.95rem' }}>
                  {s.title}
                </span>
                <span style={{ color: 'var(--muted)', fontSize: '0.8rem', marginLeft: 8 }}>
                  {s.authors}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Controls row */}
        <div style={{ display: 'flex', gap: 16, marginTop: 12, alignItems: 'center', flexWrap: 'wrap' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.85rem', color: 'var(--muted)' }}>
            <User size={14} />
            User ID
            <input type="number" min={1} max={53424} value={userId}
              onChange={e => setUserId(Number(e.target.value))}
              style={{
                width: 72, padding: '4px 8px', border: '1px solid var(--border)',
                borderRadius: 3, fontSize: '0.85rem', color: 'var(--ink)',
                fontFamily: "'DM Sans', sans-serif",
              }} />
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.85rem', color: 'var(--muted)' }}>
            Results
            <input type="range" min={3} max={15} value={topN}
              onChange={e => setTopN(Number(e.target.value))}
              style={{ accentColor: 'var(--gold)', width: 90 }} />
            <span style={{ minWidth: 16, fontWeight: 500, color: 'var(--ink)' }}>{topN}</span>
          </label>
        </div>
      </form>
    </div>
  )
}
