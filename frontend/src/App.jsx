import { useEffect, useState } from 'react'
import { BookOpen, Cpu, AlertCircle, RefreshCw } from 'lucide-react'
import SearchBar from './components/SearchBar.jsx'
import SeedCard from './components/SeedCard.jsx'
import BookCard from './components/BookCard.jsx'
import { useRecommendations } from './hooks/useRecommendations.js'

export default function App() {
  const { seed, results, loading, error, engineReady, checkHealth, search, suggest } = useRecommendations()
  const [hasSearched, setHasSearched] = useState(false)

  useEffect(() => {
    checkHealth()
    const interval = setInterval(async () => {
      const ready = await checkHealth()
      if (ready) clearInterval(interval)
    }, 5000)
    return () => clearInterval(interval)
  }, [])

  function handleSearch(query, userId, topN) {
    setHasSearched(true)
    search(query, userId, topN)
  }

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>

      {/* Header */}
      <header style={{
        borderBottom: '1px solid var(--border)',
        background: 'rgba(250,247,242,0.9)',
        backdropFilter: 'blur(8px)',
        padding: '0 24px',
        position: 'sticky', top: 0, zIndex: 100,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        height: 60,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <BookOpen size={20} style={{ color: 'var(--gold)' }} />
          <span style={{
            fontFamily: "'Cormorant Garamond', serif",
            fontSize: '1.35rem', fontWeight: 600, letterSpacing: '-0.01em',
          }}>
            Folio
          </span>
          <span style={{ fontSize: '0.75rem', color: 'var(--muted)', marginLeft: 4 }}>
            Hybrid Recommendations
          </span>
        </div>

        {/* Engine status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.75rem' }}>
          <div style={{
            width: 7, height: 7, borderRadius: '50%',
            background: engineReady === null ? '#aaa' : engineReady ? 'var(--sage)' : '#e07050',
            boxShadow: engineReady ? '0 0 0 2px rgba(74,103,65,0.25)' : 'none',
          }} />
          <span style={{ color: 'var(--muted)' }}>
            {engineReady === null ? 'Connecting…' : engineReady ? 'Engine ready' : 'Engine loading…'}
          </span>
          {engineReady === false && (
            <button onClick={checkHealth} style={{
              background: 'none', border: 'none', cursor: 'pointer',
              color: 'var(--gold)', padding: 0, marginLeft: 2,
            }}>
              <RefreshCw size={12} />
            </button>
          )}
        </div>
      </header>

      {/* Main */}
      <main style={{ flex: 1, padding: '48px 24px', maxWidth: 800, margin: '0 auto', width: '100%' }}>

        {/* Hero */}
        {!hasSearched && (
          <div className="fade-up" style={{ textAlign: 'center', marginBottom: 48 }}>
            <h1 style={{
              fontFamily: "'Cormorant Garamond', serif",
              fontSize: 'clamp(2.2rem, 6vw, 3.5rem)',
              fontWeight: 300, lineHeight: 1.1,
              color: 'var(--ink)', marginBottom: 16,
            }}>
              Discover your<br />
              <em style={{ color: 'var(--gold)', fontWeight: 400 }}>next great read</em>
            </h1>
            <p style={{ fontSize: '1rem', color: 'var(--muted)', maxWidth: 480, margin: '0 auto 36px' }}>
              Hybrid AI recommendations combining semantic embeddings, collaborative filtering,
              and meta-regression — trained on 6 million real GoodBooks ratings.
            </p>
          </div>
        )}

        {/* Search */}
        <div style={{ marginBottom: 36 }}>
          <SearchBar onSearch={handleSearch} onSuggest={suggest} loading={loading} />
        </div>

        {/* Engine not ready warning */}
        {engineReady === false && (
          <div style={{
            display: 'flex', gap: 10, alignItems: 'flex-start',
            background: '#fff8e6', border: '1px solid #f0d080',
            borderRadius: 6, padding: '14px 18px', marginBottom: 24,
            fontSize: '0.85rem', color: '#7a5c00',
          }}>
            <Cpu size={16} style={{ flexShrink: 0, marginTop: 1 }} />
            <div>
              <strong>Engine warming up.</strong> The backend is loading models
              (embeddings + FAISS + SVD). This takes ~2–4 minutes on first start.
              The status indicator will turn green when ready.
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div style={{
            display: 'flex', gap: 10, alignItems: 'flex-start',
            background: '#fff0ee', border: '1px solid #f0c0b0',
            borderRadius: 6, padding: '14px 18px', marginBottom: 24,
            fontSize: '0.85rem', color: 'var(--rust)',
          }}>
            <AlertCircle size={16} style={{ flexShrink: 0, marginTop: 1 }} />
            {error}
          </div>
        )}

        {/* Loading skeleton */}
        {loading && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {[...Array(3)].map((_, i) => (
              <div key={i} style={{
                height: 100, borderRadius: 6,
                background: 'linear-gradient(90deg, var(--parchment) 25%, #e8e2d8 50%, var(--parchment) 75%)',
                backgroundSize: '200% 100%',
                animation: 'shimmer 1.5s infinite',
              }} />
            ))}
          </div>
        )}

        {/* Results */}
        {!loading && seed && (
          <div>
            <SeedCard seed={seed} />
            <div style={{
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              margin: '28px 0 16px',
            }}>
              <h2 style={{
                fontFamily: "'Cormorant Garamond', serif",
                fontSize: '1.1rem', fontWeight: 400, color: 'var(--muted)',
              }}>
                {results.length} recommendations
              </h2>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {results.map((book, i) => (
                <BookCard key={book.book_idx} book={book} index={i} />
              ))}
            </div>
          </div>
        )}

        {/* Empty state */}
        {!loading && hasSearched && !error && results.length === 0 && (
          <div style={{ textAlign: 'center', padding: '40px 0', color: 'var(--muted)' }}>
            <BookOpen size={32} style={{ opacity: 0.3, marginBottom: 12 }} />
            <p>No recommendations found. Try a different title.</p>
          </div>
        )}

        {/* Feature pills (shown before first search) */}
        {!hasSearched && (
          <div className="fade-up" style={{
            display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8,
          }}>
            {[
              'all-MiniLM-L6-v2 embeddings',
              'FAISS IndexFlatIP',
              'SVD Collaborative Filter',
              'Meta-Regression Calibrator',
              'Open Library Cold-Start',
              'Adaptive User Profiling',
            ].map(f => (
              <span key={f} style={{
                fontSize: '0.72rem', padding: '4px 12px',
                background: 'var(--parchment)', border: '1px solid var(--border)',
                borderRadius: 20, color: 'var(--muted)',
              }}>
                {f}
              </span>
            ))}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer style={{
        borderTop: '1px solid var(--border)',
        padding: '16px 24px',
        textAlign: 'center',
        fontSize: '0.75rem', color: 'var(--muted)',
      }}>
        GoodBooks-10k · 10,000 books · ~6M ratings · Hybrid recommendation system
      </footer>
    </div>
  )
}
