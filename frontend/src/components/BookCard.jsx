import { useState } from 'react'
import { Star, BookOpen, TrendingUp, Brain, Layers } from 'lucide-react'

function ScoreBar({ label, value, color, icon: Icon }) {
  if (value == null) return null
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 3 }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '0.72rem', color: 'var(--muted)' }}>
          <Icon size={10} /> {label}
        </span>
        <span style={{ fontSize: '0.72rem', fontWeight: 500, color: 'var(--ink)' }}>
          {(value * 100).toFixed(0)}%
        </span>
      </div>
      <div style={{ height: 3, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${value * 100}%`,
          background: color, borderRadius: 2,
          transition: 'width 0.8s ease',
        }} />
      </div>
    </div>
  )
}

export default function BookCard({ book, index }) {
  const [imgError, setImgError] = useState(false)
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="fade-up" style={{
      animationDelay: `${index * 60}ms`,
      display: 'flex', gap: 16,
      background: '#fff',
      border: '1px solid var(--border)',
      borderRadius: 6,
      padding: '18px 20px',
      boxShadow: 'var(--shadow)',
      transition: 'transform 0.2s, box-shadow 0.2s',
      cursor: 'default',
    }}
    onMouseEnter={e => {
      e.currentTarget.style.transform = 'translateY(-2px)'
      e.currentTarget.style.boxShadow = 'var(--shadow-lg)'
    }}
    onMouseLeave={e => {
      e.currentTarget.style.transform = 'translateY(0)'
      e.currentTarget.style.boxShadow = 'var(--shadow)'
    }}>

      {/* Rank */}
      <div style={{
        fontFamily: "'Cormorant Garamond', serif",
        fontSize: '1.8rem', fontWeight: 300, color: 'var(--border)',
        minWidth: 28, lineHeight: 1, paddingTop: 2,
      }}>
        {String(index + 1).padStart(2, '0')}
      </div>

      {/* Cover */}
      <div style={{
        width: 56, height: 80, flexShrink: 0,
        background: 'var(--parchment)',
        borderRadius: 3, overflow: 'hidden',
        border: '1px solid var(--border)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        {book.cover_url && !imgError ? (
          <img src={book.cover_url} alt={book.title}
            onError={() => setImgError(true)}
            style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
        ) : (
          <BookOpen size={20} style={{ color: 'var(--muted)' }} />
        )}
      </div>

      {/* Content */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
          <div>
            <h3 style={{
              fontFamily: "'Cormorant Garamond', serif",
              fontSize: '1.05rem', fontWeight: 600, lineHeight: 1.3,
              color: 'var(--ink)', marginBottom: 2,
            }}>
              {book.title}
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--muted)', marginBottom: 6 }}>
              {book.authors}
              {book.publication_year && <span style={{ marginLeft: 6 }}>· {book.publication_year}</span>}
            </p>
          </div>

          {/* Final score badge */}
          <div style={{
            background: 'var(--ink)', color: 'var(--gold)',
            borderRadius: 3, padding: '4px 10px',
            fontSize: '0.75rem', fontWeight: 500, flexShrink: 0,
            fontFamily: "'DM Sans', sans-serif",
          }}>
            {(book.final_score * 100).toFixed(0)}
          </div>
        </div>

        {/* Meta row */}
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 8, flexWrap: 'wrap' }}>
          {book.average_rating && (
            <span style={{ display: 'flex', alignItems: 'center', gap: 3, fontSize: '0.78rem', color: 'var(--muted)' }}>
              <Star size={11} fill="var(--gold)" stroke="var(--gold)" />
              {book.average_rating.toFixed(1)}
            </span>
          )}
          <span style={{
            fontSize: '0.68rem', padding: '2px 7px',
            background: book.source === 'LOCAL' ? 'var(--parchment)' : '#edf3ec',
            color: book.source === 'LOCAL' ? 'var(--muted)' : 'var(--sage)',
            borderRadius: 20, border: `1px solid ${book.source === 'LOCAL' ? 'var(--border)' : '#b4d4b0'}`,
          }}>
            {book.source === 'LOCAL' ? 'GoodBooks' : 'Open Library'}
          </span>
        </div>

        {/* Score breakdown toggle */}
        <button onClick={() => setExpanded(v => !v)} style={{
          background: 'none', border: 'none', cursor: 'pointer',
          fontSize: '0.75rem', color: 'var(--gold)', padding: 0,
          display: 'flex', alignItems: 'center', gap: 4,
          marginBottom: expanded ? 10 : 0,
        }}>
          <Layers size={11} /> {expanded ? 'Hide' : 'Show'} score breakdown
        </button>

        {expanded && (
          <div style={{ marginTop: 8 }}>
            <ScoreBar label="Content similarity" value={book.content_score} color="var(--gold)" icon={Brain} />
            <ScoreBar label="Collaborative filter" value={book.collab_score} color="var(--rust)" icon={TrendingUp} />
            <ScoreBar label="Final hybrid" value={book.final_score} color="var(--sage)" icon={Layers} />
            <p style={{ fontSize: '0.7rem', color: 'var(--muted)', marginTop: 6, fontStyle: 'italic' }}>
              {book.explanation}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
