import { useState } from 'react'
import { BookOpen, Star } from 'lucide-react'

export default function SeedCard({ seed }) {
  const [imgError, setImgError] = useState(false)

  return (
    <div className="fade-up" style={{
      background: 'var(--ink)', color: '#fff',
      borderRadius: 6, padding: '20px 24px',
      display: 'flex', gap: 20, alignItems: 'center',
      boxShadow: 'var(--shadow-lg)',
      border: '1px solid rgba(201,168,76,0.3)',
    }}>
      {/* Cover */}
      <div style={{
        width: 64, height: 90, flexShrink: 0,
        background: 'rgba(255,255,255,0.08)',
        borderRadius: 3, overflow: 'hidden',
        border: '1px solid rgba(201,168,76,0.3)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        {seed.cover_url && !imgError ? (
          <img src={seed.cover_url} alt={seed.title}
            onError={() => setImgError(true)}
            style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
        ) : (
          <BookOpen size={22} style={{ color: 'var(--gold)' }} />
        )}
      </div>

      <div>
        <p style={{ fontSize: '0.7rem', letterSpacing: '0.12em', color: 'var(--gold)', marginBottom: 4, textTransform: 'uppercase' }}>
          Based on
        </p>
        <h2 style={{
          fontFamily: "'Cormorant Garamond', serif",
          fontSize: '1.4rem', fontWeight: 600, lineHeight: 1.2,
          color: '#fff', marginBottom: 4,
        }}>
          {seed.title}
        </h2>
        <p style={{ fontSize: '0.85rem', color: 'rgba(255,255,255,0.6)', marginBottom: 6 }}>
          {seed.authors}
        </p>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          {seed.average_rating && (
            <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '0.8rem', color: 'var(--gold-light)' }}>
              <Star size={12} fill="var(--gold)" stroke="var(--gold)" />
              {seed.average_rating.toFixed(1)}
            </span>
          )}
          <span style={{
            fontSize: '0.68rem', padding: '2px 8px',
            background: 'rgba(201,168,76,0.15)',
            border: '1px solid rgba(201,168,76,0.3)',
            color: 'var(--gold-light)', borderRadius: 20,
          }}>
            {seed.source === 'LOCAL' ? 'GoodBooks Catalogue' : 'Open Library (Cold-Start)'}
          </span>
        </div>
      </div>
    </div>
  )
}
