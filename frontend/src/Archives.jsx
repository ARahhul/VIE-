import { useEffect, useState } from 'react'

function formatDate(iso) {
  return new Date(iso).toLocaleString()
}

function Archives() {
  const [incidents, setIncidents] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/incidents')
      .then((res) => res.json())
      .then(setIncidents)
      .catch((err) => setError(String(err)))
  }, [])

  return (
    <section className="glass-panel archives-panel reveal-up active">
      <h2>Archives</h2>
      <p className="panel-subtitle">Every investigation run so far, newest first.</p>

      {error && <p className="error">{error}</p>}
      {incidents === null && !error && <p className="mono">Loading…</p>}
      {incidents?.length === 0 && <p className="mono">No investigations yet.</p>}

      {incidents && incidents.length > 0 && (
        <div className="archives-table-wrap">
          <table className="archives-table">
            <thead>
              <tr>
                <th>Incident</th>
                <th>Clip</th>
                <th>Device</th>
                <th>Tracked</th>
                <th>Status</th>
                <th>Report</th>
              </tr>
            </thead>
            <tbody>
              {incidents.map((row) => (
                <tr key={row.incident_id}>
                  <td className="mono">
                    <div>{row.incident_id.slice(0, 8)}…</div>
                    <div className="dim">{formatDate(row.created_at)}</div>
                  </td>
                  <td>{row.original_filename ?? '—'}</td>
                  <td className="mono">{row.device_id ?? '—'}</td>
                  <td>{row.num_tracks ?? '—'}</td>
                  <td>
                    <span className={`status status-${row.job_status}`}>{row.job_status ?? 'unknown'}</span>
                  </td>
                  <td>
                    {row.report_available ? (
                      <a className="link" href={`/incidents/${row.incident_id}/report`} target="_blank" rel="noreferrer">
                        Download
                      </a>
                    ) : (
                      <span className="dim">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

export default Archives
