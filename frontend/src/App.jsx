import { useEffect, useRef, useState } from 'react'
import Archives from './Archives'
import Background3D from './Background3D'
import Protocols from './Protocols'
import { playAlarm } from './alarm'
import './App.css'

const POLL_INTERVAL_MS = 1500

const FEATURES = [
  {
    icon: 'timeline',
    title: 'Neural Trajectory Mapping',
    body: 'Detection, tracking, and speed estimation across every vehicle and pedestrian in the clip.',
  },
  {
    icon: 'data_usage',
    title: 'Sensor Data Fusion',
    body: 'Ego-motion from OBD/GPS/IMU fused with vision-derived relative motion for a grounded speed estimate.',
  },
  {
    icon: 'summarize',
    title: 'Automated Report Generation',
    body: 'A structured, claim-verified investigation report — descriptive only, never adjudicative.',
  },
]

function App() {
  const videoInputRef = useRef(null)
  const sensorLogInputRef = useRef(null)
  const [view, setView] = useState('telemetry')
  const [alarmEnabled, setAlarmEnabled] = useState(true)
  const alarmEnabledRef = useRef(alarmEnabled)
  alarmEnabledRef.current = alarmEnabled
  const [deviceId, setDeviceId] = useState('')
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const [incidentId, setIncidentId] = useState(null)
  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null)
  const [jobError, setJobError] = useState(null)
  const [videoAsset, setVideoAsset] = useState(null)

  // Re-runs on every view change, not just mount. The reveal animation used
  // to add an `active` class via direct querySelectorAll DOM manipulation in
  // a mount-only effect — switching to Archives and back re-mounted the
  // Telemetry elements without that class, and since the effect never re-ran,
  // they stayed stuck at opacity:0 (invisible). Driving it from React state
  // instead means remounted elements always get their visible state back.
  const [revealed, setRevealed] = useState(false)
  useEffect(() => {
    setRevealed(false)
    const timer = setTimeout(() => setRevealed(true), 50)
    return () => clearTimeout(timer)
  }, [view])

  async function pollJob(id) {
    const poll = async () => {
      const res = await fetch(`/jobs/${id}`)
      const job = await res.json()
      setJobStatus(job.status)
      setJobError(job.error)
      if (job.status === 'queued' || job.status === 'running') {
        setTimeout(poll, POLL_INTERVAL_MS)
      } else if (job.status === 'completed' && alarmEnabledRef.current) {
        playAlarm()
      }
    }
    poll()
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setError(null)
    setJobStatus(null)
    setJobError(null)
    setVideoAsset(null)

    const videoFile = videoInputRef.current?.files?.[0]
    if (!videoFile) {
      setError('Choose a clip to upload.')
      return
    }

    const form = new FormData()
    form.append('video', videoFile)
    const sensorLogFile = sensorLogInputRef.current?.files?.[0]
    if (sensorLogFile) form.append('sensor_log', sensorLogFile)
    if (deviceId) form.append('device_id', deviceId)

    setUploading(true)
    try {
      const res = await fetch('/ingest', { method: 'POST', body: form })
      const body = await res.json()
      if (!res.ok) {
        setError(body.detail || 'Ingest failed.')
        return
      }
      setIncidentId(body.incident_id)
      setJobId(body.job.id)
      setJobStatus(body.job.status)
      setVideoAsset(body.video_asset)
      pollJob(body.job.id)
    } catch (err) {
      setError(String(err))
    } finally {
      setUploading(false)
    }
  }

  const reportReady = jobStatus === 'completed'

  return (
    <>
      <Background3D />

      <nav className="topnav">
        <div className="topnav-brand">VIGILNETRA</div>
        <div className="topnav-links">
          <button type="button" className={view === 'telemetry' ? 'active' : ''} onClick={() => setView('telemetry')}>
            Telemetry
          </button>
          <button type="button" className={view === 'archives' ? 'active' : ''} onClick={() => setView('archives')}>
            Archives
          </button>
          <button type="button" className={view === 'protocols' ? 'active' : ''} onClick={() => setView('protocols')}>
            Protocols
          </button>
        </div>
        <div className="topnav-actions">
          <button type="button" className="live-status">
            LIVE_STATUS
          </button>
          <span
            className={`material-symbols-outlined icon-btn${alarmEnabled ? ' icon-btn-active' : ''}`}
            title={alarmEnabled ? 'Alarm on — a sound plays when a report finishes' : 'Alarm off'}
            onClick={() => setAlarmEnabled((v) => !v)}
            role="button"
            tabIndex={0}
          >
            {alarmEnabled ? 'notifications_active' : 'notifications_off'}
          </span>
        </div>
      </nav>

      <main className="page">
        {view === 'telemetry' && (
          <>
        <header className={`hero reveal-up${revealed ? ' active' : ''}`}>
          <h1>VigilNetra Investigation Engine</h1>
          <p>Upload a clip to generate an investigation report.</p>
        </header>

        <section className={`glass-panel upload-panel reveal-up glow-active${revealed ? ' active' : ''}`}>
          <div className="hud-scanline" />
          <form onSubmit={handleSubmit} className="upload-form">
            <div className="field">
              <label>Clip (video)</label>
              <input ref={videoInputRef} type="file" accept="video/*" required />
            </div>
            <div className="field">
              <label>Sensor log (optional, JSON)</label>
              <input ref={sensorLogInputRef} type="file" accept="application/json" />
            </div>
            <div className="field">
              <label>Device ID (optional)</label>
              <input
                type="text"
                value={deviceId}
                onChange={(e) => setDeviceId(e.target.value)}
                placeholder="vn-unit-001"
              />
            </div>
            <button type="submit" disabled={uploading} className="cta-button">
              {uploading ? 'Uploading…' : 'Upload & investigate'}
              <span className="material-symbols-outlined">arrow_forward</span>
            </button>
          </form>
        </section>

        {error && <p className="error reveal-up active">{error}</p>}

        {videoAsset && (
          <section className="glass-panel info-panel reveal-up active">
            <h2>Clip</h2>
            <p className="mono">
              {videoAsset.original_filename} — {videoAsset.width}x{videoAsset.height} @{' '}
              {videoAsset.fps} fps, {videoAsset.duration_s?.toFixed(1)}s
            </p>
          </section>
        )}

        {jobId && (
          <section className="glass-panel info-panel reveal-up active">
            <h2>Job</h2>
            <p className="mono">
              Incident <code>{incidentId}</code>
            </p>
            <p>
              Status: <span className={`status status-${jobStatus}`}>{jobStatus}</span>
            </p>
            {jobError && <p className="error">{jobError}</p>}
            {reportReady && (
              <a className="cta-button" href={`/incidents/${incidentId}/report`} target="_blank" rel="noreferrer">
                Download report (PDF)
                <span className="material-symbols-outlined">arrow_forward</span>
              </a>
            )}
          </section>
        )}

        <section className="features-grid">
          {FEATURES.map((f, i) => (
            <div className={`glass-panel feature-card reveal-up${revealed ? ' active' : ''}`} style={{ transitionDelay: `${i * 100}ms` }} key={f.title}>
              <div className="feature-icon">
                <span className="material-symbols-outlined">{f.icon}</span>
              </div>
              <h3>{f.title}</h3>
              <p>{f.body}</p>
            </div>
          ))}
        </section>
          </>
        )}

        {view === 'archives' && <Archives />}
        {view === 'protocols' && <Protocols />}
      </main>

      <footer className="site-footer">
        <div className="footer-brand">VIGILNETRA</div>
        <div className="footer-copy mono">© 2026 VIGILNETRA FORENSICS. ENCRYPTED_CONNECTION_ESTABLISHED.</div>
        <div className="footer-links">
          <a href="#">Legal_Ops</a>
          <a href="#">API_Docs</a>
          <a href="#">Security_Core</a>
        </div>
      </footer>
    </>
  )
}

export default App
