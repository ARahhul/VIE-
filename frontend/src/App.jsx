import { useRef, useState } from 'react'
import './App.css'

const POLL_INTERVAL_MS = 1500

function App() {
  const videoInputRef = useRef(null)
  const sensorLogInputRef = useRef(null)
  const [deviceId, setDeviceId] = useState('')
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const [incidentId, setIncidentId] = useState(null)
  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null)
  const [jobError, setJobError] = useState(null)
  const [videoAsset, setVideoAsset] = useState(null)

  async function pollJob(id) {
    const poll = async () => {
      const res = await fetch(`/jobs/${id}`)
      const job = await res.json()
      setJobStatus(job.status)
      setJobError(job.error)
      if (job.status === 'queued' || job.status === 'running') {
        setTimeout(poll, POLL_INTERVAL_MS)
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
    <div className="page">
      <h1>VigilNetra Investigation Engine</h1>
      <p className="subtitle">Upload a clip to generate an investigation report.</p>

      <form onSubmit={handleSubmit} className="upload-form">
        <label>
          Clip (video)
          <input ref={videoInputRef} type="file" accept="video/*" required />
        </label>
        <label>
          Sensor log (optional, JSON)
          <input ref={sensorLogInputRef} type="file" accept="application/json" />
        </label>
        <label>
          Device ID (optional)
          <input
            type="text"
            value={deviceId}
            onChange={(e) => setDeviceId(e.target.value)}
            placeholder="vn-unit-001"
          />
        </label>
        <button type="submit" disabled={uploading}>
          {uploading ? 'Uploading…' : 'Upload & investigate'}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {videoAsset && (
        <div className="card">
          <h2>Clip</h2>
          <p>
            {videoAsset.original_filename} — {videoAsset.width}x{videoAsset.height} @{' '}
            {videoAsset.fps} fps, {videoAsset.duration_s?.toFixed(1)}s
          </p>
        </div>
      )}

      {jobId && (
        <div className="card">
          <h2>Job</h2>
          <p>
            Incident <code>{incidentId}</code>
          </p>
          <p>
            Status: <span className={`status status-${jobStatus}`}>{jobStatus}</span>
          </p>
          {jobError && <p className="error">{jobError}</p>}
          {reportReady && (
            <a className="button" href={`/incidents/${incidentId}/report`} target="_blank" rel="noreferrer">
              Download report (PDF)
            </a>
          )}
        </div>
      )}
    </div>
  )
}

export default App
