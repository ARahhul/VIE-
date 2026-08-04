const DATA_CONSIDERED = [
  {
    title: 'Video Quality Assessment',
    body: 'Every clip is scored on resolution and sharpness (blur variance). Clips below the quality threshold are undistorted (if the device is calibrated), stabilized, and conditionally upscaled before anything else runs.',
  },
  {
    title: 'Object Detection & Tracking',
    body: 'Every vehicle, pedestrian, and two-wheeler is detected frame-by-frame and assigned a persistent track ID across occlusion, giving a full per-object trajectory for the clip.',
  },
  {
    title: 'Kinematics',
    body: 'When a sensor log (OBD/GPS/IMU) is provided, ego-motion is measured directly and fused with vision-derived relative motion for a grounded speed estimate. Without a sensor log, speed is vision-only and explicitly flagged lower confidence.',
  },
  {
    title: 'Event Window',
    body: 'The moment of interest is located from the sensor log\'s impact timestamp when available (a hard, exact trigger), or an optical-flow-residual spike as a fallback.',
  },
  {
    title: 'Narrative Reasoning',
    body: 'A vision-language model describes what is observed in the clip, grounded in the tracking and kinematics data already computed — it reasons over measured numbers, not just pixels.',
  },
  {
    title: 'Claim Verification',
    body: 'Every statement in the generated narrative is mechanically cross-checked against the actual detected tracks and clip timing before it can appear in a report. Anything that can\'t be verified is downgraded to low confidence and flagged, not dropped silently.',
  },
]

const RULES = [
  'No fault or liability determination. The system describes what the footage and sensor data support — it does not assign blame or identify who was responsible.',
  'Every claim is traceable to the exact pipeline stage that produced it.',
  'Uncertainty is a first-class output. Every claim carries an explicit confidence level (High / Medium / Low) — never smoothed into false confidence.',
  'Vision-only estimates (no sensor log available) always carry a wider error margin than sensor-fused ones, and are labeled as such.',
  'Reports are AI-assisted drafts. They are not certified forensic or legal documents.',
]

function Protocols() {
  return (
    <section className="glass-panel protocols-panel reveal-up active">
      <h2>Protocols</h2>
      <p className="panel-subtitle">What the investigation engine considers, and the rules it operates under.</p>

      <h3 className="protocols-subhead">Data Considered</h3>
      <div className="protocols-grid">
        {DATA_CONSIDERED.map((item) => (
          <div className="protocol-item" key={item.title}>
            <h4>{item.title}</h4>
            <p>{item.body}</p>
          </div>
        ))}
      </div>

      <h3 className="protocols-subhead">Rules & Regulations</h3>
      <ul className="protocols-rules">
        {RULES.map((rule) => (
          <li key={rule}>{rule}</li>
        ))}
      </ul>
    </section>
  )
}

export default Protocols
