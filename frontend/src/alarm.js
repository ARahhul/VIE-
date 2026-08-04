// A short synthesized two-tone chime via the Web Audio API — no audio
// asset to bundle/license, works everywhere. Played when a report finishes.
export function playAlarm() {
  const Ctx = window.AudioContext || window.webkitAudioContext
  if (!Ctx) return

  const ctx = new Ctx()
  const now = ctx.currentTime

  const tones = [
    { freq: 880, start: 0, duration: 0.15 },
    { freq: 1174.66, start: 0.16, duration: 0.25 },
  ]

  tones.forEach(({ freq, start, duration }) => {
    const osc = ctx.createOscillator()
    const gain = ctx.createGain()
    osc.type = 'sine'
    osc.frequency.value = freq
    gain.gain.setValueAtTime(0, now + start)
    gain.gain.linearRampToValueAtTime(0.2, now + start + 0.02)
    gain.gain.exponentialRampToValueAtTime(0.001, now + start + duration)
    osc.connect(gain)
    gain.connect(ctx.destination)
    osc.start(now + start)
    osc.stop(now + start + duration + 0.05)
  })

  setTimeout(() => ctx.close(), 700)
}
