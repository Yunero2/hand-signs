import React, { useRef, useEffect, useState } from 'react'
import * as handPoseDetection from '@tensorflow-models/hand-pose-detection'
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-webgl'

// Hand connections for 21 landmarks (MediaPipe ordering)
const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4], // Thumb
  [0,5],[5,6],[6,7],[7,8], // Index
  [0,9],[9,10],[10,11],[11,12], // Middle
  [0,13],[13,14],[14,15],[15,16], // Ring
  [0,17],[17,18],[18,19],[19,20], // Pinky
  [5,9],[9,13],[13,17] // Palm cross-connections
]

function euclid(a, b) {
  const dx = a.x - b.x
  const dy = a.y - b.y
  return Math.hypot(dx, dy)
}

// Simple heuristic sign detection. Returns {name, confidence}
// Heuristic gesture detection helper.
// Accepts `landmarks` array (21 points) where each point is {x,y[,z]} in pixel coordinates.
// Returns an object { name, confidence } with confidence 0-100.
export function detectHandSign(landmarks) {
  if (!landmarks || landmarks.length < 21) return { name: 'Unknown', confidence: 0 }

  // Compute bounding box and use its size to normalize distances so heuristics work across sizes.
  const xs = landmarks.map(p => p.x)
  const ys = landmarks.map(p => p.y)
  const w = Math.max(...xs) - Math.min(...xs)
  const h = Math.max(...ys) - Math.min(...ys)
  const handSize = Math.max(w, h) || 1
  const wrist = landmarks[0]

  // fingertip indices
  const tips = { thumb: 4, index: 8, middle: 12, ring: 16, pinky: 20 }
  const tipDists = {
    thumb: euclid(landmarks[tips.thumb], wrist),
    index: euclid(landmarks[tips.index], wrist),
    middle: euclid(landmarks[tips.middle], wrist),
    ring: euclid(landmarks[tips.ring], wrist),
    pinky: euclid(landmarks[tips.pinky], wrist)
  }

  // normalize distances by hand size
  const nd = Object.fromEntries(Object.entries(tipDists).map(([k, v]) => [k, v / handSize]))

  // Tunable thresholds
  const extendedThresh = 0.30 // fingertip far from wrist => extended
  const curledThresh = 0.18 // fingertip near wrist => curled

  const extended = Object.fromEntries(Object.entries(nd).map(([k, v]) => [k, v > extendedThresh]))
  const curled = Object.fromEntries(Object.entries(nd).map(([k, v]) => [k, v < curledThresh]))

  // Scores
  const openScore = ['thumb', 'index', 'middle', 'ring', 'pinky'].reduce((s, k) => s + (extended[k] ? 1 : 0), 0)
  const openConfidence = openScore / 5

  const victoryScore = (extended.index ? 1 : 0) + (extended.middle ? 1 : 0) + (curled.ring ? 1 : 0) + (curled.pinky ? 1 : 0)
  const victoryConfidence = victoryScore / 4

  if (openConfidence > 0.75) return { name: 'Open Palm', confidence: Math.round(openConfidence * 100) }
  if (victoryConfidence > 0.7) return { name: 'Victory', confidence: Math.round(victoryConfidence * 100) }

  return { name: 'Unknown', confidence: Math.round(Math.max(openConfidence, victoryConfidence) * 100) }
}

export default function HandSignDetector({ targetSign = null, maxHands = 2 }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const detectorRef = useRef(null)
  const rafRef = useRef(null)
  const streamRef = useRef(null)
  const detectedRef = useRef({ name: 'None', confidence: 0 })
  const [detected, setDetected] = useState({ name: 'None', confidence: 0 })
  const [match, setMatch] = useState(false)
  const [fps, setFps] = useState(0)
  const [cameraReady, setCameraReady] = useState(false)
  const [cameraError, setCameraError] = useState(null)

  useEffect(() => {
    let mounted = true

    const setup = async () => {
      // init TF backend
      await tf.setBackend('webgl')
      await tf.ready()

      // create detector using MediaPipe Hands via tfjs runtime
      const detector = await handPoseDetection.createDetector(
        handPoseDetection.SupportedModels.MediaPipeHands,
        { runtime: 'tfjs', modelType: 'full', maxHands }
      )
      detectorRef.current = detector

      // start video
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false })
        streamRef.current = stream
        const video = videoRef.current
        video.srcObject = stream
        video.onloadedmetadata = () => setCameraReady(true)
        await video.play()
      } catch (err) {
        console.error('Camera error:', err)
        setCameraError(err?.message || 'Unable to access camera. Please check permissions or try another device.')
        return
      }

      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')

      const process = async () => {
        if (!mounted) return
        const video = videoRef.current
        if (!video || video.readyState < 2) {
          rafRef.current = requestAnimationFrame(process)
          return
        }

        // sync canvas size
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth
          canvas.height = video.videoHeight
        }

        // estimate hands
        let hands = []
        try {
          hands = await detectorRef.current.estimateHands(video, { flipHorizontal: true })
        } catch (err) {
          console.error('Estimate error', err)
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height)
        ctx.save()
        ctx.scale(1, 1)

        // draw each hand
        const signNames = []
        for (const hand of hands) {
          const keypoints = hand.keypoints || hand.landmarks || []
          ctx.lineWidth = 2
          ctx.strokeStyle = 'rgba(16, 185, 129, 0.9)'
          for (const [a,b] of CONNECTIONS) {
            const p1 = keypoints[a]
            const p2 = keypoints[b]
            if (!p1 || !p2) continue
            ctx.beginPath()
            ctx.moveTo(p1.x, p1.y)
            ctx.lineTo(p2.x, p2.y)
            ctx.stroke()
          }
          for (let i = 0; i < keypoints.length; i++) {
            const p = keypoints[i]
            ctx.beginPath()
            ctx.fillStyle = 'rgba(16, 185, 129, 0.95)'
            ctx.arc(p.x, p.y, 5, 0, Math.PI * 2)
            ctx.fill()
          }

          const res = detectHandSign(keypoints)
          signNames.push(res)
        }

        let final = { name: 'None', confidence: 0 }
        if (signNames.length === 2 && signNames[0].name === 'Victory' && signNames[1].name === 'Victory') {
          final = { name: 'Shadow Clone', confidence: Math.round((signNames[0].confidence + signNames[1].confidence) / 2) }
        } else if (signNames.length >= 1) {
          final = signNames.reduce((best, cur) => cur.confidence > best.confidence ? cur : best, { name: 'None', confidence: 0 })
        }

        if (final.name !== detectedRef.current.name || final.confidence !== detectedRef.current.confidence) {
          setDetected(final)
          setMatch(targetSign ? final.name === targetSign : false)
          detectedRef.current = final
        }

        const now = performance.now()
        if (!process._last) process._last = now
        if (!process._frames) process._frames = 0
        process._frames++
        const elapsed = now - process._last
        if (process._frames >= 10) {
          const fpsCalc = Math.round((process._frames / elapsed) * 1000)
          setFps(fpsCalc)
          process._frames = 0
          process._last = now
        }

        ctx.restore()
        rafRef.current = requestAnimationFrame(process)
      }

      rafRef.current = requestAnimationFrame(process)
    }

    setup()

    return () => {
      mounted = false
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop())
      }
      if (detectorRef.current && detectorRef.current.dispose) detectorRef.current.dispose()
    }
  }, [targetSign, maxHands])

  return (
    <div className="w-full flex flex-col items-center gap-6">
      <div className="relative w-full max-w-4xl overflow-hidden rounded-[2rem] border border-white/10 bg-slate-950/70 shadow-2xl shadow-slate-950/30 backdrop-blur-xl">
        <div className="relative aspect-[16/9] bg-slate-900/95">
          <video
            ref={videoRef}
            className="absolute inset-0 h-full w-full object-cover"
            style={{ transform: 'scaleX(-1)' }}
            playsInline
            muted
          />
          <canvas ref={canvasRef} className="absolute inset-0 h-full w-full pointer-events-none" />

          {!cameraError && !cameraReady && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-slate-950/90 text-center text-slate-200 px-6">
              <div className="rounded-full bg-slate-800/80 p-4 text-3xl">📷</div>
              <p className="text-lg font-semibold">Starting camera...</p>
              <p className="max-w-md text-sm text-slate-400">Allow camera access in your browser and hold your hand in view.</p>
            </div>
          )}

          {cameraError && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-slate-950/90 text-center text-slate-200 px-6">
              <div className="rounded-full bg-rose-500/15 p-4 text-3xl">⚠️</div>
              <p className="text-lg font-semibold">Camera unavailable</p>
              <p className="max-w-md text-sm text-slate-400">{cameraError}</p>
              <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Try another laptop, device, or browser if the camera does not appear.</p>
            </div>
          )}
        </div>
      </div>

      <div className="grid w-full max-w-4xl gap-4 md:grid-cols-4">
        <div className="rounded-3xl bg-slate-950/80 p-5 ring-1 ring-white/10 text-white shadow-xl shadow-slate-950/20">
          <p className="text-sm uppercase tracking-[0.24em] text-slate-400">Detected</p>
          <p className="mt-3 text-3xl font-semibold text-white">{detected.name}</p>
        </div>
        <div className="rounded-3xl bg-slate-950/80 p-5 ring-1 ring-white/10 text-white shadow-xl shadow-slate-950/20">
          <p className="text-sm uppercase tracking-[0.24em] text-slate-400">Confidence</p>
          <p className="mt-3 text-3xl font-semibold text-emerald-300">{detected.confidence}%</p>
        </div>
        <div className="rounded-3xl bg-slate-950/80 p-5 ring-1 ring-white/10 text-white shadow-xl shadow-slate-950/20">
          <p className="text-sm uppercase tracking-[0.24em] text-slate-400">Performance</p>
          <p className="mt-3 text-3xl font-semibold text-cyan-300">{fps} FPS</p>
        </div>
        <div className="rounded-3xl bg-slate-950/80 p-5 ring-1 ring-white/10 text-white shadow-xl shadow-slate-950/20">
          <p className="text-sm uppercase tracking-[0.24em] text-slate-400">Match Status</p>
          <div className="mt-3 flex items-center gap-3">
            <span className={`inline-flex h-4 w-4 shrink-0 rounded-full ${match ? 'bg-emerald-400' : 'bg-rose-400'}`}></span>
            <p className="text-xl font-semibold">{targetSign ? (match ? 'Matched' : 'No match') : 'No target'}</p>
          </div>
        </div>
      </div>

      <div className="w-full max-w-4xl rounded-[1.75rem] border border-white/10 bg-slate-900/80 px-6 py-5 text-slate-300 shadow-2xl shadow-slate-950/20">
        <h2 className="text-base font-semibold text-slate-100">How to use</h2>
        <p className="mt-3 text-sm leading-6 text-slate-400">Place your hand in front of the camera and try the configured sign. The overlay will draw hand landmarks and update the detection card in real time. If the camera is not available, check browser permissions or test on another device.</p>
      </div>
    </div>
  )
}
