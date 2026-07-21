import React from 'react'
import HandSignDetector from './components/HandSignDetector'

export default function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 text-slate-100 px-4 py-8 flex items-center justify-center">
      <div className="w-full max-w-5xl">
        <header className="mb-8 rounded-[2rem] border border-white/10 bg-white/5 p-8 shadow-2xl shadow-slate-950/20 backdrop-blur-xl">
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div>
              <h1 className="text-4xl font-semibold tracking-tight">Hand Sign Detector</h1>
              <p className="mt-3 max-w-2xl text-slate-300">A small browser demo that tracks hand landmarks and highlights the current sign. Designed to work in a modern browser with camera access.</p>
            </div>
            <div className="inline-flex items-center rounded-full bg-slate-900/80 px-4 py-2 text-sm font-medium text-slate-200 ring-1 ring-white/10">
              Target sign: <span className="ml-2 rounded-full bg-emerald-500/15 px-3 py-1 text-emerald-200">Shadow Clone</span>
            </div>
          </div>
        </header>

        <HandSignDetector targetSign="Shadow Clone" />
      </div>
    </div>
  )
}
