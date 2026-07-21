# Hand Sign Detector

A simple React + Vite + Tailwind app that uses the webcam and `@tensorflow-models/hand-pose-detection` to detect hand landmarks and recognize gestures in real time.

## Features
- Live camera preview with landmark overlay
- Real-time detection of common hand signs
- Target match logic for the `Shadow Clone` gesture
- Stylish dark UI with status cards for detected sign, confidence, FPS, and match state

## Setup

```bash
npm install
npm run dev
```

Open the local preview at:

```text
http://localhost:4173/
```

## Usage
1. Allow camera access in your browser.
2. Place your hand in front of the webcam.
3. Watch the overlay draw hand landmarks and update detection status.

## Project structure
- `src/App.jsx` — app shell and page layout
- `src/components/HandSignDetector.jsx` — camera setup, detection, and UI cards
- `src/index.css` — global styling and background effects

## Troubleshooting
- If the camera does not start, check browser permissions and try another browser or device.
- On Windows, install Node.js if `npm` is not available.
- If the app runs slowly, close other camera apps and browser tabs.

## Notes
This repo is configured for development with Vite and Tailwind CSS. The detection logic is heuristic-based and intended for demo purposes.

