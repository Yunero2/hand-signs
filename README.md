# Hand Sign Detector

A hands-on React demo using Vite and Tailwind that runs in the browser and uses the webcam to track hand landmarks.

## What this does
- Shows a live camera preview with hand landmark overlay
- Detects simple gestures with a lightweight heuristic
- Marks the target gesture as `Shadow Clone`
- Displays status cards for current sign, confidence, FPS, and match state

## Run it locally

```bash
npm install
npm run dev
```

Then open:

```text
http://localhost:4173/
```

## How to use it
1. Allow camera access in your browser.
2. Hold your hand in view of the webcam.
3. The overlay should draw points and lines, and the status cards will update.

## Files to look at
- `src/App.jsx` — page layout and wrapper UI
- `src/components/HandSignDetector.jsx` — camera setup, hand detection, and status cards
- `src/index.css` — base styles and background treatment

## Notes
This is a manual demo project, built to be easy to read and easy to tweak.
It is meant for human review and improvement.

See `HUMAN_REVIEW.md` for the verification checklist.

## Troubleshooting
- If the camera does not start, check browser permissions or try a different browser/device.
- On Windows, install Node.js if `npm` is not available.
- If the app feels slow, close other camera apps and browser tabs.

