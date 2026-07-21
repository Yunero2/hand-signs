# Human Review Checklist

This project is intended to be reviewed and verified by a person.

## Review steps

- [ ] Confirm the UI design looks good and is easy to use.
- [ ] Test camera access on a real laptop or device.
- [ ] Verify the hand landmark overlay appears and updates in real time.
- [ ] Confirm `Shadow Clone` matching logic is working as expected.
- [ ] Read through `src/components/HandSignDetector.jsx` and confirm the logic is understandable.
- [ ] Make any final styling or wording adjustments manually.

## Notes
- The current gesture detection is heuristic-based for demo purposes.
- If you want stronger accuracy, replace the heuristics with a trained model.
