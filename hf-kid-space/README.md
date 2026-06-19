---
title: How Does The Computer Know
emoji: "❄️"
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
python_version: "3.12"
pinned: false
---

# How does the computer know? (kids' edition)

A pick-only teaching demo for the Frozen character classifier. A kid taps a
picture, guesses which part the computer looked at most, then reveals the
attention heatmap and finds out. Built for "teaching kids about AI."

Source repo: https://github.com/megano/deep-learning-image-classifier

## Notes
- Pick-only (no upload): keeps inputs clean and the lesson on *how* the model
  decides, not *that* it classifies.
- Correct answers are pre-labeled from each image's actual Grad-CAM hot region.
- Pin `torch`/`fastai`/`python_version` to the training env or `export.pkl`
  fails to load (see the main repo's deploy notes).
