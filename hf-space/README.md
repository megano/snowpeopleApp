---
title: Frozen Character Classifier
emoji: "☃️"
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
python_version: "3.12"
pinned: false
---

# Frozen Character Classifier (Hugging Face Space)

Gradio demo for the 3-class Frozen classifier (Olaf / Elsa / Sven). Source repo:
https://github.com/megano/deep-learning-image-classifier

## Load-bug fix

The pkl was exported from Colab (Python 3.12). The Space previously failed on
`torch.load` ("code expected at most 16 arguments, got 18") because it ran a
different Python. `python_version: "3.12"` above matches the training env.
`requirements.txt` must also pin `torch` and `fastai` to the Colab versions
(capture with: `import sys, fastai, torch; print(sys.version, fastai.__version__, torch.__version__)`).
