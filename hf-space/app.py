"""Gradio app for the Frozen Character Classifier (Hugging Face Space).

Predicts over the three classes (Olaf, Elsa, Sven) with:
  - a confidence threshold so unclear / out-of-distribution images get an honest
    "not sure" instead of a forced guess, and
  - a Grad-CAM heatmap (shown to users as "Where the computer looked") so you can
    see which regions drove the prediction.

Deploy: copy app.py, requirements.txt, README.md, and export.pkl into the
Space repo (quidditch/deep-learning-image-classifier).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from PIL import Image
import torch
import gradio as gr
from fastai.vision.all import load_learner, PILImage, first  # noqa: F401

learn = load_learner("export.pkl")
labels = list(learn.dls.vocab)

# Below this top-class probability, decline to guess rather than force one of three.
THRESHOLD = 0.60


class _Hook:
    def __init__(self, m): self.h = m.register_forward_hook(self.f)
    def f(self, m, i, o): self.stored = o.detach().clone()
    def __enter__(self, *a): return self
    def __exit__(self, *a): self.h.remove()


class _HookBwd:
    def __init__(self, m): self.h = m.register_full_backward_hook(self.f)
    def f(self, m, gi, go): self.stored = go[0].detach().clone()
    def __enter__(self, *a): return self
    def __exit__(self, *a): self.h.remove()


def _attention_overlay(pil, act, grad):
    """Blend a Grad-CAM heatmap over the input image (both 224x224)."""
    cam = (grad.mean(dim=[1, 2], keepdim=True) * act).sum(0).relu()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_np = cam.detach().cpu().numpy()
    cam_img = Image.fromarray((cam_np * 255).astype("uint8")).resize((224, 224), Image.BILINEAR)
    heat = (cm.magma(np.array(cam_img) / 255.0)[..., :3] * 255).astype("uint8")
    base = pil.resize((224, 224)).convert("RGB")
    return Image.blend(base, Image.fromarray(heat), alpha=0.5)


def classify(img):
    if img is None:
        return {}, "Upload an image to get started.", None
    pil = PILImage.create(img)
    learn.model.zero_grad()
    x, = first(learn.dls.test_dl([pil]))

    with _HookBwd(learn.model[0]) as hb, _Hook(learn.model[0]) as h:
        out = learn.model.eval()(x)
        act = h.stored[0]
        idx = int(out.argmax(1).item())
        out[0, idx].backward()
        grad = hb.stored[0]

    probs = torch.softmax(out[0], dim=0)
    confidences = {labels[i]: float(probs[i]) for i in range(len(labels))}
    top, pred = float(probs[idx]), labels[idx]
    if top >= THRESHOLD:
        msg = f"I am {top:.0%} sure this is {pred}! The bright spots show where I looked."
    else:
        msg = (
            f"Hmm, I am not confident this is one of my three characters "
            f"(best guess {pred} at {top:.0%}). I only know Olaf, Elsa, and Sven."
        )
    return confidences, msg, _attention_overlay(pil, act, grad)


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="filepath", label="Your image"),
    outputs=[
        gr.Label(num_top_classes=3, label="Prediction"),
        gr.Textbox(label="What I think"),
        gr.Image(label="Where the computer looked"),
    ],
    title="Frozen Character Classifier",
    description=(
        "Upload a picture of Olaf, Elsa, or Sven from Disney's Frozen. The model "
        "guesses who it is, how sure it is, and shows where it looked to decide. "
        "The bright spots are what the computer paid most attention to."
    ),
    examples=None,  # TODO: add a few sample images to the Space and list paths here
)

if __name__ == "__main__":
    demo.launch()
