"""Kid teaching demo: 'How does the computer know?'

Pick a picture, guess which part the computer looked at most, then reveal the
heatmap and find out. Pick-only (no upload) so inputs stay clean and the lesson
stays on *how* the model decides, not *that* it classifies.

Correct answers are pre-labeled from the actual Grad-CAM hot region of each
gallery image (deterministic inference keeps them valid). Deploy: copy app.py,
requirements.txt, README.md, examples/, and export.pkl into the Space repo.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from PIL import Image
import gradio as gr
from fastai.vision.all import load_learner, PILImage, first  # noqa: F401

learn = load_learner("export.pkl")

EX = "examples"
# Display order: grouped by character.
EXAMPLE_PATHS = [
    f"{EX}/olaf_009.png", f"{EX}/olaf_028.png", f"{EX}/olaf_004.png",
    f"{EX}/elsa_028.jpg", f"{EX}/elsa_015.jpg", f"{EX}/elsa_026.jpg",
    f"{EX}/sven_004.jpg", f"{EX}/sven_005.jpg", f"{EX}/sven_008.jpg",
]
# Square padded thumbnails for a clean, uniform gallery grid (display only).
# The model still runs on the originals in EXAMPLE_PATHS (selected by index).
THUMB_PATHS = [p.replace("examples/", "examples_thumb/") for p in EXAMPLE_PATHS]

# Pre-labeled hottest region per image (from Grad-CAM).
CORRECT_PART = {
    "olaf_009.png": "Face",
    "olaf_028.png": "Carrot nose",
    "olaf_004.png": "Whole body",
    "elsa_028.jpg": "Face / eyes",
    "elsa_015.jpg": "Dress / body",
    "elsa_026.jpg": "Hair / braid",
    "sven_004.jpg": "Fuzzy face & mane",
    "sven_005.jpg": "Fuzzy face & mane",
    "sven_008.jpg": "Fuzzy face & mane",
}

# Tappable options per character (correct answer + plausible distractors).
OPTIONS = {
    "olaf": ["Carrot nose", "Face", "Whole body", "Coal buttons"],
    "elsa": ["Face / eyes", "Hair / braid", "Dress / body", "Hands"],
    "sven": ["Fuzzy face & mane", "Antlers", "Nose", "Legs"],
}


class _Hook:
    def __init__(s, m): s.h = m.register_forward_hook(s.f)
    def f(s, m, i, o): s.stored = o.detach().clone()
    def __enter__(s, *a): return s
    def __exit__(s, *a): s.h.remove()


class _HookBwd:
    def __init__(s, m): s.h = m.register_full_backward_hook(s.f)
    def f(s, m, gi, go): s.stored = go[0].detach().clone()
    def __enter__(s, *a): return s
    def __exit__(s, *a): s.h.remove()


def _overlay(path):
    pil = PILImage.create(path)
    learn.model.zero_grad()
    x, = first(learn.dls.test_dl([pil]))
    with _HookBwd(learn.model[0]) as hb, _Hook(learn.model[0]) as h:
        out = learn.model.eval()(x)
        act = h.stored[0]
        idx = int(out.argmax(1).item())
        out[0, idx].backward()
        grad = hb.stored[0]
    cam = (grad.mean(dim=[1, 2], keepdim=True) * act).sum(0).relu()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_np = np.array(
        Image.fromarray((cam.detach().cpu().numpy() * 255).astype("uint8")).resize((224, 224), Image.BILINEAR)
    ) / 255.0
    base = np.array(pil.resize((224, 224)).convert("RGB")).astype(float)
    heat = cm.magma(cam_np)[..., :3] * 255.0
    alpha = (cam_np * 0.75)[..., None]
    return Image.fromarray((base * (1 - alpha) + heat * alpha).astype("uint8"))


def on_select(evt: gr.SelectData):
    path = EXAMPLE_PATHS[evt.index]
    name = os.path.basename(path)
    char = name.split("_")[0]
    pred, idx, probs = learn.predict(PILImage.create(path))
    conf = float(probs[idx])
    msg = (
        f"### I am {conf:.0%} sure this is **{str(pred).title()}**!\n"
        "Which part do you think I looked at most? Pick one, then press **Show me!**"
    )
    return (
        path,
        CORRECT_PART[name],
        gr.update(value=msg, visible=True),
        gr.update(choices=OPTIONS[char], value=None, visible=True),
        gr.update(visible=True),
        gr.update(value=None, visible=False),
        gr.update(visible=False),
    )


def on_reveal(path, correct, guess):
    if not path:
        return gr.update(), gr.update(value="Pick a picture first!", visible=True), gr.update()
    overlay = _overlay(path)
    if guess is None:
        fb = "Pick a part first, then press Show me!"
    elif guess == correct:
        fb = f"### You got it! The computer looked most at the **{correct}**."
    else:
        fb = f"### Good guess! But the computer actually looked most at the **{correct}**."
    return (gr.update(value=overlay, visible=True), gr.update(value=fb, visible=True),
            gr.update(label="Pick again!"))


CSS = ".gradio-container { max-width: 640px !important; }"

with gr.Blocks(title="How does the computer know?", css=CSS) as demo:
    gr.Markdown(
        "# How does the computer know?\n"
        "Tap a picture. The computer guesses who it is. Then *you* guess which "
        "part it looked at, and we reveal where it actually looked!"
    )
    state_path = gr.State()
    state_correct = gr.State()

    pred_md = gr.Markdown(visible=False)
    options = gr.Radio(choices=[], label="Which part did the computer look at most?", visible=False)
    reveal_btn = gr.Button("Show me!", variant="primary", visible=False)
    heat = gr.Image(label="Where the computer looked", visible=False)
    feedback = gr.Markdown(visible=False)

    gallery = gr.Gallery(value=THUMB_PATHS, label="Tap a picture", columns=3,
                         object_fit="cover", allow_preview=False)

    gallery.select(
        on_select, None,
        [state_path, state_correct, pred_md, options, reveal_btn, heat, feedback],
    )
    reveal_btn.click(on_reveal, [state_path, state_correct, options], [heat, feedback, gallery])

if __name__ == "__main__":
    demo.launch()
