"""Gradio app for the Frozen Character Classifier (Hugging Face Space).

Loads the exported fastai model and serves predictions over the three classes
(Olaf, Elsa, Sven). Includes a confidence threshold so out-of-distribution or
unclear images get an honest "not sure" instead of a forced guess.

Deploy: copy app.py, requirements.txt, README.md, and export.pkl into the
Space repo (quidditch/deep-learning-image-classifier). The load-bug fix lives
in README.md (python_version) + requirements.txt (pinned torch/fastai to match
the training env).
"""

import gradio as gr
from fastai.vision.all import load_learner, PILImage

learn = load_learner("export.pkl")
labels = list(learn.dls.vocab)

# Below this top-class probability, we decline to guess rather than force one
# of the three. Tune after seeing real out-of-distribution inputs.
THRESHOLD = 0.60


def predict(img):
    if img is None:
        return {}, "Upload an image to get started."
    pred, idx, probs = learn.predict(PILImage.create(img))
    confidences = {labels[i]: float(probs[i]) for i in range(len(labels))}
    top = float(probs[idx])
    if top < THRESHOLD:
        msg = (
            f"Hmm, I am not confident this is one of my three characters "
            f"(best guess {pred} at {top:.0%}). I only know Olaf, Elsa, and Sven."
        )
    else:
        msg = f"I am {top:.0%} sure this is {pred}!"
    return confidences, msg


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath", label="Your image"),
    outputs=[
        gr.Label(num_top_classes=3, label="Prediction"),
        gr.Textbox(label="What I think"),
    ],
    title="Frozen Character Classifier",
    description=(
        "Upload a picture of Olaf, Elsa, or Sven from Disney's Frozen and the "
        "model will guess who it is. Trained on a small, curated dataset with "
        "transfer learning (ResNet-18)."
    ),
    examples=None,  # TODO: add a few sample images to the Space and list paths here
)
# Note: input flagging (for later Evidently drift monitoring) uses `flagging_mode`
# in Gradio 5 and `allow_flagging` in Gradio 4. Add it back when wiring monitoring,
# matched to the pinned Gradio version.

if __name__ == "__main__":
    demo.launch()
