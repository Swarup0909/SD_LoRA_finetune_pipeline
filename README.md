# LoRA Learnings — Fine-Tuning Stable Diffusion 1.5 with LoRA

A from-scratch, fully annotated Colab notebook for learning the **production pattern** used to fine-tune diffusion models with LoRA — not the "5 photos + magic trigger word" DreamBooth toy, but the **caption-based fine-tuning pipeline** real teams use on real (image, caption) datasets.

This repo is built to be **reused on any image dataset**. Swap one dataset name and a couple of training knobs, and you're fine-tuning on your own data.

> If you're new to LoRA / diffusion fine-tuning: read the notebook top to bottom once before touching anything. Every section has a markdown cell explaining **why** the step exists, not just what it does. The code is intentionally verbose/explicit instead of hidden behind a one-line `trainer.train()`, because the point is to understand the pipeline, not just run it.

---

## What's in here

| File | What it is |
|---|---|
| `SD_LoRA_finetune_pipeline.ipynb` | The full Colab notebook — dataset → model components → LoRA injection → training loop → inference |

## What this teaches

- How a diffusion model's training objective actually works (predicting **noise**, not pixels)
- Why fine-tuning only touches a tiny slice of the UNet (LoRA on the attention projections), not the whole model
- The real production stack: `diffusers` + `peft` + `accelerate` + `bitsandbytes`
- How to read a training loss curve and tell healthy vs. diverging vs. overfitting runs
- How to load a trained adapter back and compare base-model vs. fine-tuned output side by side
- How to point the exact same pipeline at **your own dataset**

## What it deliberately is *not*

- Not DreamBooth (single trigger word, 5–20 images, no captions) — that's a different, narrower pattern
- Not SDXL or FLUX — those need 24GB+ VRAM even with LoRA/QLoRA tricks; this is tuned to fit a **free Colab T4 (15GB VRAM)**
- Not a black-box script — every model component is loaded individually and explained

---

## Quick start

1. Open `SD_LoRA_finetune_pipeline.ipynb` in Google Colab.
2. `Runtime → Change runtime type → T4 GPU`.
3. Run the install cell (Section 0).
4. **`Runtime → Restart session`** — do this once, right after the install cell, before running anything else. Colab's base image preloads older versions of some of these libraries; restarting clears them so your pinned versions actually take effect. Do not re-run the install cell after restarting.
5. Run the rest of the notebook top to bottom.

A full run (15 epochs on ~833 images) takes roughly 35–60 minutes on a free T4.

### Pinned versions

These are pinned deliberately, not left open, because `diffusers` and `peft` move fast and a mismatched pair will fail at import:

```
diffusers==0.38.0
transformers==4.57.1
accelerate==1.13.0
peft==0.17.1
bitsandbytes==0.49.2
datasets==3.6.0
```

`torch` is **not** pinned or reinstalled — Colab's preinstalled torch is already matched to the GPU driver in your runtime, and reinstalling it yourself is the most common way to break a Colab GPU environment. Leave it alone.

If you ever see `ImportError: peft>=0.X.0 is required`, that error is telling you exactly which package is out of sync — bump it to the version named in the message.

---

## The dataset used here

[`reach-vb/pokemon-blip-captions`](https://huggingface.co/datasets/reach-vb/pokemon-blip-captions) — 833 Pokémon images, each with a real auto-generated caption (via BLIP). This is a live mirror of the original Lambda Labs dataset, which was taken down over a trademark dispute.

This dataset was chosen because it's:
- Small enough to train fully on a free T4 in under an hour
- Captioned per-image (not a single trigger word) — the pattern that generalizes to real client/product/brand datasets
- A clean example of **style + concept transfer**: the model learns "Pokémon-style creature" as a visual concept, conditioned on whatever you describe in the prompt

---

## Using your own dataset

This is the part that matters most if you want hands-on practice beyond this one dataset. The notebook is built so that **only Section 1 (dataset loading) needs to change** — everything downstream (model loading, LoRA injection, training loop, inference) works as-is regardless of what images you put in.

### Step 1 — Get your images into the right shape

You need a folder of images plus a caption for each one. The easiest format HuggingFace `datasets` understands is the `imagefolder` format:

```
my_dataset/
├── metadata.csv
├── image_001.jpg
├── image_002.jpg
└── ...
```

`metadata.csv`:
```csv
file_name,text
image_001.jpg,a red sports car parked on a city street at night
image_002.jpg,a red sports car driving on a coastal highway at sunset
```

- **100–300 images is a reasonable range to start.** Fewer than ~50 and the model has too little signal; LoRA doesn't need thousands like a full fine-tune would.
- **Write a real, varied caption per image** — describe what's actually in the picture (subject, setting, style, lighting), not just a repeated label. The model learns the relationship between *words* and *visual concepts*, so caption quality directly determines what it learns.
- **Keep the underlying "thing" coherent.** A specific art style, a single product line, your own photography, a specific character design — pick one coherent visual concept per training run.

### Step 2 — Load it instead of the Pokémon dataset

In Section 1, replace:

```python
dataset = load_dataset("reach-vb/pokemon-blip-captions", split="train")
```

With either:

```python
# Option A: local folder
dataset = load_dataset("imagefolder", data_dir="/content/my_dataset", split="train")

# Option B: pushed to the Hub (recommended — survives Colab disconnects)
dataset = load_dataset("your-username/my-dataset", split="train")
```

To push your folder to the Hub first:
```python
from datasets import load_dataset
ds = load_dataset("imagefolder", data_dir="/content/my_dataset", split="train")
ds.push_to_hub("your-username/my-dataset")
```

Everything else in Section 1 (the visual sanity check, caption length check, image size check) works unchanged — **run it**. Looking at your data before training is the single habit most worth keeping from this notebook.

### Step 3 — Adjust a few knobs based on your dataset size

In Section 5 (training loop setup):

| Your dataset size | `NUM_EPOCHS` | Notes |
|---|---|---|
| 50–150 images | 20–30 | Small datasets need more passes to converge |
| 150–400 images | 10–20 | What this notebook uses by default (833 images, 15 epochs) |
| 400+ images | 5–10 | More data per epoch means fewer epochs needed |

In Section 3 (LoRA config):

| Goal | `r` (rank) | Notes |
|---|---|---|
| Subtle style transfer | 4–8 | Fewer trainable params, less risk of overfitting on small data |
| Default / balanced | 16 | What this notebook uses |
| Complex new concept, larger dataset | 32+ | More capacity, needs more data to justify it |

### Step 4 — Change the validation prompt

In Section 8, the prompt `"a cute fire-type creature with wings, pixel art style"` is Pokémon-specific. Change it to something that exercises whatever concept your dataset represents — and ideally test 2–3 prompts, including one that's deliberately *outside* your dataset's exact captions, to see how well the model generalized vs. just memorized.

That's it — no other cell needs to change. If your dataset is captioned and sized reasonably, the pipeline handles the rest.

---

## Reading the loss curve (Section 6)

| What you see | What it means |
|---|---|
| Smooth downward trend | Healthy |
| Flat near the starting value | Learning rate too low, or LoRA rank too small for the task |
| Spikes or NaNs | Learning rate too high, or fp16 numerical instability |
| Drops to near-zero very fast | Likely memorizing a small dataset — check generated samples for lost diversity, don't trust the loss number alone |

---

## Troubleshooting

**`ImportError: peft>=0.X.0 is required`**
Your installed `peft` is older than what `diffusers` expects. Re-run the install cell, then restart the runtime — don't just `pip install --upgrade peft` mid-session without restarting, since the old version may already be loaded in memory.

**404 / repo not found when loading the base model**
`runwayml/stable-diffusion-v1-5` was deleted from Hugging Face in 2024 when RunwayML stopped maintaining their HF org. This notebook uses the official mirror, `stable-diffusion-v1-5/stable-diffusion-v1-5` — if you copy code from an older tutorial, update the model ID.

**CUDA out of memory**
- Confirm you're on a T4 runtime (`Runtime → Change runtime type`), not CPU.
- Lower `BATCH_SIZE` further (already at 1 by default) is not possible — instead lower `GRAD_ACCUM_STEPS` slightly, or lower `resolution` to 384.
- Restart the runtime to clear any leftover allocated memory from a previous failed run.

**Generated images look unrelated to your dataset**
- Check your captions actually describe the images (Section 1's visual sanity check exists for this reason).
- Increase `NUM_EPOCHS` or `r` (rank) — the model may not have had enough signal/capacity to learn the concept.

**Output images look identical regardless of prompt**
- Likely overfitting/collapse on a very small or repetitive dataset. Add more varied images, or lower `r` and `NUM_EPOCHS`.

---

## Where to go from here

1. **Vary the LoRA rank (`r`)** on the same dataset — 4, 8, 32 — and compare output quality and how the loss curve shifts. This is the single highest-leverage hyperparameter to build intuition for.
2. **Vary learning rate** — try `5e-5` and `5e-4` against the default `1e-4`.
3. **Train on 2–3 different datasets** using the steps above, to see how dataset size and caption quality change training behavior.
4. **Move to SDXL with LoRA** once you have access to a bigger GPU (Colab Pro / A100) — same conceptual pipeline, swap the model name, add a second text encoder, bump resolution to 1024.
5. **Read the official `diffusers` training script** this notebook mirrors: [`train_text_to_image_lora.py`](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py). Once this notebook makes sense, that script should be readable instead of intimidating.
6. **Try the LLM fine-tuning track** (QLoRA + SFT on a text dataset) — same underlying pattern: frozen base model, small trainable adapter, domain data, watch the loss, compare before/after.

---

## Reference

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — the original paper
- [HuggingFace PEFT docs](https://huggingface.co/docs/peft)
- [HuggingFace Diffusers LoRA training guide](https://huggingface.co/docs/diffusers/training/lora)
- [Lambda Labs' original Pokémon fine-tuning write-up](https://lambdalabs.com/blog/how-to-fine-tune-stable-diffusion-pokemon) (the inspiration for the dataset choice here)

## License

MIT for the notebook/code in this repo. The base model (`stable-diffusion-v1-5/stable-diffusion-v1-5`) is distributed under the CreativeML OpenRAIL-M license — review it before using any fine-tuned output commercially.
