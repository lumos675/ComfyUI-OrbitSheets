# ComfyUI-OrbitSheets

Character turnaround and location reference sheets built from a **MiniMax-H3**
camera move, with vision-model frame selection.

Generating a reference sheet angle-by-angle from text drifts — the same
courtyard comes back with a different arch count, the same character with a
different collar. H3 doesn't drift, because every angle is *the same shot*. One
continuous move around a still subject gives you views that genuinely agree
with each other. What's left is picking the useful frames, which is what this
pack does.

| Character | Location |
|---|---|
| Krea2 (or any image model) paints a full-body anchor → H3 circles the figure and pushes in to the face → 8 frames → sheet | An anchor still → H3 turns through 360° → 8 frames → sheet |

The character sheet also hands back a **voice sample**: H3 renders audio from
the same latent as the picture, so having the figure speak during the turn
costs no extra sampling and gives you a voice-timbre reference.

---

## Nodes

| Node | What it does |
|---|---|
| **Location Orbit Prompt (H3)** | Writes the camera prompt to H3's documented format — motion type + amplitude + speed, style statement leading, `N/A` where there's deliberately no sound |
| **Character Turnaround Prompt (H3)** | Same, for a figure: arc around, then push in to the face. Optional spoken line with `(S1)` / `<d>[Lang] …</d>` tagging |
| **Frame Select (vision-judged)** | Picks the keepers from the decoded frames |
| **Contact Sheet** | Lays them out as one grid |

### Interior vs exterior — the setting that matters most

An **arc shot** circles a subject from outside it. That's right for a building
or a monument. Inside a room there's nothing to circle, so asking for an arc
gives you a small sideways drift down the same wall: eight frames of one view,
with the wall behind the camera never seen.

Interiors need a **pan** — the camera turning on its own axis. Set `space`
accordingly; it's the difference between a usable sheet and eight copies of one
photo.

Both modes name a full 360° explicitly. "Reveals the space from every side"
describes an outcome and the model treats it as flavour; "turns through a
complete 360-degree rotation" is an instruction about the camera, and it
follows it.

### How frames get picked

Even spacing is the obvious approach and the wrong one: a camera move isn't
uniform, so evenly-spaced samples land on blurred mid-swing frames and on pairs
showing the same wall twice.

Instead the node shortlists by time, tiles the candidates into one numbered
contact sheet, and asks a vision model to compare them **against each other** —
scoring frames in isolation can't tell a near-duplicate from a genuinely new
angle, and one call beats twenty. `selection_brief` steers what it looks for;
the character workflow uses it to ask for one tight face close-up plus seven
full-body angles.

With no vision model reachable it falls back to Laplacian-variance sharpness
plus farthest-point spread, and says so in its `info` output. That's a normal
state, not an error.

---

## Requirements

**Models** (paths as used by the example workflows):

| Role | File |
|---|---|
| H3 diffusion | `diffusion_models/MinimaxH3/minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| H3 text encoder | `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| H3 video VAE | `vae/minimax_h3_video_vae_fp16.safetensors` |
| H3 audio VAE *(character only)* | `vae/minimax_h3_audio_vae_fp32.safetensors` |
| H3 turbo LoRA *(optional)* | `loras/Minimax/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors` |
| Anchor image model | any — the examples use Krea2 |

Swap the anchor model for whatever you like; only the H3 half is fixed.

**A vision-language model** is optional. Any OpenAI-compatible endpoint that
accepts images works — llama-server, LM Studio, Ollama, vLLM. Leave `llm_url`
empty and the node probes `127.0.0.1` on `8010`, `1234`, `11434`, `8000`, in
that order.

**Python:** `requests`. Everything else (torch, numpy, Pillow) already ships
with ComfyUI.

---

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lumos675/ComfyUI-OrbitSheets
```

Restart ComfyUI. The nodes appear under the **OrbitSheets** category.

Load a graph from `example_workflows/` (canvas format, drag onto the canvas).
`api_workflows/` holds the same graphs in API format for programmatic
submission.

---

## Notes on the settings that bite

**cfg is already 1.** The sampler runs through `BasicGuider`, which takes a
model and one conditioning and has no cfg input at all — that *is* cfg 1, which
is what a distilled LoRA wants. `CFGGuider` is the swap if you ever want
guidance above 1.

**Frame count sits on H3's grid.** `length` must be `17k + 5`. The examples use
260 (~10.8s at 24fps), which is what a full turn needs to read as a turn rather
than a drift. At 175 (~7.3s) the camera never gets round.

**Steps follow the LoRA.** The examples run 4 steps because the turbo LoRA is
distilled for 4. Bypass the LoRA node and raise them.

**VRAM.** `Frame Select` calls `unload_all_models()` before the vision call by
default — H3 is still resident at that point, and a large vision model landing
on top of it is how the render succeeds and the judging OOMs. Turn
`free_vram_first` off if your card has room.

**Labels are off** in the example sheets. Burnt-in text sits over the subject
and follows the frame into anything that reuses it. The numbers you see in the
vision model's montage are internal and never saved.

---

## Tests

No GPU, no ComfyUI, no model required:

```bash
python tests/test_nodes.py
```

Builds a synthetic camera move with deliberately blurred frames and checks that
selection rejects them, that the vision path honours picks and tops up short
answers, that malformed replies parse, and that both prompt builders emit
H3-conformant text.

---

## License

MIT
