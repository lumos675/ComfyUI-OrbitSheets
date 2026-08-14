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
| Krea2 (or any image model) paints the character from your own description → H3 renders the 5-shot turnaround (full body → face → left → right → back) → 8 frames → sheet | An anchor still → H3 renders the 5-shot location sheet (front → right → rear → left → wide) → 5 frames → sheet |

The character sheet also hands back a **voice sample**: H3 renders audio from
the same latent as the picture, so having the figure speak during the turn
costs no extra sampling and gives you a voice-timbre reference.

---

## Nodes

| Node | What it does |
|---|---|
| **Location Sheet Prompt (H3)** | Writes the location take. `coverage=cut views` (default) writes locked-off tripod shots joined by hard cuts (front → right → rear → left, plus optional wide + detail) — no camera motion means nothing for the model to get wrong, and it beat every continuous move tried against it. `coverage=continuous move` is one unbroken take instead: a pan on the spot indoors, an arc round the building outdoors, through the angle `rotation` asks for and `take_seconds` can hold |
| **Character Turnaround Prompt (H3)** | Same, for a figure: a 5-second I2VA sequence of six views joined by hard cuts — full body → face close-up → left profile → right profile → back → frightened face (`scared_shot`). `shot_seconds` sets each shot's length; the last keeps the remainder, so six shots fit the same 124 frames. Optional spoken line with `(S1)` / `<d>[Lang] …</d>` tagging. `framing` sets the full-body margin |
| **Frame Select (vision-judged)** | Picks the keepers from the decoded frames — one per detected shot, plus vision-judged extras |
| **Contact Sheet** | Lays them out as one grid |
| **Attention Backend** | Swaps the attention kernel for one model — see below |

### Five hard cuts beat a continuous orbit

Both sheets are now **5-shot sequences joined by hard cuts**, one ~1 second per
view, written to H3's guide format (`[Shot N] At 00:MM.mmm, the shot cuts
to...`). A cut forces the model to re-establish the subject at each angle
instead of drifting through a continuous camera move, so every view genuinely
lands on the sheet. A character gets full body → face → left → right → back; a
location gets front → back → left → right → top-down.

`space` tunes the location's wording: an **exterior** shows the building's
rear, sides and an aerial top-down; an **interior** shows the far wall, the two
side walls and a high overhead view of the room.

### Cuts beat every continuous move

On a cathedral interior, every continuous camera move lost. The in-place pan
rolled at every rate it was tried at — the frames tilt up and spiral into the
vault. A translational glide held level but only ever went one way, and six
frames of the same wall sliding past is not a location sheet. What won was
**hard cuts between locked-off tripod frames**: a static shot has no camera
motion to get wrong, so the tumbling is structurally impossible.

That is the default now — `coverage = cut views`, which writes H3's multi-shot
format (`[Shot N] At 00:MM.mmm, the shot cuts to...`) with every view pinned as
*"a locked-off static camera at eye level, the horizon level and centred, no
camera movement of any kind."*

The shipped workflow runs four shots at 1.0s each in a 124-frame take: front →
right-hand wall → rear wall → left-hand wall, with `wide_establishing_shot` and
`detail_shot` off. Turn them on for a big or irregular location where four
orthogonal views miss how the parts sit together.

`coverage = continuous move` keeps the unbroken take for locations the model
cannot extrapolate from a single view, where a cut list would have to invent
the far side rather than continue into it.

### The turn has to fit the take

For the moves that *do* turn — the exterior arc, and the interior pan if you
ask for one — what decides whether the take stays level is arithmetic. A
complete 360° inside 124 frames is **72°/s**, against the 24°/s of H3's own
worked example: *"one single continuous 360 degree orbital move, clockwise, at
a slow constant rate, completing exactly one full rotation across fifteen
seconds."* Nothing turns three times that fast and stays level, so the model
renders the move it *can* do at that rate — it rolls.

That prompt can no longer be written. `rotation` is rationed against
`take_seconds` at a ceiling of 40°/s, and asking for more **clamps rather than
obeys**, logging what it wrote and how to get what you asked for:

```
[OrbitSheets] full turn (360 degrees, needs a 9s take) across 5.0s is 72 deg/s,
past the 40 deg/s this model holds level. Writing 180 degrees instead, which
fits. For the full 360, give the take more frames: 9s needs length 216 on
MiniMaxH3ImageToVideo, and take_seconds to match.
```

| `rotation` | needs | frames (`length`) | rate |
|---|---|---|---|
| **auto (as far as the take allows)** *(default)* | — | any | ≤ 40°/s |
| quarter turn (90°) | 2.3s | 124 | 18°/s |
| half turn (180°) | 4.5s | 124 | 36°/s |
| full turn (360°) | **9.0s** | **220+** | ≤ 40°/s |

`take_seconds` has to match `length` on `MiniMaxH3ImageToVideo` (124 frames =
5.17s at 24fps, 260 = 10.8s). For a true 360°, set both — the take costs twice
as much to sample, and it is the only way to get all the way round.

There is no `speed` widget: the angle and the take length already fix the
speed, and "fast speed" bolted onto a 360 is how the tumbling take got
written. Turns are written at slow speed or not at all.

Two more things the rolling take taught, both from H3's guide:

* **Amplitude and the angle are the same instruction.** "Pans right with large
  amplitude at fast speed, through a complete 360-degree rotation" states the
  range twice, and H3's guide asks for the modifier only *when meaningful*.
  The angle stays; `amplitude` and `speed` are both gone — the guide rates the
  orbit its highest-risk move and asks for it small and slow, so that is the
  only way it gets written.
* **Say what happens, not what must not.** "The camera stays level and never
  tilts up, tilts down, rolls or leans" names four of H3's own documented
  motion types — Tilt Up, Tilt Down, Roll Clockwise/Counterclockwise — inside
  a negation the model does not reliably honour. The constraint is now
  positive: *the horizon stays level, the vertical lines of the walls and
  pillars stay vertical, the floor along the bottom of the frame and the
  ceiling along the top.* Same for the exterior arc, which used to be told it
  "never goes indoors" and is now told it "stays outside at ground level, at
  the same eye-level height and the same distance from the building" — which
  is also what stops the take ending on a drone shot of the roof.

### Framing — wide characters need margin

A reference sheet is only useful if every angle shows the whole character,
and H3 will crop whatever the frame can't hold: a dragon's spread wings, a
tail or a held weapon outstretches a 16:9 frame, and the model cuts it off
rather than zooming out on its own.

`framing` on the character node sets the shot scale and the margin:

| Option | Use when |
|---|---|
| **full body, generous margin** (default) | Wings, tails, horns, props — anything that can leave the body. The prompt demands the whole figure stay inside the frame with empty margin on every side for the entire rotation, and forbids any part touching or crossing the edges |
| full body, tight | A plain standing figure you want to fill the frame |
| medium (waist-up) | A sheet focused on the upper body |

For very wide characters, generate the anchor image (the opening frame from
your image model) with the same generous margin. H3 follows the reference's
framing, so a cropped anchor passes its crop on to every angle of the turn.

### Attention backend

Attention dominates a video model's cost: H3 attends over every frame at once,
so the kernel choice moves wall-clock far more than it would for a still image.

ComfyUI can pick one globally with `--use-ck-attention`, but that's a launch
flag applying to every workflow on the server. This node patches the model
object instead, so the choice lives in the graph, changes per run, and can't
surprise anything else.

Core ships `ModelAttentionBackend`, which patches the same way but offers only
pytorch and comfy kitchen. This one lists whatever your install has registered
— sage, flash and xformers included when present — so you don't need a separate
pack to reach the other kernels.

**comfy kitchen's kernel is int8**: it quantizes the attention computation. It's
the fastest option and the one to try first, but it's lossy in a way pytorch
and sage are not. Compare a render before committing. `default (unchanged)`
leaves the model exactly as the loader produced it and is always safe; a
backend that isn't installed falls back to the default with a warning rather
than failing the run.

### How frames get picked

Even spacing is the obvious approach and the wrong one: a camera move isn't
uniform, so evenly-spaced samples land on blurred mid-swing frames and on pairs
showing the same wall twice.

Instead the node shortlists by time, drops the soft frames, tiles the survivors
into one numbered contact sheet, and asks a vision model to compare them
**against each other** — scoring frames in isolation can't tell a
near-duplicate from a genuinely new angle, and one call beats twenty.
`selection_brief` steers what it looks for; the character workflow uses it to
ask for one tight face close-up plus seven full-body angles.

The judging model is whichever is available, in order of preference:

1. **An in-graph CLIP** — connect the same Qwen3-VL that encoded the anchor
   (`clip` input, already wired in the example workflows). The node judges the
   montage itself via core's `TextGenerate` path: no external server, no second
   model resident. When a CLIP is wired, no HTTP endpoint is ever consulted.
2. An **OpenAI-compatible HTTP** endpoint (`llm_url`, or auto-probed) — only
   when no CLIP is connected.
3. Laplacian-variance sharpness plus farthest-point spread when neither model
   is available.

**A location is not a character.** The turnaround cuts between views because a
figure on a plain backdrop can be re-established from any angle — the model
knows what a back looks like. A specific building it has seen once, it does
not: told to cut to "the rear", it either re-frames the facade it already has
(the sheet fills with the same view three times) or invents somewhere else and
wanders indoors. Rotation avoids the guess entirely, because every frame
overlaps the last and only has to be *continued*. That is why the location
sheet rotates and the character sheet cuts, and why `shots` on a rotation means
"pull six different directions out of the take", not "six shots were filmed".

**Hard-cut sheets are guaranteed by `shots` + `shot_split`.** A cut-based
sheet (both of them) is a known number of distinct static views, so `shots=5`
forces one sharp frame from each — every view lands on the sheet, no matter
what the vision model does. Set `count` equal to `shots` and the vision model
is skipped entirely; raise it and the extra picks come from the model.

`shot_split` decides how those groups are formed, and only one option is
reliable:

| Option | Groups by | Fails when |
|---|---|---|
| **`views (by content)`** (default) | k-means over per-frame colour signatures | never, for distinct views — this is the one to use |
| `cuts (detected)` | window-before vs window-after change detection | a view appears in two shots: two groups, one angle, another view lost |
| `even (by time)` | equal time slices | the model's shots are uneven, which they always are |

The reason the default is content and not time: H3 gives its shots wildly
different lengths from run to run — a front view can run three seconds and a
profile half of one, and the front can come back a second time. Any split by
time then lands two groups on the same angle and drops a third view entirely.
Grouping by appearance cannot make that mistake: five distinct views are five
clusters wherever the cuts fall, and near-duplicate frames share one cluster
instead of consuming two slots. Seeding is farthest-point rather than random,
so a re-run reproduces the same sheet.

**When the sheet repeats itself, read the `info` output.** Clustering asks for
`shots` groups and always gets that many, whether or not the video holds that
many different views — so a take where the model lingered on one subject for
three of its shots gets split into three slices of the same picture. That is
upstream of selection and no picker can fix it, so the node measures it: `info`
warns `only N distinct views in these X frames` when the picks repeat. Re-run
the video, or lower `shots` to what it actually delivered.

**Turn `keep_first_frame` off on a clustered sheet.** It prepends frame 0, and
frame 0 is already the anchor view — so it spends a slot on a duplicate and
pushes the last cluster off the end of `count`.

**`boards` — judge the whole move, not 16 tiles.** A single montage can only
hold ~16 readable tiles, so most of the orbit never reaches the model. With
`boards=4` the node splits the timeline into four sectors, shows each sector as
its own readable numbered board, and merges the picks — the model sees 4× the
frames at the same per-tile quality, at the cost of more vision calls.
`temperature` / `max_length` tune the in-graph judge.

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

**Frame count follows the shot count.** `length` must be `17k + 5`. Both
sheets are five one-second shots joined by hard cuts, so the examples run 124
frames (~5.2s at 24fps) — exactly the five shots, and nothing spent on camera
moves between them. Cuts are also why the sheets are stable: an unbroken
360° move has to invent every in-between frame, and mid-arc it tumbles into
rolled horizons and floor tiles. A cut costs nothing and re-establishes a
level frame at each angle.

**Steps follow the LoRA.** The examples run 4 steps because the turbo LoRA is
distilled for 4. Bypass the LoRA node and raise them.

**VRAM.** `Frame Select` calls `unload_all_models()` before the vision call by
default — H3 is still resident at that point, and a large vision model landing
on top of it is how the render succeeds and the judging OOMs. Turn
`free_vram_first` off if your card has room.

**Resolution.** The Krea2 anchor renders larger than the H3 output and is
downscaled into the first frame, so the move starts from a sharper reference.
Both sheets now run **1920×1080 anchor → 1216×672 H3** — a wide 16:9 frame
gives a dragon's wingspan, tail or held weapon physical room in every shot
without relying on the prompt to squeeze it in. `MiniMaxH3ImageToVideo`'s
width/height is where the final video's resolution lives — raise it only if
VRAM allows, since video cost scales with pixel count.

**Five-shot character sheets.** The turnaround is one 5-second I2VA sequence
(`124` frames ≈ 5s) of five distinct views joined by hard cuts, written to
H3's prompt guide format — the I2VA first-frame instruction line, then
`integrated_multimodal_description` with `[Shot 1]` … `[Shot 5]` and cut times
at each second. Full body → face close-up → left profile → right profile →
back, with `silent_during_closeup` keeping the mouth closed everywhere except
the opening shot. Both workflows save the finished move as an MP4
(`CreateVideo` → `SaveVideo`, `video/OrbitSheets` in your output directory).

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
