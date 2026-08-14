"""Character and location reference sheets from a MiniMax-H3 camera move.

A reference sheet generated angle-by-angle from text drifts: the same courtyard
comes back with a different arch count, the same character with a different
collar. H3 does not drift, because every angle is the *same shot* — one
continuous move around a still subject yields views that genuinely agree with
each other. The remaining job is picking the useful frames, which is what this
pack does.

Four nodes, each covering a step ComfyUI has no answer for:

  * Location Sheet Prompt      writes a 4-view location prompt to H3's spec
  * Character Turnaround Prompt  same for a figure, plus an expression shot
  * Frame Select               picks the frames worth keeping
  * Contact Sheet              lays them out as a sheet

Generation stays ordinary graph wiring — an image model for the opening
reference, `MiniMaxH3ImageToVideo` plus a sampler for the move — so seeds,
steps and models remain tunable where they belong. See `example_workflows/`
for the two complete graphs.

Frame selection asks a vision-language model to compare the candidates as one
contact sheet, and falls back to sharpness-and-spread scoring whenever no such
model is reachable. Nothing here requires a specific server: any
OpenAI-compatible vision endpoint works, and the pack has no dependencies
outside ComfyUI's own runtime.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re

import numpy as np
import torch
import torch.nn.functional as F

#: Frames the vision model is asked to compare in one montage. Beyond roughly
#: this many the tiles get too small for it to judge sharpness reliably.
DEFAULT_CANDIDATES = 16

#: Framing for the turnaround. Generous margin is the default because it is
#: the one a reference sheet usually fails without: a dragon's wingspan, a tail
#: or a held weapon outstretches a 16:9 frame, and H3 faithfully crops whatever
#: the prompt did not insist stays inside it.
_FRAMINGS = ["full body, generous margin", "full body, tight"]

_FRAMING_CLAUSES = {
    "full body, generous margin": (
        " Every full-body shot is framed with generous empty margin on every "
        "side of the figure, so that the whole figure and anything extending "
        "beyond its body stays fully inside the frame at all times. No part "
        "of the figure may ever touch, cross or be cut off by the edges of "
        "the frame."
    ),
    "full body, tight": (
        " Every full-body shot is framed with tight cropping, the figure "
        "nearly filling the frame from edge to edge but never cropped at any "
        "point."
    ),
}


#: How far the camera turns, in ascending order. Stating the angle is what
#: replaces H3's "large/small amplitude" here: the angle *is* the amplitude,
#: and saying both invites the model to average two instructions. The angle and
#: the take length between them also fix the speed, so there is no speed widget
#: either — a turn is written "at slow speed" or it is not written at all.
_TURNS = [
    (90.0, "a quarter turn of 90 degrees"),
    (180.0, "a half turn of 180 degrees"),
    (360.0, "a complete turn of all 360 degrees"),
]

#: Degrees per second above which the move stops being a camera move and
#: becomes a whip. H3's own worked example budgets a full 360 across fifteen
#: seconds — 24 deg/s — and asking for three times that is what makes the
#: frames tumble: with no way to turn that fast and stay level, the model
#: substitutes the roll and tilt it *can* render at that rate. Nothing past
#: this limit is written into a prompt; `_fit_turn` clamps it first.
_ROTATION_RATE_LIMIT = 40.0

#: The widget. `None` means "as far as this take can hold", which is the
#: default because it is the only setting that cannot be wrong. The others say
#: what they need in the label, since the label is the only documentation a
#: dropdown ever gets read with.
_ROTATION_CHOICES = {
    "auto (as far as the take allows)": None,
    "quarter turn (90 degrees)": 90.0,
    "half turn (180 degrees)": 180.0,
    "full turn (360 degrees, needs a 9s take)": 360.0,
}


def _fit_turn(rotation, take_seconds):
    """Pick the turn to write. Never returns one the take cannot hold.

    Returns (degrees, phrase, note): `note` is None when the request was
    honoured and a sentence explaining the clamp when it was not.
    """
    span = max(float(take_seconds), 1.0)
    ceiling = _ROTATION_RATE_LIMIT * span
    fits = [t for t in _TURNS if t[0] <= ceiling] or _TURNS[:1]

    wanted = _ROTATION_CHOICES.get(rotation, None)
    if wanted is None:                       # auto, or a stale widget value
        return fits[-1][0], fits[-1][1], None

    degrees, phrase = next(t for t in _TURNS if t[0] == wanted)
    if degrees <= ceiling:
        return degrees, phrase, None

    capped, capped_phrase = fits[-1]
    return capped, capped_phrase, (
        f"{rotation} across {span:.1f}s is {degrees / span:.0f} deg/s, past "
        f"the {_ROTATION_RATE_LIMIT:.0f} deg/s this model holds level. "
        f"Writing {capped:.0f} degrees instead, which fits. For the full "
        f"{degrees:.0f}, give the take more frames: {degrees / _ROTATION_RATE_LIMIT:.0f}s "
        f"needs length {int(round(degrees / _ROTATION_RATE_LIMIT * 24 / 4)) * 4} "
        f"on MiniMaxH3ImageToVideo, and take_seconds to match."
    )

#: How long the sheet's video runs, in seconds. 124 frames on H3's 17k+5 grid
#: is 5.17s at 24fps, and the shots divide that span rather than each taking a
#: whole second — six shots simply make each one shorter, which H3 follows
#: perfectly well and which costs no extra frames.
SHEET_SECONDS = 5.0


def _timecode(seconds: float) -> str:
    """Seconds -> H3's MM:SS.mmm cut marker."""
    minutes, rest = divmod(seconds, 60.0)
    return f"{int(minutes):02d}:{rest:06.3f}"


# ---------------------------------------------------------------- utilities


def _tensor_to_pils(images: torch.Tensor) -> list:
    """ComfyUI IMAGE (B,H,W,C float 0..1) -> list of PIL images."""
    from PIL import Image

    arr = (images.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return [Image.fromarray(arr[i]) for i in range(arr.shape[0])]


def _pils_to_tensor(pils: list) -> torch.Tensor:
    """PIL images -> ComfyUI IMAGE. Sizes must already agree."""
    stack = [np.asarray(p.convert("RGB"), dtype=np.float32) / 255.0 for p in pils]
    return torch.from_numpy(np.stack(stack, axis=0))


def _load_font(size: int):
    """A legible truetype face if the box has one, else PIL's bitmap default.

    The default face does not scale, so labels on a 512px tile would be
    unreadable; every common Linux image ships at least one of these.
    """
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _montage(pils: list, columns: int, cell_width: int, padding: int,
             labels: list | None, background: tuple = (18, 18, 20)):
    """Grid the images, optionally stamping a label on each tile."""
    from PIL import Image, ImageDraw

    if not pils:
        raise ValueError("No frames to lay out.")

    columns = max(1, min(columns, len(pils)))
    rows = (len(pils) + columns - 1) // columns

    ratio = pils[0].height / pils[0].width
    cell_h = max(1, int(round(cell_width * ratio)))

    sheet_w = columns * cell_width + padding * (columns + 1)
    sheet_h = rows * cell_h + padding * (rows + 1)
    sheet = Image.new("RGB", (sheet_w, sheet_h), background)

    draw = ImageDraw.Draw(sheet)
    font = _load_font(max(14, cell_width // 16))

    for index, pil in enumerate(pils):
        row, col = divmod(index, columns)
        x = padding + col * (cell_width + padding)
        y = padding + row * (cell_h + padding)
        sheet.paste(pil.convert("RGB").resize((cell_width, cell_h)), (x, y))

        if not labels:
            continue
        text = str(labels[index])
        # A filled plate behind the text: a bare glyph vanishes against a
        # bright sky or a pale wall, and these tiles are exactly that.
        try:
            box = draw.textbbox((0, 0), text, font=font)
            tw, th = box[2] - box[0], box[3] - box[1]
        except Exception:
            tw, th = 8 * len(text), 14
        pad = max(4, cell_width // 96)
        draw.rectangle(
            [x, y, x + tw + pad * 2, y + th + pad * 2], fill=(0, 0, 0)
        )
        draw.text((x + pad, y + pad), text, fill=(255, 255, 255), font=font)

    return sheet


def _pil_to_data_uri(pil) -> str:
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------- classical scoring


def _sharpness(images: torch.Tensor) -> torch.Tensor:
    """Variance of the Laplacian per frame — the standard blur detector.

    An orbit spends part of its arc mid-motion, and those frames are soft. A
    soft frame is useless as a reference no matter how good its angle is.
    """
    gray = images.mean(dim=3).unsqueeze(1)  # B,1,H,W
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=gray.dtype, device=gray.device,
    ).view(1, 1, 3, 3)
    lap = F.conv2d(gray, kernel, padding=1)
    return lap.view(lap.shape[0], -1).var(dim=1)


def _descriptors(images: torch.Tensor) -> torch.Tensor:
    """A coarse, brightness-normalised thumbnail per frame.

    Contrast-normalising means two views separate by *layout*, not by one
    being a stop brighter — which is what "a different angle" actually means.
    """
    gray = images.mean(dim=3).unsqueeze(1)
    small = F.interpolate(gray, size=(16, 16), mode="area").view(gray.shape[0], -1)
    small = small - small.mean(dim=1, keepdim=True)
    return small / (small.std(dim=1, keepdim=True) + 1e-6)


def _view_descriptors(images: torch.Tensor) -> torch.Tensor:
    """A per-frame signature fine enough to tell two camera angles apart.

    Colour is kept and the resolution is higher than `_descriptors` because
    this one has to separate views a coarse grey thumbnail merges: a left
    profile from a right profile (mirror images, identical silhouette area) and
    a front from a back (same outline, different content). Each frame is
    normalised on its own so the comparison is of layout, not exposure.
    """
    small = images.permute(0, 3, 1, 2)                       # B,C,H,W
    small = F.interpolate(small, size=(32, 32), mode="area")
    small = small.reshape(small.shape[0], -1)
    small = small - small.mean(dim=1, keepdim=True)
    return small / (small.std(dim=1, keepdim=True) + 1e-6)


def _cluster_views(images: torch.Tensor, k: int) -> list[list[int]]:
    """Group frames by what they show, returning k groups of frame indices.

    This is the answer to shot boundaries that move. Splitting the timeline by
    time — evenly, or even at correctly detected cuts — only works if each
    slice happens to hold a different view, and across runs it does not: the
    model gives one view three seconds and another half a second, so two slices
    come back with the same angle and a third view never reaches the sheet at
    all. Grouping by appearance cannot make that mistake. Five distinct views
    are five clusters wherever the cuts happen to fall, and near-duplicate
    frames land in one cluster instead of consuming two slots.

    Plain k-means over `_view_descriptors`, with farthest-point seeding rather
    than random, so the same video always yields the same sheet.
    """
    total = int(images.shape[0])
    k = max(1, min(k, total))
    desc = _view_descriptors(images)

    # Farthest-point seeding: start from the anchor frame and repeatedly take
    # the frame least like anything seeded so far. That lands one seed on each
    # genuinely distinct view, which random seeding routinely fails to do —
    # and it is deterministic, so a re-run reproduces the sheet exactly.
    seeds = [0]
    while len(seeds) < k:
        dist = torch.cdist(desc, desc[seeds]).min(dim=1).values
        dist[torch.tensor(seeds, device=dist.device)] = -1.0
        seeds.append(int(torch.argmax(dist).item()))
    centres = desc[seeds].clone()

    labels = torch.zeros(total, dtype=torch.long)
    for _ in range(25):
        labels = torch.cdist(desc, centres).argmin(dim=1)
        moved = False
        for c in range(k):
            members = labels == c
            if not bool(members.any()):
                # An emptied cluster would silently cost a view, so re-seed it
                # on the frame currently worst served by any centre.
                worst = int(torch.cdist(desc, centres).min(dim=1).values.argmax())
                centres[c] = desc[worst]
                moved = True
                continue
            mean = desc[members].mean(dim=0)
            if not torch.allclose(mean, centres[c]):
                centres[c] = mean
                moved = True
        if not moved:
            break

    labels = torch.cdist(desc, centres).argmin(dim=1)
    return [
        [i for i in range(total) if int(labels[i]) == c]
        for c in range(k)
    ]


def _greedy_spread(images: torch.Tensor, count: int, sharpness_weight: float,
                   forced_first: bool) -> list[int]:
    """Farthest-point selection, biased toward frames that are in focus."""
    total = images.shape[0]
    if total <= count:
        return list(range(total))

    sharp = _sharpness(images)
    desc = _descriptors(images)
    span = sharp.max() - sharp.min()
    sharp_norm = (sharp - sharp.min()) / (span + 1e-9)

    chosen = [0] if forced_first else [int(torch.argmax(sharp).item())]
    while len(chosen) < count:
        dist = torch.cdist(desc, desc[chosen]).min(dim=1).values
        dist = dist / (dist.max() + 1e-9)
        score = sharpness_weight * sharp_norm + (1.0 - sharpness_weight) * dist
        score[torch.tensor(chosen, device=score.device)] = float("-inf")
        chosen.append(int(torch.argmax(score).item()))
    return sorted(chosen)


# ------------------------------------------------------------- vision call


#: Where an OpenAI-compatible vision endpoint usually lives. Probed in order
#: when the node is left on its default, so the common setups work untouched:
#: llama-server, LM Studio, Ollama, vLLM and friends all speak this API.
_LLM_CANDIDATES = (
    "http://127.0.0.1:8010",   # llama-server, the default in these workflows
    "http://127.0.0.1:1234",   # LM Studio
    "http://127.0.0.1:11434",  # Ollama
    "http://127.0.0.1:8000",   # vLLM / generic
)


def _llm_base_url(override: str) -> str:
    """Resolve the vision endpoint, preferring an explicit address.

    Self-contained by design: this pack asks the network what is listening
    rather than importing from a sibling node pack, so it works on a plain
    ComfyUI install with any OpenAI-compatible server.
    """
    if override.strip():
        return override.strip().rstrip("/")

    import requests

    for base in _LLM_CANDIDATES:
        try:
            response = requests.get(f"{base}/v1/models", timeout=1.5)
            if response.status_code < 500:
                return base
        except Exception:
            continue
    # Nothing answered. Return the primary anyway so the caller's error names a
    # concrete address instead of an empty string.
    return _LLM_CANDIDATES[0]


#: What to look for when the caller says nothing. Tuned for a location orbit,
#: where every frame is the same place and the only question is which views
#: are sharp and genuinely different.
DEFAULT_BRIEF = (
    "Prefer: sharp, well-exposed frames; clearly distinct camera angles that "
    "each reveal a different side or aspect.\n"
    "Reject: motion-blurred or smeared frames; near-duplicates of a frame you "
    "already chose; frames where the subject is cropped, occluded, or the "
    "camera is too close to read the space."
)


def _vlm_pick(pils: list, count: int, hint: str, base_url: str,
              timeout: int, brief: str = "") -> tuple[list[int] | None, str]:
    """Ask the vision model to choose, comparing every candidate at once.

    One montage rather than one call per frame: scoring frames in isolation
    cannot tell a near-duplicate from a genuinely new angle, and a 31B vision
    pass per frame would cost more than the orbit did. Returns indices into
    [pils], or None when the model is unreachable or unusable.
    """
    import requests

    labels = [str(i + 1) for i in range(len(pils))]
    board = _montage(pils, columns=4, cell_width=384, padding=10, labels=labels)

    subject = hint.strip() or "a location"
    instruction = (
        f"This is a numbered contact sheet of {len(pils)} frames taken from a "
        f"single continuous camera move around {subject}. Choose exactly "
        f"{count} frames that together document the subject best.\n"
        f"{brief.strip() or DEFAULT_BRIEF}\n"
        'Reply with JSON only, no prose: {"picks": [numbers], "why": "one short sentence"}'
    )

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url",
                     "image_url": {"url": _pil_to_data_uri(board)}},
                ],
            }
        ],
        "max_tokens": 400,
        "temperature": 0.2,
    }

    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions", json=payload, timeout=timeout
        )
        if response.status_code != 200:
            return None, f"vision model returned HTTP {response.status_code}"
        content = response.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        return None, f"vision model unreachable ({type(exc).__name__})"

    picks = _parse_picks(content, len(pils))
    if not picks:
        return None, "vision model gave no usable frame numbers"

    why = ""
    try:
        why = str(json.loads(_json_slice(content)).get("why", ""))[:160]
    except Exception:
        pass

    # Short answers are salvageable; top up by spread rather than failing.
    return picks[:count], (why or "chosen by vision model")


def _json_slice(text: str) -> str:
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    return cleaned[start:end + 1] if start >= 0 and end > start else cleaned


def _parse_picks(text: str, total: int) -> list[int]:
    """Pull frame numbers out of the reply, tolerating chatty models."""
    numbers: list[int] = []
    try:
        payload = json.loads(_json_slice(text))
        raw = payload.get("picks", [])
        numbers = [int(n) for n in raw if isinstance(n, (int, float, str))
                   and str(n).strip().lstrip("-").isdigit()]
    except Exception:
        numbers = [int(n) for n in re.findall(r"\b\d{1,3}\b", text)]

    seen, out = set(), []
    for number in numbers:
        index = number - 1  # labels are 1-based
        if 0 <= index < total and index not in seen:
            seen.add(index)
            out.append(index)
    return out


def _clip_pick(clip, pils, count, hint, brief, max_length, temperature):
    """Judge the montage with an in-graph vision-language CLIP (Qwen3-VL etc.).

    Mirrors core's `TextGenerate` path — tokenize the instruction with the
    image attached, autoregressively generate, decode — so the very CLIP that
    encoded the anchor's prompt can pick the frames, with no external server
    and no second model resident. Only CLIPs whose tokenizer accepts images
    and whose model can generate will work; anything else is caught by the
    caller and falls back to the HTTP path.
    """
    labels = [str(i + 1) for i in range(len(pils))]
    board = _montage(pils, columns=4, cell_width=384, padding=10, labels=labels)

    subject = hint.strip() or "a location"
    instruction = (
        f"This is a numbered contact sheet of {len(pils)} frames taken from a "
        f"single continuous camera move around {subject}. Choose exactly "
        f"{count} frames that together document the subject best.\n"
        f"{brief.strip() or DEFAULT_BRIEF}\n"
        'Reply with JSON only, no prose: {"picks": [numbers], "why": "one short sentence"}'
    )

    image = _pils_to_tensor([board])
    tokens = clip.tokenize(instruction, image=image, min_length=1)
    ids = clip.generate(
        tokens,
        do_sample=float(temperature) > 0.0,
        max_length=int(max_length),
        temperature=float(temperature),
        top_k=64,
        top_p=0.95,
        min_p=0.05,
        repetition_penalty=1.05,
        seed=0,
    )
    return clip.decode(ids), "in-graph CLIP"


# ------------------------------------------------------- attention backend


#: Shown when a model should be left exactly as the loader produced it.
_ATTENTION_DEFAULT = "default (unchanged)"

#: Registry names are terse; these read better in a menu. Anything not listed
#: is offered under its own name, so a backend added to ComfyUI later still
#: appears here without this pack being updated.
_ATTENTION_LABELS = {
    "comfy_kitchen_int8": "comfy kitchen (int8)",
    "pytorch": "pytorch (SDPA)",
    "sage": "sage",
    "sage3": "sage 3",
    "flash": "flash",
    "xformers": "xformers",
    "sub_quad": "sub-quadratic",
    "split": "split",
}


def _attention_registry() -> dict:
    """Attention backends this ComfyUI actually has, name -> callable.

    Read live rather than hardcoded: which backends exist depends on what is
    installed (sage, flash, xformers) and on the launch flags. Empty on a build
    with no registry, which is what keeps this node harmless there.
    """
    try:
        from comfy.ldm.modules.attention import REGISTERED_ATTENTION_FUNCTIONS

        return dict(REGISTERED_ATTENTION_FUNCTIONS)
    except Exception:
        return {}


def _attention_choices() -> list:
    registry = _attention_registry()
    ordered = [n for n in _ATTENTION_LABELS if n in registry]
    ordered += [n for n in sorted(registry) if n not in _ATTENTION_LABELS]
    return [_ATTENTION_DEFAULT] + [_ATTENTION_LABELS.get(n, n) for n in ordered]


class LumosAttentionBackend:
    """Swap the attention kernel for one model, without touching the rest.

    Attention dominates the cost of a video model: H3 attends over every frame
    at once, so the kernel choice moves wall-clock far more than it would for a
    still image. ComfyUI can select one globally with `--use-ck-attention`, but
    that is a launch flag applying to every workflow on the server. Patching
    the model object instead keeps the choice inside the graph, where it can be
    changed per run and cannot surprise anything else.

    Core ships `ModelAttentionBackend`, which does the same patching but offers
    only pytorch and comfy kitchen. This lists whatever is registered — sage,
    flash and xformers included when they are installed — so there is no need
    for a separate pack just to reach the other kernels.

    Note that comfy kitchen's kernel is *int8*: it quantizes the attention
    computation. It is the fastest option here and the one worth trying first,
    but it is lossy in a way pytorch and sage are not, so compare a render
    before committing to it.

    Selecting the default leaves the model untouched and is always safe.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "attention": (_attention_choices(),),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "OrbitSheets"

    @classmethod
    def VALIDATE_INPUTS(cls, attention):
        # A graph saved on a box with sage must still load on one without it,
        # rather than failing validation before the user can change the widget.
        return True

    def patch(self, model, attention):
        if attention == _ATTENTION_DEFAULT:
            return (model,)

        registry = _attention_registry()
        wanted = next(
            (name for name, label in _ATTENTION_LABELS.items()
             if label == attention and name in registry),
            attention if attention in registry else None,
        )
        if wanted is None:
            logging.warning(
                "[OrbitSheets] attention backend %r is not available here; "
                "leaving the model on its default kernel.", attention,
            )
            return (model,)

        patched = model.clone()
        try:
            patched.set_model_optimized_attention(registry[wanted])
        except Exception as exc:
            logging.warning(
                "[OrbitSheets] this ComfyUI cannot patch attention (%s); "
                "leaving the model unchanged.", exc,
            )
            return (model,)
        logging.info("[OrbitSheets] attention backend: %s", wanted)
        return (patched,)


# ------------------------------------------------------------------- nodes


class LumosOrbitPrompt:
    """Write H3's prompt for a location sheet, by rotation or by hard cuts.

    A location is not a character, and the turnaround structure that makes the
    character sheet work is the wrong shape for it. A cut asks the model to
    re-establish the subject from a camera position it has never seen. For a
    figure on a plain backdrop that is easy — the model knows what a back looks
    like. For a specific building it is not: told to cut to "the rear", with no
    idea what this rear looks like, the model either re-frames the view it
    already has (the sheet fills with the same facade three times) or invents
    somewhere else entirely and wanders inside the building.

    Rotating the camera is a different matter, and the one move H3 does well
    here: every frame overlaps the last, so nothing has to be invented, only
    continued. The first version of this node rotated and produced genuinely
    different directions — its failure was that the frames tumbled and the
    horizon rolled, which is a stability problem, not a coverage one.

    That tumble had a cause, and it was arithmetic. A complete 360 asked of a
    124-frame take is 70 degrees a second, against the 24 deg/s of H3's own
    worked example ("one full rotation across fifteen seconds"), and asking
    for it "with large amplitude at fast speed" pushed it further still. No
    camera turns that fast and stays level, so the model rendered the move it
    could render at that rate: a roll, tilting up into the vault. `_fit_turn`
    is the answer to that: no angle is ever written into a prompt that the
    take, at `take_seconds`, cannot hold. Asking for more clamps and says so.

    The wording it replaced made the same failure worse from the other end.
    "The camera stays level ... and never tilts up, tilts down, rolls or
    leans" names four of H3's documented motion types — Tilt Up, Tilt Down,
    Roll Clockwise, Roll Counterclockwise — in a model whose guide asks for
    what *does* happen, not what does not. The stability constraint is now
    positive: the horizon stays level, the verticals stay vertical.

    None of which was enough, and the honest verdict on a cathedral interior
    is that no continuous move beat cuts. A pan rolled at every rate it was
    tried at; a translational glide held level but only ever went right, and
    six frames of the same wall sliding past is not a location sheet. Locked
    tripod frames have no camera motion to get wrong, and that is what won.

    Hence `coverage`:

      * "cut views" (default) — the locked-off shots below. The tumbling is
        structurally impossible here: every shot is a static frame.
      * "continuous move" — one unbroken take, the camera panning on the spot
        (interior) or arcing round the subject (exterior). Kept because it is
        the only thing that works on a location the model cannot extrapolate,
        and because Frame Select's view clustering can pull distinct frames
        out of a take that a cut list would have had to invent.

          [Shot 1] the front, straight on     (the anchor, = the first frame)
          [Shot 2] the right side, 90 degrees round
          [Shot 3] the rear, from directly behind
          [Shot 4] the left side, 90 degrees the other way
          [Shot 5] a wide view of the whole place
          [Shot 6] a close look at the main feature

    `space` decides what the continuous move is: inside, the camera pivots on
    its own axis to reach the wall behind it; outside, it travels round the
    subject and is told where it stays, not where it may not go.

    It needs telling that nothing moves, too: left to itself the model animates
    flags, water and passers-by, and a reference sheet wants none of that.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "location_description": ("STRING", {"multiline": True, "default": ""}),
                "visual_style": ("STRING", {"default": "Cinematic, live-action"}),
                # Decides move-around versus shoot-across. Getting this wrong
                # is what makes an interior sheet show one wall four times.
                "space": (["interior", "exterior"],),
                # Cuts first, because cuts are what won. Six locked-off
                # tripod frames cannot roll, and on a cathedral interior they
                # beat every continuous move tried against them.
                "coverage": (["cut views", "continuous move"],),
                # How far the camera comes round on the continuous move: the
                # exterior arc, or the interior pan. A turn the take cannot
                # hold is clamped, not obeyed.
                "rotation": (list(_ROTATION_CHOICES),),
            },
            "optional": {
                # The take this prompt is written for, which has to match the
                # `length` on MiniMaxH3ImageToVideo: 124 frames is 5.17s at
                # 24fps, 260 frames is 10.8s. It sets the cut times, and it is
                # what the rotation is rationed against — a full turn wants
                # the longer take, and is cut down to fit if it does not get
                # one.
                "take_seconds": ("FLOAT", {"default": SHEET_SECONDS, "min": 1.0, "max": 30.0, "step": 0.1}),
                # The fifth shot. Worth having on a big or irregular location
                # where four orthogonal views miss how the parts sit together,
                # and worth dropping on a small one, where it is a duplicate.
                "wide_establishing_shot": ("BOOLEAN", {"default": True}),
                # The sixth: the closest thing the sheet has to a detail
                # reference, since every other view is framed at the same
                # distance and reads the place only as a shape.
                "detail_shot": ("BOOLEAN", {"default": True}),
                # Length of each shot but the last, which runs to the end of
                # the take — six shorter shots fit the same 124 frames.
                "shot_seconds": ("FLOAT", {"default": 0.75, "min": 0.25, "max": 2.0, "step": 0.05}),
                "time_of_day": ("STRING", {"default": ""}),
                "ambient_sound": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "soundscape", "music")
    FUNCTION = "build"
    CATEGORY = "OrbitSheets"

    def build(self, location_description, visual_style, space,
              coverage="cut views",
              rotation="auto (as far as the take allows)",
              take_seconds=SHEET_SECONDS, wide_establishing_shot=True,
              detail_shot=True, shot_seconds=0.75, time_of_day="",
              ambient_sound=""):
        description = location_description.strip().rstrip(".")
        style = visual_style.strip().rstrip(".") or "Cinematic, live-action"
        subject = description or "the location"

        when = time_of_day.strip().rstrip(".")
        when_clause = (
            f" The time of day is {when} and it does not change." if when else ""
        )

        # Repeated on every shot. The tumbling frames were the model reading a
        # moving camera as licence to tilt and roll, so each view is nailed
        # down as a tripod frame rather than a moment in a move.
        locked = (
            "a locked-off static camera at eye level, the horizon level and "
            "centred, no camera movement of any kind"
        )

        if space == "interior":
            # Inside there is nothing to walk around, so the camera crosses to
            # the far side and shoots back — that is what reveals the wall
            # behind the opening frame.
            views = [
                ("the front wall of the space straight on, shot square from "
                 "the middle of the room"),
                ("the right-hand wall of the space straight on, the camera "
                 "turned 90 degrees from the opening view"),
                ("the rear wall of the space straight on — the wall directly "
                 "behind the opening frame — the camera now facing the exact "
                 "opposite direction to Shot 1, so that none of the wall seen "
                 "in Shot 1 appears anywhere in this frame"),
                ("the left-hand wall of the space straight on, the camera "
                 "turned 90 degrees the other way"),
            ]
            overview = (
                "a wide corner view taking in the whole space at once, shot "
                "from one corner across to the far corner, the far walls and "
                "the full width of the floor all inside the frame, showing "
                "how the walls, floor and ceiling meet. This is the widest "
                "shot of the sequence"
            )
            detail = (
                "a tight close-up of the surface and construction of the "
                "space's main feature, the camera close enough that the "
                "material fills the frame and no wall, floor or sky is "
                "visible behind it. This is by far the closest shot of the "
                "sequence and looks nothing like the wide views before it"
            )
        else:
            views = [
                ("the front of the location straight on, the facade square to "
                 "the camera and entirely inside the frame"),
                ("the right side of the location straight on, the camera moved "
                 "90 degrees round, the full side elevation in frame"),
                ("the rear of the location straight on, shot from directly "
                 "behind — the side hidden in the opening frame — the camera "
                 "now facing the exact opposite direction to Shot 1, so that "
                 "none of the facade seen in Shot 1 appears anywhere in this "
                 "frame, the full rear elevation filling it instead"),
                ("the left side of the location straight on, the camera moved "
                 "90 degrees round the other way, the full side elevation in "
                 "frame"),
            ]
            overview = (
                "a wide three-quarter establishing view of the whole location "
                "from further back, front and one side both visible, the "
                "entire place inside the frame with generous margin"
            )
            detail = (
                "a tight close-up of the main entrance and the wall surface "
                "around it, the camera close enough that the doorway and its "
                "masonry fill the frame with no sky and no surrounding "
                "ground visible. This is by far the closest shot of the "
                "sequence and looks nothing like the wide views before it"
            )

        if coverage.startswith("continuous"):
            # Deliberately NOT wrapped in the I2VA <Picture 1> envelope that
            # the cut path uses. That envelope was found to change the camera
            # behaviour on this node long before the cut rewrite, and carrying
            # it into the rotation path is what brought the rolling back: a
            # cathedral interior pan came out as a spin under the vault. The
            # plain style-subject-motion sentence is what produces a clean
            # rotation, and the move is stated in H3's own camera terms —
            # one move, its direction, its speed, and what stays stable.
            degrees, turn_phrase, clamped = _fit_turn(rotation, take_seconds)
            if clamped:
                logging.warning("[OrbitSheets] %s", clamped)

            # What the turn is *for*, said as the thing that comes into view.
            if degrees >= 360.0:
                reveal = (
                    "Every wall comes into view in turn, including the wall "
                    "directly behind the opening frame, and the turn carries "
                    "all the way round to where it started."
                )
            elif degrees >= 180.0:
                reveal = (
                    "The wall directly behind the opening frame comes fully "
                    "into view, and the take ends facing it."
                )
            else:
                reveal = (
                    "The take ends facing the wall to the right of the "
                    "opening frame, square on."
                )

            # Positive throughout: H3's guide asks for what happens, and the
            # wording this replaced listed the model's own Tilt Up / Tilt Down
            # / Roll motion types under a "never" it does not reliably read.
            steady = (
                "The horizon stays level and the vertical lines of the walls "
                "and pillars stay vertical from the first frame to the last. "
                "The camera turns on its own axis alone, at one constant "
                "height, the floor along the bottom of the frame and the "
                "ceiling along the top exactly as in the opening frame."
            )

            if space == "interior":
                motion = (
                    "The camera holds its position in the middle of the space "
                    "and pans right at slow speed, one single continuous "
                    f"unbroken take turning steadily through {turn_phrase} "
                    f"across the whole take and stopping there. {reveal}"
                )
            else:
                motion = (
                    "The camera performs an arc shot around the location at "
                    "slow speed, one single continuous unbroken take travelling "
                    f"{turn_phrase} around it at a constant radius. "
                    + ("The far side of the location — the side hidden in the "
                       "opening frame — is fully revealed before the shot "
                       "ends. " if degrees >= 180.0 else
                       "The right-hand side of the location is brought fully "
                       "into view before the shot ends. ")
                    + "The camera stays outside at ground level for the whole "
                    "take, at the same eye-level height and the same distance "
                    "from the building in every frame."
                )
                steady = (
                    "The horizon stays level and the vertical lines of the "
                    "walls stay vertical from the first frame to the last."
                )

            prompt = (
                f"{style}. {subject}. {motion}"
                f"{when_clause} {steady} "
                "The location is completely empty: no people, no animals and "
                "no vehicles are present, and nothing within the environment "
                "moves. Lighting, weather and atmosphere stay exactly as "
                "established. Architecture, materials, colours, and the "
                "position of every object remain identical from every angle. "
                "A single continuous take with no cuts, no transitions, no "
                "on-screen text and no titles."
            )
            return (prompt, ambient_sound.strip() or "N/A", "N/A")

        if wide_establishing_shot:
            views = views + [overview]
        if detail_shot:
            views = views + [detail]

        # Same timing rule as the turnaround: each shot but the last runs
        # `shot_seconds`, the last keeps the remainder, so the shot count can
        # change without the video needing more frames. Clamped against the
        # take the user actually rendered, not a fixed five seconds, so a
        # longer `length` spreads the cuts instead of stacking them all in
        # the opening half.
        span = max(float(take_seconds), 1.0)
        step = max(0.25, min(float(shot_seconds), span / len(views)))
        at = [_timecode(i * step) for i in range(len(views))]

        shots = [
            f"[Shot 1] {style}, {views[0]}, {locked}.{when_clause}"
        ]
        for index, view in enumerate(views[1:], start=2):
            shots.append(
                f" [Shot {index}] At {at[index - 1]}, the shot cuts to "
                f"{view}, {locked}."
            )

        body = (
            "".join(shots)
            + " Every shot is a different framing of the place and no two "
            "shots repeat the same view: the camera position, direction and "
            "distance are visibly different in each one. "
            "The location is completely empty: no people, no animals and no "
            "vehicles are present, and nothing within the environment moves. "
            "Lighting, weather and atmosphere stay exactly as established and "
            "identical in every shot. Architecture, materials, colours, and the "
            "position of every object remain identical from every angle — it is "
            "the same place seen from a different side each time. No on-screen "
            "text and no titles."
        )

        ambient = ambient_sound.strip()
        soundscape = ambient if ambient else "N/A"

        # I2VA per H3's prompt guide: first-frame instruction first, then the
        # three core fields — the same envelope the character sheet uses, which
        # is what anchors Shot 1 to the image the sheet starts from.
        prompt = (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.\n\n"
            f"integrated_multimodal_description: {subject}. {body}\n\n"
            f"overall_soundscape: {soundscape}\n\n"
            "non_diegetic_music: N/A"
        )

        return (prompt, soundscape, "N/A")


class LumosCharacterTurnaroundPrompt:
    """Write H3's prompt for a 6-shot character turnaround with hard cuts.

    The reference sheet is one I2VA sequence of six distinct views, each about
    a second long, joined by hard cuts per H3's prompt guide:

      [Shot 1] full body, facing the camera   (the anchor, = the first frame)
      [Shot 2] tight close-up of the face
      [Shot 3] left side profile
      [Shot 4] right side profile
      [Shot 5] rear view of the body
      [Shot 6] the face again, frightened     (optional, `scared_shot`)

    Shot 6 earns its slot by being the one thing the other five cannot show:
    they are all deliberately neutral, and a neutral face is no use as a
    reaction reference. Six shots need six seconds, so `length` wants 158
    frames (~6.6s at 24fps) rather than the 124 a five-shot sheet uses.

    Cuts, not a continuous orbit: a cut forces the model to re-establish the
    figure at each angle, which is exactly what a turnaround sheet needs, and
    the identity stays locked because every shot reuses the same description.
    The first frame from the image model is the full body, so the identity is
    anchored before the face close-up arrives.

    Framing is its own lever because a 16:9 frame crops wide subjects: a dragon
    with spread wings, a tail or a held weapon outstretches the shot, and H3
    will cut off whatever the prompt did not insist stays inside it. The
    default framing demands generous empty margin on every full-body shot.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_description": ("STRING", {"multiline": True, "default": ""}),
                "visual_style": ("STRING", {"default": "Cinematic, live-action"}),
            },
            "optional": {
                "backdrop": ("STRING", {"default": "plain seamless neutral grey studio backdrop"}),
                # Framing + margin. Generous margin is the default because
                # reference sheets fail on wide subjects (wings, tails) when the
                # frame crops them.
                "framing": (_FRAMINGS,),
                # H3 renders audio alongside video from the same latent, so a
                # spoken line in the opening shot costs no extra sampling and
                # yields a voice-timbre sample for the story's <Audio N> slot.
                "spoken_line": ("STRING", {"multiline": True, "default": ""}),
                "voice_description": ("STRING", {"default": "", "multiline": True}),
                "language": ("STRING", {"default": "English"}),
                "silent_during_closeup": ("BOOLEAN", {"default": True}),
                # Length of each shot but the last, which runs to the end of
                # the take. 0.75 fits six shots inside the same 124 frames a
                # five-shot sheet used, and leaves the final shot the longest.
                "shot_seconds": ("FLOAT", {"default": 0.75, "min": 0.25, "max": 2.0, "step": 0.05}),
                # A sixth shot: the same face, frightened. An expression
                # reference is worth a slot because it is the one thing the
                # other five shots cannot show — they are all deliberately
                # neutral, and a neutral face is no use for casting a reaction.
                "scared_shot": ("BOOLEAN", {"default": True}),
                "ambient_sound": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "soundscape", "music")
    FUNCTION = "build"
    CATEGORY = "OrbitSheets"

    def build(self, character_description, visual_style,
              backdrop="plain seamless neutral grey studio backdrop",
              framing="full body, generous margin",
              spoken_line="", voice_description="", language="English",
              silent_during_closeup=True, scared_shot=True, shot_seconds=0.75,
              ambient_sound=""):
        description = character_description.strip().rstrip(".")
        style = visual_style.strip().rstrip(".") or "Cinematic, live-action"
        subject = description or "the character"
        set_dressing = backdrop.strip().rstrip(".") or "plain seamless neutral grey studio backdrop"
        # It opens a sentence, so it has to read like one.
        set_dressing = set_dressing[0].upper() + set_dressing[1:]
        margin = _FRAMING_CLAUSES.get(framing, _FRAMING_CLAUSES[_FRAMINGS[0]])

        line = spoken_line.strip()
        speaks = bool(line)

        if speaks:
            voice = voice_description.strip().rstrip(".")
            voice_clause = f" The voice is {voice}." if voice else ""
            # The guide keeps the line's own punctuation inside <d>, so only
            # add a full stop when the line does not already end in one.
            tail = "" if line[-1] in ".!?…\"'" else "."
            speech = (
                f" The figure (S1) speaks calmly and directly to camera, saying "
                f"<d>[{language}] {line}</d>{tail}{voice_clause}"
            )
            # Speaking is motion; the stillness rule has to allow for it or the
            # two instructions contradict and the model picks one at random.
            stillness = (
                " Apart from the movement of speech, the figure stands still in "
                "a neutral upright pose with arms relaxed at the sides, and does "
                "not move, walk, gesture or turn at any point."
            )
            ambient = ambient_sound.strip() or (
                "A quiet interior studio room tone, with the figure's voice "
                "close-miked and clear. No other sound."
            )
        else:
            speech = ""
            stillness = (
                " The figure stands still in a neutral upright pose with arms "
                "relaxed at the sides, and does not move, walk, gesture or "
                "turn at any point" + (
                    ", and holds a neutral expression until the final shot."
                    if scared_shot else
                    ", or change expression at any point."
                )
            )
            ambient = ambient_sound.strip() or "N/A"

        # The mouth stays closed on every shot except the opening one, so the
        # face close-up and profiles come out clean.
        mouth = (
            ", the mouth closed and a still, neutral expression"
            if silent_during_closeup else
            " with a still, neutral expression"
        )

        # The expression reference. It has to name the face's features rather
        # than the emotion alone — "scared" on its own gets read as a whole
        # performance, and the figure starts flinching and turning away, which
        # breaks the locked framing every other shot depends on.
        # The turnaround views are static, so they need no more than a moment
        # each: the first shots run `shot_seconds` and the last one takes
        # whatever is left of the take. Adding the expression shot therefore
        # costs no extra frames — it just eats into the final shot's slack —
        # and the expression shot is the one worth the extra length anyway,
        # being the only one where the face has to settle into something.
        total_shots = 6 if scared_shot else 5
        step = max(0.25, min(float(shot_seconds), SHEET_SECONDS / total_shots))
        at = [_timecode(i * step) for i in range(total_shots)]

        scared = (
            f" [Shot 6] At {at[5]}, the shot cuts to a medium close-up of the "
            "face and shoulders, still facing the camera, the expression now "
            "frightened: eyes wide and brows raised and drawn together, mouth "
            "slightly open, the head held still and upright. Only the "
            "expression changes — the figure does not flinch, recoil, turn "
            "away or move, and the framing and lighting stay as before."
        ) if scared_shot else ""

        body = (
            f"[Shot 1] {style}, a full-body shot frames {subject}, the entire "
            f"figure visible from head to toe, facing the camera.{margin}"
            f"{speech}{stillness}"
            f" [Shot 2] At {at[1]}, the shot cuts to a tight close-up of the "
            f"face, framed from the top of the hair to the chin, still facing "
            f"the camera{mouth}."
            f" [Shot 3] At {at[2]}, the shot cuts to a left side profile of "
            f"the full figure, the whole body visible from head to toe with "
            f"the same framing margin, the head turned to show the left "
            f"profile{mouth}."
            f" [Shot 4] At {at[3]}, the shot cuts to a right side profile of "
            f"the full figure, the whole body visible from head to toe with "
            f"the same framing margin, the head turned to show the right "
            f"profile{mouth}."
            f" [Shot 5] At {at[4]}, the shot cuts to a rear view of the full "
            f"figure, the back of the body visible from head to toe with the "
            f"same framing margin{mouth}."
            f"{scared}"
            f" {set_dressing}, even neutral lighting from every side, no props "
            "and no cast shadows. Clothing, hair, colours, proportions and "
            "every visible detail remain identical across every shot."
        )

        # I2VA per H3's prompt guide: first-frame instruction first, then the
        # three core fields, matching what the app's compiler emits.
        prompt = (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.\n\n"
            f"integrated_multimodal_description: {body}\n\n"
            f"overall_soundscape: {ambient}\n\n"
            "non_diegetic_music: N/A"
        )
        return (prompt, ambient, "N/A")


class LumosFrameSelect:
    """Choose the frames from an orbit that are worth keeping.

    Even spacing is the obvious approach and the wrong one: an orbit is not
    uniform, and evenly-spaced samples land on blurred mid-swing frames and on
    pairs that show the same wall twice. This shortlists by time, gates out the
    soft frames, then has a vision model compare the survivors as one image and
    say which views actually differ and which are sharp.

    The judging model is whichever one is available, in order of preference:
    an in-graph vision-language CLIP connected to `clip` (the same Qwen3-VL
    that encoded the anchor — no external server, no second model resident),
    then an OpenAI-compatible HTTP endpoint, then plain sharpness-and-spread
    scoring when neither model is reachable. The report in `info` names the
    path that actually ran.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "count": ("INT", {"default": 6, "min": 1, "max": 24}),
                "mode": (["vision_llm", "sharpness_diversity"],),
            },
            "optional": {
                # An in-graph vision-language CLIP — the same Qwen3-VL that
                # encoded the anchor. When connected, the node judges the
                # montage itself instead of calling an HTTP server.
                "clip": ("CLIP",),
                "candidates": ("INT", {"default": DEFAULT_CANDIDATES, "min": 4, "max": 32}),
                "keep_first_frame": ("BOOLEAN", {"default": True}),
                "subject_hint": ("STRING", {"default": "", "multiline": True}),
                # Lets a workflow ask for a *mix* rather than just "distinct":
                # a character sheet wants one tight face plus full-body angles,
                # which no generic instruction would produce.
                "selection_brief": ("STRING", {"default": "", "multiline": True}),
                "sharpness_weight": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "free_vram_first": ("BOOLEAN", {"default": True}),
                "llm_url": ("STRING", {"default": ""}),
                "timeout_seconds": ("INT", {"default": 180, "min": 10, "max": 1800}),
                # Only sharp frames are shown to the vision model: the
                # Laplacian already found the mid-swing blurs, so the model's
                # judgement goes to framing and angle instead.
                "sharpness_gate": ("BOOLEAN", {"default": True}),
                "max_length": ("INT", {"default": 400, "min": 64, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05}),
                # A hard-cut turnaround is a known sequence of static views, so
                # set this to the shot count to force one sharp frame per shot
                # — every view (the face close-up included) is then guaranteed
                # on the sheet no matter what the vision model does. 1 = off.
                "shots": ("INT", {"default": 1, "min": 1, "max": 32}),
                # Judge the move through several readable numbered montages
                # instead of one crowded board: each board covers a time sector
                # at a readable tile size, so every part of the orbit is seen.
                # 1 = single montage (the old behaviour).
                "boards": ("INT", {"default": 1, "min": 1, "max": 16}),
                # How the `shots` groups are formed. "views" groups frames by
                # what they show, which is the only one that cannot hand the
                # sheet the same angle twice — H3 gives its shots wildly
                # different lengths from run to run, so any split by time
                # eventually lands two slices on one view and drops another.
                "shot_split": (["views (by content)", "cuts (detected)",
                                "even (by time)"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "info")
    FUNCTION = "select"
    CATEGORY = "OrbitSheets"

    def select(self, images, count, mode, clip=None, candidates=DEFAULT_CANDIDATES,
               keep_first_frame=True, subject_hint="", selection_brief="",
               sharpness_weight=0.35, free_vram_first=True, llm_url="",
               timeout_seconds=180, sharpness_gate=True, max_length=400,
               temperature=0.2, shots=1, boards=1,
               shot_split="views (by content)"):
        total = int(images.shape[0])
        count = max(1, min(int(count), total))
        shots = max(1, min(int(shots), total))
        boards = max(1, min(int(boards), 16))

        if total <= count:
            return (images, f"kept all {total} frames (asked for {count})")

        if mode == "sharpness_diversity":
            picked = _greedy_spread(images, count, sharpness_weight, keep_first_frame)
            return (images[picked], self._report("sharpness+spread", picked, total))

        # Deterministic base for a known shot layout: one sharp frame per equal
        # segment. A hard-cut turnaround is a sequence of distinct static views,
        # so this guarantees every view — the face close-up included — makes the
        # sheet no matter what the vision model does.
        forced: list[int] = []
        found = ""
        if shots >= 2:
            if shot_split.startswith("views"):
                forced, distinct = self._per_view(images, shots)
                found = " (grouped by view)"
                if distinct < len(forced):
                    # Worth saying loudly: the sheet is about to repeat itself,
                    # and the cause is upstream in the video, not here.
                    found += (
                        f" — WARNING: only {distinct} distinct views in these "
                        f"{total} frames, so {len(forced) - distinct} of the "
                        "picks repeat one. The take did not deliver "
                        f"{shots} different shots; re-run it or lower `shots`"
                    )
            else:
                detect = shot_split.startswith("cuts")
                forced, cuts = self._per_shot(images, shots, detect)
                if cuts:
                    found = " (cuts found at %s)" % ", ".join(str(c) for c in cuts)
                else:
                    found = " (no cuts found, even slices)" if detect else " (even slices)"
        remaining = max(0, count - len(forced))

        if remaining == 0:
            return (
                images[sorted(forced)],
                self._report(f"per-shot{found} (no vision)", sorted(forced), total),
            )

        if free_vram_first:
            # Hand the card back before any judging runs: H3 is still resident,
            # and a vision model landing on top of it is how the orbit succeeds
            # and the judging OOMs.
            self._free_vram()

        note = ""
        picks: list[int] | None = None

        if boards >= 2:
            # Multi-board scan: every sector of the move gets a readable,
            # numbered board and contributes its best frames, so no part of the
            # orbit is invisible to the model the way a single board is.
            per_board = max(1, (remaining + boards - 1) // boards)
            picks, note = self._board_scan(
                images, boards, per_board, clip, subject_hint, selection_brief,
                llm_url, int(timeout_seconds), int(max_length), float(temperature),
            )
            if not picks:
                picks = None
        else:
            # Single montage of the sharpest shortlist.
            shortlist = self._shortlist(total, max(remaining, int(candidates)), keep_first_frame)
            if sharpness_gate:
                shortlist = self._gate_sharp(images, shortlist, max(6, remaining * 2))
            pils = _tensor_to_pils(images[shortlist])
            if clip is not None:
                # The workflow carries its own vision model; never reach for an
                # external server. If the CLIP fails, fall back to scoring.
                try:
                    text, note = _clip_pick(
                        clip, pils, remaining, subject_hint, selection_brief,
                        int(max_length), float(temperature),
                    )
                    picks = _parse_picks(text, len(pils))
                except Exception as exc:
                    note = f"in-graph CLIP failed ({type(exc).__name__})"
                    picks = None
            else:
                picks, note = _vlm_pick(
                    pils, remaining, subject_hint, _llm_base_url(llm_url),
                    int(timeout_seconds), selection_brief,
                )
            if picks:
                picks = [shortlist[i] for i in picks]

        # Merge: anchor frame first, then the forced per-shot views, then the
        # vision picks, capped at count.
        ordered: list[int] = []
        if keep_first_frame and total > 1:
            ordered.append(0)
        ordered += sorted(forced)
        ordered += picks or []

        picked_list: list[int] = []
        for index in ordered:
            if index not in picked_list:
                picked_list.append(index)
            if len(picked_list) >= count:
                break
        # A model that under-returned still gave us its good ones; fill the
        # remainder by spread rather than discarding its judgement.
        if len(picked_list) < count:
            for index in _greedy_spread(images, count, sharpness_weight, keep_first_frame):
                if index not in picked_list:
                    picked_list.append(index)
                if len(picked_list) >= count:
                    break
        picked = sorted(picked_list)

        if picks:
            if forced:
                method = (f"per-shot{found} + multi-board" if boards >= 2
                          else f"per-shot{found} + vision model")
            else:
                method = f"{'multi-board' if boards >= 2 else 'vision model'} — {note}"
        elif forced:
            method = (f"per-shot{found} + spread (fallback: {note})" if note
                      else f"per-shot{found} + spread")
        else:
            method = f"sharpness+spread (fallback: {note})" if note else "sharpness+spread"
        return (images[picked], self._report(method, picked, total))

    def _judge(self, pils, count, clip, hint, brief, llm_url, timeout,
               max_length, temperature):
        """Ask whichever vision path is wired to pick `count` frames from `pils`."""
        if clip is not None:
            try:
                text, note = _clip_pick(clip, pils, count, hint, brief,
                                        int(max_length), float(temperature))
                return _parse_picks(text, len(pils)), note
            except Exception as exc:
                return None, f"in-graph CLIP failed ({type(exc).__name__})"
        return _vlm_pick(pils, count, hint, _llm_base_url(llm_url),
                         int(timeout), brief)

    def _board_scan(self, images, boards, per_board, clip, hint, brief,
                    llm_url, timeout, max_length, temperature):
        """Judge the whole timeline through several readable montages.

        One numbered board per time sector, each capped at 16 sharp tiles so
        every tile stays readable, then one vision call per board. Together the
        boards cover the full move and every sector contributes picks, so no
        part of the orbit is invisible to the model the way a single board is.
        """
        total = images.shape[0]
        sharp = _sharpness(images)
        ordered_picks: list[int] = []
        notes: list[str] = []
        for b in range(boards):
            start = int(round(b * total / boards))
            end = int(round((b + 1) * total / boards))
            bucket = list(range(start, end))
            if len(bucket) < 2:
                continue
            gate_n = min(16, len(bucket))
            order = sorted(bucket, key=lambda i: float(sharp[i]), reverse=True)
            keep = sorted(order[:gate_n])
            pils = _tensor_to_pils(images[keep])
            board_picks, note = self._judge(
                pils, per_board, clip, hint, brief,
                llm_url, timeout, max_length, temperature,
            )
            notes.append(note)
            if board_picks:
                ordered_picks.extend(keep[i] for i in board_picks[:per_board])
        seen, out = set(), []
        for index in ordered_picks:
            if index not in seen:
                seen.add(index)
                out.append(index)
        return out, "; ".join(n for n in notes if n)

    @staticmethod
    def _segments(images, shots, detect=True):
        """Split the timeline into `shots` segments, one per shot.

        Equal time slices are the obvious approach and the wrong one: they
        assume the model cut at exactly the seconds the prompt asked for, and
        it does not. A shot that runs long pushes the next one past its slice
        boundary, so one slice samples the previous view a second time and a
        whole angle — usually a profile — never reaches the sheet.

        So find the cuts instead of guessing them. A hard cut changes most of
        the frame at once, which is a large spike in consecutive-frame distance
        against an otherwise near-still shot; the `shots - 1` biggest spikes,
        kept apart by a minimum shot length, are the cuts. Falls back to equal
        slices when the spikes are not there to be found (a continuous move, or
        a video that genuinely never cut).
        """
        total = int(images.shape[0])
        even = [
            (int(round(i * total / shots)), int(round((i + 1) * total / shots)))
            for i in range(shots)
        ]
        if not detect or shots < 2 or total < shots * 3:
            return even, []

        # Brightness kept in (no contrast normalising): a cut to a different
        # view changes exposure and layout together, and both are signal here.
        gray = images.mean(dim=3).unsqueeze(1)
        small = F.interpolate(gray, size=(32, 32), mode="area").view(total, -1)

        min_shot = max(2, total // (shots * 3))

        # Compare a window before the boundary against a window after it, not
        # one frame against the next. A single blurred frame — and the move has
        # several — spikes an adjacent-frame difference exactly as hard as a cut
        # does, but it does not change what comes after it. A cut does, and only
        # a cut still shows across the gap once the blurred frame is averaged in
        # with its neighbours.
        window = max(2, min(4, min_shot))
        delta = torch.zeros(total)
        for b in range(window, total - window):
            before = small[b - window:b].mean(dim=0)
            after = small[b:b + window].mean(dim=0)
            delta[b] = (before - after).abs().mean()

        # A cut has to stand out from the shot's own noise, not merely be the
        # largest number present — otherwise a still video invents cuts.
        # Measured against the median absolute deviation, not the standard
        # deviation: the cuts are themselves the biggest numbers in the series,
        # so they inflate a standard deviation enough to hide the smallest real
        # cut behind it. A MAD ignores them, and the gap it exposes is not
        # marginal — real cuts land three orders of magnitude above the noise,
        # so the exact threshold below barely matters.
        core = delta[window:total - window]
        median = float(core.median())
        mad = float((core - core.median()).abs().median()) * 1.4826 + 1e-9

        ranked = sorted(range(total), key=lambda i: float(delta[i]), reverse=True)
        cuts: list[int] = []
        for boundary in ranked:
            if len(cuts) >= shots - 1:
                break
            if (float(delta[boundary]) - median) / mad < 20.0:
                break
            if boundary < min_shot or total - boundary < min_shot:
                continue
            if any(abs(boundary - c) < min_shot for c in cuts):
                continue
            cuts.append(boundary)

        if len(cuts) != shots - 1:
            # Partial detection is worse than none: it would leave two real
            # views sharing one segment while splitting another in half.
            return even, sorted(cuts)

        cuts.sort()
        bounds = [0] + cuts + [total]
        return list(zip(bounds, bounds[1:])), cuts

    @staticmethod
    def _per_view(images, views):
        """One sharp frame per distinct view, found by appearance not by time.

        The sheet's real requirement is `views` frames that each show something
        different, and clustering states exactly that. Within a cluster the
        sharpest frame wins, which is the same rule the per-shot path uses —
        the only change is how the group was formed.

        Asking for k clusters, though, always yields k of them, whether or not
        the video holds that many different views. When a take under-delivers —
        the model lingers on one subject for three of its shots instead of
        moving on — k-means splits that one long view into slices, and the
        sheet fills with the same picture three times over. That is not a
        selection failure and no picker can fix it, so it is measured and
        reported instead: representatives closer to each other than a fraction
        of the sheet's own spread are counted as one view, and the caller is
        told how many genuinely distinct views the video actually contained.

        Returns (frame indices, distinct view count).
        """
        sharp = _sharpness(images)
        groups = [g for g in _cluster_views(images, views) if g]
        picks = [max(group, key=lambda f: float(sharp[f])) for group in groups]
        picks = sorted(set(picks))
        if len(picks) < 2:
            return picks, len(picks)

        # Scale-free threshold: near-duplicate relative to how far apart this
        # sheet's own views are, since absolute distances mean nothing across
        # a studio backdrop and a night exterior.
        desc = _view_descriptors(images)[picks]
        gaps = torch.cdist(desc, desc)
        span = float(gaps.max())
        tau = 0.35 * span

        # Biggest clusters first: a view the video actually dwelt on is the
        # real one, and the slivers split off it are the duplicates.
        order = sorted(range(len(picks)), key=lambda i: -len(groups[i]))
        distinct: list[int] = []
        for i in order:
            if all(float(gaps[i, j]) >= tau for j in distinct):
                distinct.append(i)

        return picks, len(distinct)

    @staticmethod
    def _per_shot(images, shots, detect=True):
        """One sharp frame per shot.

        For a hard-cut turnaround every shot is a distinct static view, so the
        sharpest frame of each shot's central region is its best
        representative — the sheet's full set of views is guaranteed without
        trusting a vision model to find them. The central region skips the
        frames near each cut, which can be a cross-blend of two shots.
        """
        sharp = _sharpness(images)
        segments, cuts = LumosFrameSelect._segments(images, shots, detect)
        out = []
        for start, end in segments:
            span = max(1, end - start)
            c0 = start + span // 5
            c1 = start + 4 * span // 5
            candidates = list(range(c0, max(c0 + 1, c1)))
            out.append(max(candidates, key=lambda f: float(sharp[f])))
        return sorted(set(out)), cuts

    @staticmethod
    def _gate_sharp(images, indices, keep_n):
        """Keep only the sharpest frames of a shortlist, at most keep_n.

        An orbit spends part of its arc mid-motion, and those frames are soft.
        Showing them to the vision model wastes its judgement; filtering them
        out first is exactly the kind of mechanical check a Laplacian does
        better than a model.
        """
        if len(indices) <= keep_n:
            return indices
        sharp = _sharpness(images)
        order = sorted(indices, key=lambda i: float(sharp[i]), reverse=True)
        return sorted(order[:keep_n])

    @staticmethod
    def _shortlist(total: int, wanted: int, keep_first: bool) -> list[int]:
        wanted = max(1, min(wanted, total))
        indices = np.linspace(0, total - 1, wanted).round().astype(int).tolist()
        if keep_first:
            indices[0] = 0
        return sorted(dict.fromkeys(indices))

    @staticmethod
    def _free_vram():
        """Hand VRAM back before the vision model is asked for anything.

        H3 is still resident when this runs, and a 30B vision model landing on
        top of it is how the orbit succeeds and the judging OOMs.
        """
        try:
            import comfy.model_management

            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
        except Exception as exc:  # pragma: no cover - defensive
            logging.debug("[LumosFrameSelect] could not free VRAM: %s", exc)

    @staticmethod
    def _report(method: str, picked: list[int], total: int) -> str:
        return f"{len(picked)}/{total} frames via {method} — indices {picked}"


class LumosContactSheet:
    """Lay frames out as one labelled sheet.

    Kept separate from selection so the sheet can be re-laid out — different
    column count, labels off — without paying for the orbit again.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "columns": ("INT", {"default": 3, "min": 1, "max": 8}),
                "cell_width": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 32}),
            },
            "optional": {
                "padding": ("INT", {"default": 8, "min": 0, "max": 128}),
                "label_frames": ("BOOLEAN", {"default": True}),
                "label_prefix": ("STRING", {"default": "Angle"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("sheet",)
    FUNCTION = "compose"
    CATEGORY = "OrbitSheets"

    def compose(self, images, columns, cell_width, padding=8,
                label_frames=True, label_prefix="Angle"):
        pils = _tensor_to_pils(images)
        prefix = label_prefix.strip()
        labels = (
            [f"{prefix} {i + 1}".strip() for i in range(len(pils))]
            if label_frames else None
        )
        sheet = _montage(pils, columns, int(cell_width), int(padding), labels)
        return (_pils_to_tensor([sheet]),)


# Node ids are namespaced to this pack. ComfyUI registers every pack into one
# global dict with a plain assignment — no duplicate check, no warning — so a
# shared id would let whichever pack happens to load last silently shadow the
# other. Distinct ids let this pack sit alongside any other install of the same
# nodes without either one clobbering the other.
NODE_CLASS_MAPPINGS = {
    "OrbitSheetsLocationPrompt": LumosOrbitPrompt,
    "OrbitSheetsCharacterPrompt": LumosCharacterTurnaroundPrompt,
    "OrbitSheetsFrameSelect": LumosFrameSelect,
    "OrbitSheetsContactSheet": LumosContactSheet,
    "OrbitSheetsAttentionBackend": LumosAttentionBackend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OrbitSheetsLocationPrompt": "Location Sheet Prompt (H3)",
    "OrbitSheetsCharacterPrompt": "Character Turnaround Prompt (H3)",
    "OrbitSheetsFrameSelect": "Frame Select (vision-judged)",
    "OrbitSheetsContactSheet": "Contact Sheet",
    "OrbitSheetsAttentionBackend": "Attention Backend",
}
