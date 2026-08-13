"""Character and location reference sheets from a MiniMax-H3 camera move.

A reference sheet generated angle-by-angle from text drifts: the same courtyard
comes back with a different arch count, the same character with a different
collar. H3 does not drift, because every angle is the *same shot* — one
continuous move around a still subject yields views that genuinely agree with
each other. The remaining job is picking the useful frames, which is what this
pack does.

Four nodes, each covering a step ComfyUI has no answer for:

  * Location Orbit Prompt      writes a location arc-shot prompt to H3's spec
  * Character Turnaround Prompt  same for a figure, ending on the face
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

_CAMERA_AMPLITUDES = ["large amplitude", "small amplitude"]
_CAMERA_SPEEDS = ["slow speed", "fast speed"]


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
    """Write H3's camera-move prompt for a location, to the model's own spec.

    The motion type has to match where the camera stands, and this is the one
    thing that decides whether the sheet is useful. An *arc shot* circles a
    subject from outside it — correct for a building or a monument. Inside a
    room there is nothing to circle, and asking for an arc yields a small
    sideways drift down the same wall: eight frames of one view, with the wall
    behind the camera never seen. Interiors need a *pan*, the camera turning on
    its own axis, which is a separate motion type in H3's vocabulary.

    Either way the instruction has to name a full 360 degrees explicitly.
    "Reveals the space from every side" is a description of an outcome, and the
    model treats it as flavour; "turns through a complete 360-degree rotation"
    is an instruction about the camera, and it follows it.

    H3's guide also asks for motion stated as type + amplitude + speed, a style
    statement opening the shot, and `N/A` where there is deliberately no sound.
    It needs telling that nothing moves, too: left to itself it animates flags,
    water and passers-by, and a reference sheet wants none of that.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "location_description": ("STRING", {"multiline": True, "default": ""}),
                "visual_style": ("STRING", {"default": "Cinematic, live-action"}),
                # Decides pan-in-place versus orbit-around. Getting this wrong
                # is what makes an interior sheet show one wall eight times.
                "space": (["interior", "exterior"],),
                "orbit_direction": (["clockwise", "counterclockwise"],),
                "amplitude": (_CAMERA_AMPLITUDES,),
                "speed": (_CAMERA_SPEEDS,),
            },
            "optional": {
                "time_of_day": ("STRING", {"default": ""}),
                "ambient_sound": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "soundscape", "music")
    FUNCTION = "build"
    CATEGORY = "OrbitSheets"

    def build(self, location_description, visual_style, space, orbit_direction,
              amplitude, speed, time_of_day="", ambient_sound=""):
        description = location_description.strip().rstrip(".")
        style = visual_style.strip().rstrip(".") or "Cinematic, live-action"
        subject = description or "the location"

        when = time_of_day.strip().rstrip(".")
        when_clause = (
            f" The time of day is {when} and it does not change." if when else ""
        )

        # Pan turns the camera; arc carries it around something. Inside a
        # room only the first can reach the wall behind the opening view.
        if space == "interior":
            turn = "left" if orbit_direction == "counterclockwise" else "right"
            motion = (
                f"The camera holds its position in the middle of the space and "
                f"pans {turn} with {amplitude} at {speed}, turning steadily "
                "through a complete 360-degree rotation in one continuous "
                "unbroken take. Every wall comes into view in turn, including "
                "the wall directly behind the opening frame, and the rotation "
                "carries all the way round until it returns to where it started."
            )
        else:
            motion = (
                f"The camera performs an arc shot {orbit_direction} around the "
                f"location, travelling a complete 360-degree circle with "
                f"{amplitude} at {speed} in one continuous unbroken take. The "
                "far side of the location — the side hidden in the opening "
                "frame — is fully revealed before the shot ends."
            )

        prompt = (
            # Stated once, as its own sentence — see the turnaround node.
            f"{style}. {subject}. {motion}"
            f"{when_clause}"
            " The location is completely empty: no people, no animals and no "
            "vehicles are present, and nothing within the environment moves. "
            "Lighting, weather and atmosphere stay exactly as established. "
            "Architecture, materials, colours, and the position of every object "
            "remain identical from every angle. A single continuous take with no "
            "cuts, no transitions, no on-screen text and no titles."
        )

        ambient = ambient_sound.strip()
        soundscape = ambient if ambient else "N/A"
        return (prompt, soundscape, "N/A")


class LumosCharacterTurnaroundPrompt:
    """Write H3's prompt for a character turnaround that ends on the face.

    A location wants a plain orbit; a character sheet wants two framings — the
    full figure from every side, and one tight face. Asking for both in a
    single continuous move keeps them the same person: a separate close-up run
    is a separate generation, which is exactly where identity drifts.

    The order matters. Arc first, push in last: the opening frame is the
    full-body reference, so the body is anchored and the face is pushed into
    rather than invented.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_description": ("STRING", {"multiline": True, "default": ""}),
                "visual_style": ("STRING", {"default": "Cinematic, live-action"}),
                "orbit_direction": (["clockwise", "counterclockwise"],),
                "end_on_closeup": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "backdrop": ("STRING", {"default": "plain seamless neutral grey studio backdrop"}),
                # H3 renders audio alongside video from the same latent, so a
                # spoken line during the turnaround costs no extra sampling and
                # yields a voice-timbre sample for the story's <Audio N> slot.
                "spoken_line": ("STRING", {"multiline": True, "default": ""}),
                "voice_description": ("STRING", {"default": "", "multiline": True}),
                "language": ("STRING", {"default": "English"}),
                "silent_during_closeup": ("BOOLEAN", {"default": True}),
                "ambient_sound": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "soundscape", "music")
    FUNCTION = "build"
    CATEGORY = "OrbitSheets"

    def build(self, character_description, visual_style, orbit_direction,
              end_on_closeup, backdrop="plain seamless neutral grey studio backdrop",
              spoken_line="", voice_description="", language="English",
              silent_during_closeup=True, ambient_sound=""):
        description = character_description.strip().rstrip(".")
        style = visual_style.strip().rstrip(".") or "Cinematic, live-action"
        subject = description or "the character"
        set_dressing = backdrop.strip().rstrip(".") or "plain seamless neutral grey studio backdrop"
        # It opens a sentence, so it has to read like one.
        set_dressing = set_dressing[0].upper() + set_dressing[1:]

        closeup = (
            " After completing the circle the camera pushes in slowly to a tight "
            "close-up of the face, framed from the top of the hair to the chin."
            if end_on_closeup else ""
        )

        line = spoken_line.strip()
        speaks = bool(line)

        if speaks:
            voice = voice_description.strip().rstrip(".")
            voice_clause = f" The voice is {voice}." if voice else ""
            # Speech is confined to the arc. The close-up is the frame most
            # worth having clean, and a mouth caught mid-phoneme ruins it —
            # whereas at full-body scale the mouth is a few pixels wide.
            hush = (
                " As the camera begins to push in the figure finishes speaking, "
                "closes their mouth and holds a still, neutral expression for "
                "the whole close-up."
                if (silent_during_closeup and end_on_closeup) else ""
            )
            # The guide says to keep the line's own punctuation inside <d>, so
            # only add a full stop when the line does not already end in one.
            tail = "" if line[-1] in ".!?…\"'" else "."
            speech = (
                f" While the camera circles, the figure (S1) speaks calmly and "
                f"directly to camera, saying <d>[{language}] {line}</d>{tail}"
                f"{voice_clause}{hush}"
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
                "relaxed at the sides, and does not move, walk, gesture, turn, "
                "or change expression at any point."
            )
            ambient = ambient_sound.strip() or "N/A"

        body = (
            # The description is stated once, as its own sentence. Repeating it
            # inside the camera sentence spends context on nothing and reads as
            # two subjects to the encoder.
            f"{style}. Full-body turnaround of a single standing figure. "
            f"{subject}. The camera performs "
            f"an arc shot {orbit_direction} around the figure with large "
            "amplitude at slow speed, circling to show the front, both side "
            f"profiles and the back."
            f"{closeup}"
            f"{speech}"
            f"{stillness} "
            f"{set_dressing}, even neutral lighting from every side, no props "
            "and no cast shadows. Clothing, hair, colours, proportions and every "
            "visible detail remain identical from every angle. A single "
            "continuous take with no cuts, no transitions, no on-screen text "
            "and no titles."
        )

        # The guide's three-field shape, matching what the app's prompt
        # compiler emits so both halves of the pipeline read the same.
        prompt = (
            f"{body}\n\n"
            f"overall_soundscape:\n{ambient}\n\n"
            f"non_diegetic_music:\nN/A"
        )
        return (prompt, ambient, "N/A")


class LumosFrameSelect:
    """Choose the frames from an orbit that are worth keeping.

    Even spacing is the obvious approach and the wrong one: an orbit is not
    uniform, and evenly-spaced samples land on blurred mid-swing frames and on
    pairs that show the same wall twice. This shortlists by time, then has the
    vision model compare the shortlist as one image and say which views
    actually differ and which are sharp.

    Falls back to sharpness-and-spread scoring whenever the vision model is
    not there — which, given the model is unloaded on every tab switch, is a
    normal state and not an error.
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
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "info")
    FUNCTION = "select"
    CATEGORY = "OrbitSheets"

    def select(self, images, count, mode, candidates=DEFAULT_CANDIDATES,
               keep_first_frame=True, subject_hint="", selection_brief="",
               sharpness_weight=0.35, free_vram_first=True, llm_url="",
               timeout_seconds=180):
        total = int(images.shape[0])
        count = max(1, min(int(count), total))

        if total <= count:
            return (images, f"kept all {total} frames (asked for {count})")

        if mode == "sharpness_diversity":
            picked = _greedy_spread(images, count, sharpness_weight, keep_first_frame)
            return (images[picked], self._report("sharpness+spread", picked, total))

        # Shortlist by time so the montage stays readable, keeping the opening
        # frame when asked: it is the Krea2 reference the orbit was built from.
        shortlist = self._shortlist(total, max(count, int(candidates)), keep_first_frame)
        pils = _tensor_to_pils(images[shortlist])

        if free_vram_first:
            self._free_vram()

        picks, note = _vlm_pick(
            pils, count, subject_hint, _llm_base_url(llm_url),
            int(timeout_seconds), selection_brief,
        )

        if picks is None:
            picked = _greedy_spread(images, count, sharpness_weight, keep_first_frame)
            return (
                images[picked],
                self._report(f"sharpness+spread (fallback: {note})", picked, total),
            )

        picked = sorted({shortlist[i] for i in picks})
        # A model that returned too few still gave us its good ones; fill the
        # remainder by spread rather than discarding its judgement.
        if len(picked) < count:
            for index in _greedy_spread(images, count, sharpness_weight, keep_first_frame):
                if index not in picked:
                    picked.append(index)
                if len(picked) >= count:
                    break
            picked = sorted(picked)

        return (images[picked], self._report(f"vision model — {note}", picked, total))

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
    "OrbitSheetsLocationPrompt": "Location Orbit Prompt (H3)",
    "OrbitSheetsCharacterPrompt": "Character Turnaround Prompt (H3)",
    "OrbitSheetsFrameSelect": "Frame Select (vision-judged)",
    "OrbitSheetsContactSheet": "Contact Sheet",
    "OrbitSheetsAttentionBackend": "Attention Backend",
}
