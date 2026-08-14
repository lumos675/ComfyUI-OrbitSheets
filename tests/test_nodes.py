"""Exercise every node without ComfyUI, a GPU, or a vision model.

Builds a synthetic camera move (a scene that rotates, with some frames
deliberately blurred), then checks that selection rejects blur, spreads across
angles, honours a vision model's picks, and degrades sanely when no model is
reachable. Prompt construction is checked against MiniMax-H3's documented
format.

Run from the pack directory:  python tests/test_nodes.py
"""

import importlib.util
import logging
import sys
import types
from pathlib import Path

import numpy as np
import torch

NODE = Path(__file__).resolve().parent.parent

pkg = types.ModuleType("orbitsheets")
pkg.__path__ = [str(NODE)]
sys.modules["orbitsheets"] = pkg

spec = importlib.util.spec_from_file_location(
    "orbitsheets.nodes", NODE / "nodes.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["orbitsheets.nodes"] = mod
spec.loader.exec_module(mod)
print("loaded nodes:", list(mod.NODE_CLASS_MAPPINGS))

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


# ---------------------------------------------------------------- fixtures

def synthetic_orbit(frames=96, h=96, w=128, blur_at=(20, 21, 22, 55, 56)):
    """A 'rotating' scene: a bright bar sweeps position with the camera.

    Blurred frames get heavy smoothing, which is what a real mid-swing frame
    looks like to a Laplacian.
    """
    out = np.zeros((frames, h, w, 3), dtype=np.float32)
    for i in range(frames):
        img = np.full((h, w, 3), 0.15, dtype=np.float32)
        # A hard-edged feature whose position encodes the viewing angle.
        pos = int((i / frames) * (w - 20))
        img[:, pos:pos + 18] = 0.9
        img[h // 3: h // 3 + 6, :] = 0.55  # horizon line, always sharp
        # Per-angle texture so descriptors differ beyond the bar position.
        rng = np.random.default_rng(i)
        img += rng.normal(0, 0.01, img.shape).astype(np.float32)
        if i in blur_at:
            k = 9
            pad = np.pad(img, ((k, k), (k, k), (0, 0)), mode="edge")
            acc = np.zeros_like(img)
            for dy in range(-k, k + 1):
                for dx in range(-k, k + 1):
                    acc += pad[k + dy:k + dy + h, k + dx:k + dx + w]
            img = acc / ((2 * k + 1) ** 2)
        out[i] = np.clip(img, 0, 1)
    return torch.from_numpy(out)


images = synthetic_orbit()
BLURRED = {20, 21, 22, 55, 56}

# ------------------------------------------------------- sharpness detector

sharp = mod._sharpness(images)
blur_scores = [float(sharp[i]) for i in sorted(BLURRED)]
sharp_scores = [float(sharp[i]) for i in range(images.shape[0]) if i not in BLURRED]
check(
    "Laplacian separates blurred frames",
    max(blur_scores) < min(sharp_scores),
    f"blurred max={max(blur_scores):.5f} < sharp min={min(sharp_scores):.5f}",
)

# ------------------------------------------------------- classical fallback

sel = mod.LumosFrameSelect()
frames, info = sel.select(
    images, count=6, mode="sharpness_diversity", keep_first_frame=True
)
picked = [int(n) for n in info.split("indices ")[1].strip("[]").split(",")]
check("classical returns the requested count", frames.shape[0] == 6, info)
check("classical avoids blurred frames", not (set(picked) & BLURRED), f"picked {picked}")
check("classical keeps the reference frame", picked[0] == 0)
spread = min(b - a for a, b in zip(picked, picked[1:]))
check("classical spreads across the orbit", spread >= 5, f"min gap {spread}")

# ------------------------------------------ vision path with model absent

frames_v, info_v = sel.select(
    images, count=6, mode="vision_llm", keep_first_frame=True,
    free_vram_first=False, llm_url="http://127.0.0.1:9  ",  # nothing there
)
check(
    "vision mode falls back cleanly when the model is absent",
    frames_v.shape[0] == 6 and "fallback" in info_v,
    info_v,
)

# ------------------------------------------ vision path with a stub model

import http.server
import json as _json
import threading

captured = {}


class Stub(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = _json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        content = body["messages"][0]["content"]
        captured["has_image"] = any(p.get("type") == "image_url" for p in content)
        captured["text"] = next(p["text"] for p in content if p.get("type") == "text")
        reply = {"choices": [{"message": {"content":
            '```json\n{"picks": [1, 5, 9, 12, 14, 16], "why": "distinct sides, all sharp"}\n```'}}]}
        payload = _json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


server = http.server.HTTPServer(("127.0.0.1", 8977), Stub)
threading.Thread(target=server.serve_forever, daemon=True).start()

frames_s, info_s = sel.select(
    images, count=6, mode="vision_llm", keep_first_frame=True,
    subject_hint="a stone courtyard at dusk", free_vram_first=False,
    llm_url="http://127.0.0.1:8977",
)
check("vision model actually receives an image", captured.get("has_image") is True)
check("prompt names the subject", "stone courtyard" in captured.get("text", ""))
check("vision picks honoured", frames_s.shape[0] == 6 and "vision model" in info_s, info_s)
check("vision reason surfaced", "distinct sides" in info_s, info_s)

# a model that under-returns should be topped up, not rejected
class Short(Stub):
    def do_POST(self):
        body = _json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        payload = _json.dumps({"choices": [{"message": {"content": '{"picks":[2,7]}'}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

server.shutdown()
server2 = http.server.HTTPServer(("127.0.0.1", 8978), Short)
threading.Thread(target=server2.serve_forever, daemon=True).start()
frames_t, info_t = sel.select(
    images, count=6, mode="vision_llm", free_vram_first=False,
    llm_url="http://127.0.0.1:8978",
)
check("short answer topped up to count", frames_t.shape[0] == 6, info_t)
server2.shutdown()

# ------------------------------------------------------ in-graph CLIP judging

class FakeClip:
    """Stub of the core CLIP generate API (tokenize -> generate -> decode)."""
    def __init__(self, reply):
        self.reply = reply
        self.seen = {}
    def tokenize(self, text, **kwargs):
        self.seen["text"] = text
        self.seen["has_image"] = kwargs.get("image") is not None
        return {"tokens": text}
    def generate(self, tokens, **kwargs):
        self.seen["do_sample"] = kwargs.get("do_sample")
        return [1, 2, 3]
    def decode(self, ids):
        return self.reply

clip = FakeClip('{"picks":[1, 5, 9, 12, 14, 16], "why": "distinct sides, all sharp"}')
frames_c, info_c = sel.select(
    images, count=6, mode="vision_llm", clip=clip, free_vram_first=False,
)
check("in-graph CLIP path picks frames",
      frames_c.shape[0] == 6 and "in-graph CLIP" in info_c, info_c)
check("in-graph CLIP received the montage image",
      clip.seen.get("has_image") is True)
check("in-graph CLIP got the frame-selection instruction",
      "number of frame" in clip.seen.get("text", "") or "numbered contact sheet" in clip.seen.get("text", ""))
check("in-graph CLIP sampling on by default", clip.seen.get("do_sample") is True)

class BrokenClip:
    def tokenize(self, *a, **k):
        raise RuntimeError("no vision tower")

frames_b, info_b = sel.select(
    images, count=6, mode="vision_llm", clip=BrokenClip(),
    free_vram_first=False, llm_url="http://127.0.0.1:9  ",
)
check("broken in-graph CLIP falls back cleanly",
      frames_b.shape[0] == 6 and "fallback" in info_b, info_b)

# -------------------------------------------------------------- sharpness gate

gated = sel._gate_sharp(images, list(range(96)), keep_n=20)
check("gate caps the shortlist", len(gated) <= 20, f"kept {len(gated)}")
check("gate drops the blurred frames", not (set(gated) & BLURRED), f"kept {gated}")
check("gate preserves order", gated == sorted(gated))

# -------------------------------------------------------------- per-shot picks

per, _ = sel._per_shot(images, shots=5)
segs = [range(int(i * 96 / 5), int((i + 1) * 96 / 5)) for i in range(5)]
check("per-shot returns one frame per segment", len(per) == 5, f"{per}")
check("per-shot keeps each segment represented",
      all(any(p in s for s in segs) for p in per))
check("per-shot avoids the blurred frames", not (set(per) & BLURRED), f"{per}")

# shots >= count => pure per-shot, no vision needed at all
frames_p, info_p = sel.select(
    images, count=5, mode="vision_llm", shots=5,
    free_vram_first=False, llm_url="http://127.0.0.1:9  ",
)
check("per-shot mode needs no vision server",
      frames_p.shape[0] == 5 and "per-shot" in info_p, info_p)

# shots < count => per-shot base, then fill
frames_q, info_q = sel.select(
    images, count=8, mode="vision_llm", shots=5,
    free_vram_first=False, llm_url="http://127.0.0.1:9  ",
)
picked_q = [int(x) for x in info_q.split("indices ")[1].strip("[]").split(",")]
check("per-shot base plus fill reaches count",
      frames_q.shape[0] == 8 and len(picked_q) == 8 and "per-shot" in info_q, info_q)
check("per-shot base keeps every segment even when filling",
      all(any(p in s for s in segs) for p in picked_q))

# with a clip wired, a dead llm_url must never be touched (no probe/hang)
frames_r, info_r = sel.select(
    images, count=6, mode="vision_llm", clip=FakeClip('{"picks":[2,7]}'),
    free_vram_first=False, llm_url="http://127.0.0.1:9  ",
)
check("wired clip ignores llm_url entirely",
      frames_r.shape[0] == 6 and "in-graph CLIP" in info_r, info_r)

# ------------------------------------------------------------ multi-board scan

frames_mb, info_mb = sel.select(
    images, count=6, mode="vision_llm", boards=4,
    clip=FakeClip('{"picks":[1, 5, 9, 12, 14, 16]}'),
    free_vram_first=False,
)
picked_mb = [int(x) for x in info_mb.split("indices ")[1].strip("[]").split(",")]
bs = [range(int(i * 96 / 4), int((i + 1) * 96 / 4)) for i in range(4)]
covered = len({b for p in picked_mb for b, r in enumerate(bs) if p in r})
check("multi-board scan returns count frames",
      frames_mb.shape[0] == 6 and "multi-board" in info_mb, info_mb)
check("multi-board scan spans several sectors",
      covered >= 3, f"covered {covered}/4 boards")
# boards + shots together still guarantee the per-shot views
frames_ms, info_ms = sel.select(
    images, count=8, mode="vision_llm", shots=5, boards=4,
    free_vram_first=False, llm_url="http://127.0.0.1:9  ",
)
segs5 = [range(int(i * 96 / 5), int((i + 1) * 96 / 5)) for i in range(5)]
picked_ms = [int(x) for x in info_ms.split("indices ")[1].strip("[]").split(",")]
check("shots + boards reaches count",
      frames_ms.shape[0] == 8 and "per-shot" in info_ms, info_ms)
check("shots + boards keeps every shot segment",
      all(any(p in s for s in segs5) for p in picked_ms))

# --------------------------------------------------------- parsing edge cases

check("parser handles prose", mod._parse_picks("I pick 3, 7 and 11.", 16) == [2, 6, 10])
check("parser drops out-of-range", mod._parse_picks('{"picks":[1,99,4]}', 16) == [0, 3])
check("parser dedupes", mod._parse_picks('{"picks":[5,5,2]}', 16) == [4, 1])

# ------------------------------------------------------------ contact sheet

sheet_node = mod.LumosContactSheet()
(sheet,) = sheet_node.compose(frames, columns=3, cell_width=256, padding=8)
check("sheet is a single image", sheet.shape[0] == 1, f"shape {tuple(sheet.shape)}")
exp_w = 3 * 256 + 8 * 4
check("sheet width matches the grid", sheet.shape[2] == exp_w, f"{sheet.shape[2]} vs {exp_w}")
check("sheet has 2 rows for 6 frames", sheet.shape[1] > sheet.shape[2] * 0.3)
check("sheet values in range", float(sheet.min()) >= 0.0 and float(sheet.max()) <= 1.0)

# --------------------------------------------------- cut detection per shot

def synthetic_cuts(bounds, h=96, w=128):
    """A hard-cut sequence: each shot is a distinct static view.

    Shot lengths are deliberately uneven, which is the real failure — H3 does
    not cut at the exact seconds the prompt asks for, so equal time slices
    sample one view twice and miss another entirely.
    """
    total = bounds[-1]
    out = np.zeros((total, h, w, 3), dtype=np.float32)
    for shot, (a, b) in enumerate(zip(bounds, bounds[1:])):
        view = np.full((h, w, 3), 0.1 + 0.15 * shot, dtype=np.float32)
        view[:, 10 * shot:10 * shot + 25] = 0.95        # layout differs per shot
        view[h // 3: h // 3 + 5, :] = 0.5
        for f in range(a, b):
            rng = np.random.default_rng(f)
            out[f] = np.clip(view + rng.normal(0, 0.01, view.shape), 0, 1)
    return torch.from_numpy(out)

BOUNDS = [0, 40, 52, 95, 104, 124]          # five very uneven shots
cutvid = synthetic_cuts(BOUNDS)
TRUE_CUTS = BOUNDS[1:-1]

segs, found_cuts = mod.LumosFrameSelect._segments(cutvid, 5, detect=True)
check("cuts found where the shots actually change", found_cuts == TRUE_CUTS,
      f"found {found_cuts}, expected {TRUE_CUTS}")

picks_detected, _ = mod.LumosFrameSelect._per_shot(cutvid, 5, detect=True)
shot_of = lambda f: max(i for i, b in enumerate(BOUNDS[:-1]) if f >= b)
check("one frame from each real shot",
      sorted(shot_of(f) for f in picks_detected) == [0, 1, 2, 3, 4],
      f"shots hit: {sorted(shot_of(f) for f in picks_detected)}")

# The bug this replaces: equal slices on the same video miss a view.
picks_even, _ = mod.LumosFrameSelect._per_shot(cutvid, 5, detect=False)
check("equal slices would have missed a shot (why detection exists)",
      len(set(shot_of(f) for f in picks_even)) < 5,
      f"shots hit: {sorted(shot_of(f) for f in picks_even)}")

# A continuous move has no cuts to find; it must not invent them.
_, none_found = mod.LumosFrameSelect._segments(images, 5, detect=True)
check("no cuts invented on a continuous move", none_found == [], f"found {none_found}")

sel_cut = mod.LumosFrameSelect()
frames_cut, info_cut = sel_cut.select(
    cutvid, count=5, mode="vision_llm", shots=5, keep_first_frame=False,
    free_vram_first=False, shot_split="cuts (detected)",
)
check("shot-count == frame-count skips the vision model entirely",
      frames_cut.shape[0] == 5 and "no vision" in info_cut, info_cut)
check("info reports where the cuts were found", "cuts found at" in info_cut, info_cut)

# ------------------------------------------- grouping by view, not by time

def synthetic_views(plan, h=96, w=128):
    """A cut sequence where the same view appears in more than one segment.

    This is the observed failure: H3 gave the front view two long stretches and
    the right profile half a second, so an even split — and a cut-detected
    split asked for five groups — returned front twice and no right profile.
    `plan` is (view_id, frame_count) pairs.
    """
    views = {}
    for vid, _ in plan:
        if vid in views:
            continue
        v = np.full((h, w, 3), 0.1 + 0.13 * vid, dtype=np.float32)
        v[:, 12 * vid:12 * vid + 22, vid % 3] = 0.95
        v[h // 3 + 3 * vid: h // 3 + 3 * vid + 6, :] = 0.55
        views[vid] = v
    frames, truth = [], []
    for vid, n in plan:
        for _ in range(n):
            rng = np.random.default_rng(len(frames))
            frames.append(np.clip(views[vid] + rng.normal(0, 0.01, (h, w, 3)), 0, 1))
            truth.append(vid)
    return torch.from_numpy(np.stack(frames).astype(np.float32)), truth

# front runs long and returns; the right profile gets 6 frames.
PLAN = [(0, 44), (1, 14), (2, 9), (0, 31), (3, 6), (4, 20)]
viewvid, truth = synthetic_views(PLAN)
VIEWS = len({v for v, _ in PLAN})

groups = mod._cluster_views(viewvid, VIEWS)
check("clustering finds one group per distinct view",
      sorted(len({truth[f] for f in g}) for g in groups) == [1] * VIEWS,
      f"groups span views {[sorted({truth[f] for f in g}) for g in groups]}")

by_view, distinct_views = mod.LumosFrameSelect._per_view(viewvid, VIEWS)
check("every distinct view reaches the sheet",
      sorted(truth[f] for f in by_view) == sorted(set(truth)),
      f"got views {sorted(truth[f] for f in by_view)}")

# The bug this replaces, on the very same video.
by_time, _ = mod.LumosFrameSelect._per_shot(viewvid, VIEWS, detect=False)
check("splitting by time misses a view (why clustering exists)",
      len({truth[f] for f in by_time}) < VIEWS,
      f"got views {sorted(truth[f] for f in by_time)}")
by_cuts, _ = mod.LumosFrameSelect._per_shot(viewvid, VIEWS, detect=True)
check("even correct cut detection misses it, since a view repeats",
      len({truth[f] for f in by_cuts}) < VIEWS,
      f"got views {sorted(truth[f] for f in by_cuts)}")

sel_v = mod.LumosFrameSelect()
frames_v2, info_v2 = sel_v.select(
    viewvid, count=VIEWS, mode="vision_llm", shots=VIEWS,
    shot_split="views (by content)", keep_first_frame=False, free_vram_first=False,
)
check("view grouping is the default path and needs no vision model",
      frames_v2.shape[0] == VIEWS and "grouped by view" in info_v2, info_v2)
check("a video with five real views reports five distinct",
      distinct_views == VIEWS, f"reported {distinct_views}")

# A take that under-delivers: the model gave three of its six shots to the
# same subject, which is what fills a location sheet with one fountain.
POOR = [(0, 30), (1, 22), (2, 18), (2, 26), (2, 28)]
poorvid, poor_truth = synthetic_views(POOR)
poor_picks, poor_distinct = mod.LumosFrameSelect._per_view(poorvid, 6)
check("a repeated view is counted once, not six times",
      poor_distinct == 3, f"reported {poor_distinct} distinct")
_, poor_info = mod.LumosFrameSelect().select(
    poorvid, count=6, mode="vision_llm", shots=6, keep_first_frame=False,
    free_vram_first=False,
)
check("the sheet warns that the video, not the picker, ran out of views",
      "only 3 distinct views" in poor_info, poor_info)

check("clustering is deterministic across runs",
      mod.LumosFrameSelect._per_view(viewvid, VIEWS)[0] == by_view)

# ---------------------------------------------------------------- prompt

prompt_node = mod.LumosOrbitPrompt()
prompt, sound, music = prompt_node.build(
    location_description="A moonlit stone courtyard ringed by cypress",
    visual_style="Cinematic, live-action",
    space="exterior",
    coverage="cut views",
    time_of_day="moonlit night",
)
check("prompt carries the I2VA first-frame envelope",
      "For the target video" in prompt
      and "integrated_multimodal_description:" in prompt)
for needle in ("[Shot 1]", "[Shot 2]", "[Shot 3]", "[Shot 4]", "[Shot 5]",
               "the shot cuts to", "locked-off static camera", "moonlit night"):
    check(f"prompt contains {needle!r}", needle in prompt)
# The failure this replaced: one unbroken 360 pan tumbled mid-arc.
check("prompt asks for cuts, not a continuous move",
      "continuous unbroken take" not in prompt and "360-degree" not in prompt)
check("prompt forbids people", "no people" in prompt)
check("music defaults to N/A", music == "N/A")
check("soundscape defaults to N/A", sound == "N/A")
_, sound2, _ = prompt_node.build(
    location_description="x", visual_style="y", space="exterior",
    ambient_sound="wind in cypress",
)
check("ambient sound passes through", sound2 == "wind in cypress")

# --------------------------------------------- no duplicated description

DESC = "A moonlit stone courtyard ringed by cypress"
p_loc, _, _ = prompt_node.build(
    location_description=DESC, visual_style="Cinematic, live-action",
    space="exterior", coverage="cut views",
)
check("location states the description once", p_loc.count(DESC) == 1,
      f"appears {p_loc.count(DESC)}x")

turn = mod.LumosCharacterTurnaroundPrompt()
CDESC = "A 30-year-old woman with cropped black hair"
p_ch, _, music_ch = turn.build(
    character_description=CDESC, visual_style="Cinematic, live-action",
)
# ------------------------------------------------ the expression shot (6)

p_six = turn.build(character_description="A blonde woman", visual_style="Photography")[0]
check("sixth shot is the frightened face", "[Shot 6]" in p_six
      and "expression now frightened" in p_six)
check("the scared shot names features, not just the emotion",
      "eyes wide" in p_six and "brows raised" in p_six)
# Otherwise the model reads "scared" as a performance and breaks the framing.
check("the scared shot forbids flinching away",
      "does not flinch, recoil, turn away or move" in p_six)
check("stillness no longer contradicts the expression change",
      "or change expression at any point" not in p_six
      and "neutral expression until the final shot" in p_six)
p_five = turn.build(character_description="A blonde woman",
                    visual_style="Photography", scared_shot=False)[0]
check("the scared shot can be switched off",
      "[Shot 6]" not in p_five and "[Shot 5]" in p_five)
check("without it the strict no-expression-change rule returns",
      "or change expression at any point" in p_five)

check("character states the description once", p_ch.count(CDESC) == 1,
      f"appears {p_ch.count(CDESC)}x")
check("prompt carries the I2VA first-frame instruction",
      "For the target video, at 0.00 seconds into the target video" in p_ch
      and "<Picture 1> (from [Shot 1]) is fully referenced" in p_ch)
check("prompt uses the integrated_multimodal_description field",
      "integrated_multimodal_description: [Shot 1]" in p_ch)
# Cut times step by shot_seconds; the final shot keeps whatever is left of
# the take, which is how six shots fit the same 124 frames as five.
for shot, at in (("[Shot 2]", "00:00.750"), ("[Shot 3]", "00:01.500"),
                 ("[Shot 4]", "00:02.250"), ("[Shot 5]", "00:03.000"),
                 ("[Shot 6]", "00:03.750")):
    check(f"{shot} has a cut time", f"{shot} At {at}" in p_ch)
check("shot length is tunable",
      "[Shot 6] At 00:03.000" in turn.build(
          character_description=CDESC, visual_style="x", shot_seconds=0.6)[0])
# Clamped to an even division, so the last shot can never be pushed past the
# end of the take and lost.
check("shot length cannot overrun the take",
      "[Shot 6] At 00:04.167" in turn.build(
          character_description=CDESC, visual_style="x", shot_seconds=2.0)[0])
for needle in ("full-body shot frames", "tight close-up of the face",
               "left side profile", "right side profile", "rear view",
               "does not move", "identical across every shot"):
    check(f"turnaround contains {needle!r}", needle in p_ch)
check("backdrop sentence is capitalised", ". Plain seamless" in p_ch,
      p_ch[p_ch.find("no cast shadows") - 90:p_ch.find("no cast shadows")][:90])
check("turnaround music is N/A", music_ch == "N/A")

# ------------------------------------------------------ framing control

p_gm, _, _ = turn.build(
    character_description=CDESC, visual_style="Cinematic, live-action",
)
check("default framing demands generous margin",
      "generous empty margin" in p_gm
      and "touch, cross or be cut off" in p_gm)
check("framing clause appears exactly once",
      p_gm.count("generous empty margin") == 1,
      f"appears {p_gm.count('generous empty margin')}x")
p_tight, _, _ = turn.build(
    character_description=CDESC, visual_style="x", framing="full body, tight",
)
check("tight framing chosen", "tight cropping" in p_tight
      and "generous empty margin" not in p_tight)

# ------------------------------------------------ speaking turnaround

LINE = "Hello. This is how my voice sounds."
p_sp, sound_sp, music_sp = turn.build(
    character_description=CDESC, visual_style="Cinematic, live-action",
    spoken_line=LINE, voice_description="a warm, low female voice",
    language="English", silent_during_closeup=True,
)
check("dialogue uses the <d>[Lang] ...</d> tag", f"<d>[English] {LINE}</d>" in p_sp)
check("speaker id assigned", "(S1)" in p_sp)
check("no doubled full stop after the tag", "</d>." not in p_sp,
      p_sp[p_sp.find("</d>"):p_sp.find("</d>") + 24])
check("voice description carried", "warm, low female voice" in p_sp)
check("mouth stays closed after the opening shot",
      "the mouth closed and a still, neutral expression" in p_sp)
check("stillness rule allows speech",
      "Apart from the movement of speech" in p_sp
      and "change expression at any point" not in p_sp)
check("soundscape field emitted", "overall_soundscape:" in p_sp and sound_sp != "N/A")
check("music field emitted as N/A", "non_diegetic_music: N/A" in p_sp and music_sp == "N/A")

# a line already ending in punctuation must not gain another
p_q, _, _ = turn.build(character_description=CDESC, visual_style="x",
                       spoken_line="Who goes there?")
check("question mark preserved", "there?</d>" in p_q and "</d>." not in p_q)

# silent mode must be unchanged
p_si, sound_si, _ = turn.build(character_description=CDESC, visual_style="x",
                               scared_shot=False)
check("no speech when no line given",
      "<d>" not in p_si and "(S1)" not in p_si and sound_si == "N/A")
check("silent mode keeps the strict stillness rule",
      "change expression at any point" in p_si)

# mouth-closed clauses can be switched off
p_open, _, _ = turn.build(character_description=CDESC, visual_style="x",
                          spoken_line=LINE, silent_during_closeup=False)
check("mouth-closed clauses can be switched off",
      "the mouth closed and a still" not in p_open)

# ------------------------------ interior shoots across, exterior walks round

def loc(space, **kw):
    kw.setdefault("coverage", "cut views")
    return prompt_node.build(
        location_description="A grand salon with arched doors",
        visual_style="Cinematic", space=space, **kw,
    )[0]


# ------------------------------------------------- cuts are the default now

def rot(space, **kw):
    kw.setdefault("coverage", "continuous move")
    return prompt_node.build(
        location_description="A moonlit stone castle on a hill",
        visual_style="Cinematic", space=space, **kw,
    )[0]

# Every continuous move tried on a cathedral interior lost to cuts: the pan
# rolled at every rate, and the glide that held level only ever went right.
# A locked-off tripod frame has no camera motion to get wrong.
default_cov = prompt_node.build(
    location_description="A grand salon", visual_style="Cinematic",
    space="interior")[0]
check("cut views are the default coverage",
      "[Shot 2]" in default_cov and "For the target video" in default_cov)

r_in, r_out = rot("interior"), rot("exterior")
pan = r_in
check("the continuous move is still one unbroken take",
      "[Shot 2]" not in r_in and "[Shot 2]" not in r_out)
check("indoors the continuous move pivots on the spot again",
      "holds its position" in r_in and "pans right" in r_in)
# A tall symmetrical interior cues the "spin under the vault" cliche, and the
# cathedral sheet came back rolled. Negative rules lost; positive framing
# anchors the model can check per frame are what hold.
check("the interior frame is pinned by what must be visible",
      "floor along the bottom of the frame" in r_in
      and "ceiling along the top" in r_in)
check("stability is stated positively, in every move",
      all("horizon stays level" in p and "stay vertical" in p
          for p in (r_in, r_out, pan)))
# The regression that brought the rolling back: the cut path's I2VA envelope
# was carried into the continuous path, and that envelope was found long ago
# to change this node's camera behaviour. The move is stated plainly instead.
check("the continuous move does not use the I2VA first-frame envelope",
      "For the target video" not in r_in and "<Picture 1>" not in r_in
      and r_in.startswith("Cinematic."))
check("the move is stated in H3's own camera terms",
      "pans right at slow speed" in r_in
      and "arc shot" in r_out and "at slow speed" in r_out)
check("cut views still use the I2VA envelope",
      "For the target video" in loc("exterior"))
# Speed is not a widget: the angle and the take length already fix it, and
# "fast speed" bolted onto a 360 is how the tumbling take was written.
check("nothing is ever written at fast speed",
      all("fast speed" not in p for p in (r_in, r_out, pan)))

# ------------------------------------------------- the turn has to fit the take
#
# The cathedral sheet that came back as a spiral up the vault asked for a
# complete 360 inside 124 frames — 72 deg/s, against the 24 deg/s of H3's own
# worked example. That prompt can no longer be written: the angle is rationed
# against the take, and asking for more clamps rather than obeys.
check("auto fits the turn to a five-second take",
      "180 degrees" in pan and "360" not in pan)
r_full = rot("interior", take_seconds=10.8,
             rotation="full turn (360 degrees, needs a 9s take)")
check("a full turn is honoured once the take can hold it",
      "360 degrees" in r_full and "all the way round" in r_full)
check("a quarter turn is available",
      "90 degrees" in rot("interior",
                          rotation="quarter turn (90 degrees)"))
check("auto opens up as the take grows",
      "360 degrees" in rot("interior", take_seconds=12.0))

_warnings = []
_root = logging.getLogger()
class _Catch(logging.Handler):
    def emit(self, record):
        _warnings.append(record.getMessage())
_root.addHandler(_Catch())
over = rot("interior", take_seconds=5.0,
           rotation="full turn (360 degrees, needs a 9s take)")
_over = [w for w in _warnings if "deg/s" in w]
rot("interior", take_seconds=10.8,
    rotation="half turn (180 degrees)")
_root.removeHandler(_root.handlers[-1])
check("a turn the take cannot hold is clamped, not written",
      "360 degrees" not in over and "180 degrees" in over, over[:200])
check("and the clamp says what it did and how to get the full turn",
      len(_over) == 1 and "72 deg/s" in _over[0]
      and "Writing 180 degrees instead" in _over[0]
      and "length 216" in _over[0], str(_warnings))
check("a turn the take can hold is not warned about",
      len([w for w in _warnings if "deg/s" in w]) == 1, str(_warnings))

# A graph saved against the old widget layout hands this node whatever string
# used to sit in that slot. It must fall back, never raise: the run that fed
# "full turn" into a five-second take got there exactly this way.
stale = rot("interior", coverage="continuous rotation",
            rotation="large amplitude")
check("stale widget values fall back instead of raising",
      "pans right at slow speed" in stale and "fast speed" not in stale
      and "180 degrees" in stale)

check("exterior circles the outside",
      "arc shot around the location" in r_out and "180 degrees" in r_out)
# The castle failure: told to shoot the rear it had never seen, the model went
# indoors instead of round. Said as where the camera *is*, not where it is
# banned from — and the same clause is what stops the take ending on a drone
# shot of the roof.
check("exterior states where the camera stays",
      "stays outside at ground level" in r_out
      and "same distance from the building" in r_out)
# The failure that sank the first rotation attempt — and the wording that was
# meant to fix it, which named four of H3's own motion types (Tilt Up, Tilt
# Down, Roll Clockwise/Counterclockwise) under a "never" the model does not
# read as a prohibition.
check("rotation never names the tilt and roll motion types",
      not any(w in r_in or w in r_out
              for w in ("tilts up", "tilts down", "rolls", "leans")))
check("rotation is one take, not cuts",
      "no cuts, no transitions" in r_in)
check("cut views are still available",
      "[Shot 6]" in loc("exterior"))
# Cuts are clamped to the take, so a longer `length` spreads them instead of
# stacking every shot into the opening five seconds.
check("a longer take gives the cuts more room",
      "[Shot 6] At 00:09.000" in loc("exterior", shot_seconds=2.0,
                                     take_seconds=10.8)
      and "[Shot 6] At 00:04.167" in loc("exterior", shot_seconds=2.0))

p_in, p_out = loc("interior"), loc("exterior")
check("interior shoots walls instead of elevations",
      "wall of the space" in p_in and "elevation" not in p_in)
check("exterior shoots elevations instead of walls",
      "elevation" in p_out and "wall of the space" not in p_out)
check("interior names the wall behind the camera",
      "behind the opening frame" in p_in)
check("exterior names the hidden far side",
      "hidden in the opening frame" in p_out)
check("both cover four sides",
      all(f"[Shot {n}]" in p_in and f"[Shot {n}]" in p_out for n in (1, 2, 3, 4)))
# Every shot has to say it, not just the first: the tumbling came from the
# model treating the later views as points in a move rather than as frames.
check("every shot is locked off",
      p_in.count("locked-off static camera") == 6)
p_four = loc("exterior", wide_establishing_shot=False, detail_shot=False)
check("both extra shots can be dropped for a bare four-view sheet",
      "[Shot 5]" not in p_four and "[Shot 4]" in p_four)
# Dropping only the wide view leaves the detail shot as Shot 5, not a gap.
p_five_loc = loc("exterior", wide_establishing_shot=False)
check("dropping the wide view renumbers rather than leaving a hole",
      "[Shot 5]" in p_five_loc and "[Shot 6]" not in p_five_loc
      and "wide three-quarter" not in p_five_loc)
check("the overview shot is a wide view when kept",
      "wide three-quarter establishing view" in p_out)
check("the detail shot closes in on materials", "[Shot 6]" in p_out
      and "closest shot of the sequence" in p_out)
# The observed failure: the wide shot and the detail shot both settled on the
# centrepiece at a similar distance, so half the sheet was one fountain.
check("wide and detail are pulled to opposite extremes",
      "widest shot of the sequence" in p_in and "closest shot" in p_in)
check("no two shots may repeat a view",
      "no two shots repeat the same view" in p_in
      and "no two shots repeat the same view" in p_out)
check("the detail shot can be dropped",
      "[Shot 6]" not in loc("exterior", detail_shot=False))
# The rear view is the one that silently fails: the camera can face the back
# wall while still framing part of the opening view.
check("the rear shot forbids the opening view appearing in it",
      "exact opposite direction to Shot 1" in p_in
      and "exact opposite direction to Shot 1" in p_out)
# Six shots share the take rather than needing a longer one.
for shot, at in (("[Shot 2]", "00:00.750"), ("[Shot 4]", "00:02.250"),
                 ("[Shot 6]", "00:03.750")):
    check(f"location {shot} has a cut time", f"{shot} At {at}" in p_out)
check("location shot length cannot overrun the take",
      "[Shot 6] At 00:04.167" in loc("exterior", shot_seconds=2.0))

print()
if failures:
    print("FAILURES:", ", ".join(failures))
    sys.exit(1)
print("all orbit-node checks passed")
