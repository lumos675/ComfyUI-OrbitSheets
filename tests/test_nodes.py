"""Exercise every node without ComfyUI, a GPU, or a vision model.

Builds a synthetic camera move (a scene that rotates, with some frames
deliberately blurred), then checks that selection rejects blur, spreads across
angles, honours a vision model's picks, and degrades sanely when no model is
reachable. Prompt construction is checked against MiniMax-H3's documented
format.

Run from the pack directory:  python tests/test_nodes.py
"""

import importlib.util
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

# ---------------------------------------------------------------- prompt

prompt_node = mod.LumosOrbitPrompt()
prompt, sound, music = prompt_node.build(
    location_description="A moonlit stone courtyard ringed by cypress",
    visual_style="Cinematic, live-action",
    space="exterior",
    orbit_direction="clockwise",
    amplitude="large amplitude",
    speed="slow speed",
    time_of_day="moonlit night",
)
for needle in ("arc shot", "clockwise", "large amplitude", "slow speed",
               "no cuts", "moonlit night"):
    check(f"prompt contains {needle!r}", needle in prompt)
check("prompt forbids people", "no people" in prompt)
check("music defaults to N/A", music == "N/A")
check("soundscape defaults to N/A", sound == "N/A")
_, sound2, _ = prompt_node.build(
    location_description="x", visual_style="y", space="exterior",
    orbit_direction="clockwise", amplitude="large amplitude", speed="slow speed",
    ambient_sound="wind in cypress",
)
check("ambient sound passes through", sound2 == "wind in cypress")

# --------------------------------------------- no duplicated description

DESC = "A moonlit stone courtyard ringed by cypress"
p_loc, _, _ = prompt_node.build(
    location_description=DESC, visual_style="Cinematic, live-action",
    space="exterior", orbit_direction="clockwise",
    amplitude="large amplitude", speed="slow speed",
)
check("location states the description once", p_loc.count(DESC) == 1,
      f"appears {p_loc.count(DESC)}x")

turn = mod.LumosCharacterTurnaroundPrompt()
CDESC = "A 30-year-old woman with cropped black hair"
p_ch, _, music_ch = turn.build(
    character_description=CDESC, visual_style="Cinematic, live-action",
    orbit_direction="clockwise", end_on_closeup=True,
)
check("character states the description once", p_ch.count(CDESC) == 1,
      f"appears {p_ch.count(CDESC)}x")
for needle in ("arc shot", "clockwise", "close-up of the face", "no cuts",
               "does not move", "identical from every angle"):
    check(f"turnaround contains {needle!r}", needle in p_ch)
check("backdrop sentence is capitalised", ". Plain seamless" in p_ch,
      p_ch[p_ch.find("no cast shadows") - 90:p_ch.find("no cast shadows")][:90])
check("turnaround music is N/A", music_ch == "N/A")
p_nc, _, _ = turn.build(
    character_description=CDESC, visual_style="x", orbit_direction="clockwise",
    end_on_closeup=False,
)
check("close-up can be switched off", "close-up of the face" not in p_nc)

# ------------------------------------------------ speaking turnaround

LINE = "Hello. This is how my voice sounds."
p_sp, sound_sp, music_sp = turn.build(
    character_description=CDESC, visual_style="Cinematic, live-action",
    orbit_direction="clockwise", end_on_closeup=True, spoken_line=LINE,
    voice_description="a warm, low female voice", language="English",
    silent_during_closeup=True,
)
check("dialogue uses the <d>[Lang] ...</d> tag", f"<d>[English] {LINE}</d>" in p_sp)
check("speaker id assigned", "(S1)" in p_sp)
check("no doubled full stop after the tag", "</d>." not in p_sp,
      p_sp[p_sp.find("</d>"):p_sp.find("</d>") + 24])
check("voice description carried", "warm, low female voice" in p_sp)
check("close-up is hushed", "closes their mouth" in p_sp)
check("stillness rule allows speech",
      "Apart from the movement of speech" in p_sp
      and "change expression at any point" not in p_sp)
check("soundscape field emitted", "overall_soundscape:" in p_sp and sound_sp != "N/A")
check("music field emitted as N/A", "non_diegetic_music:\nN/A" in p_sp and music_sp == "N/A")

# a line already ending in punctuation must not gain another
p_q, _, _ = turn.build(character_description=CDESC, visual_style="x",
                       orbit_direction="clockwise", end_on_closeup=True,
                       spoken_line="Who goes there?")
check("question mark preserved", "there?</d>" in p_q and "</d>." not in p_q)

# silent mode must be unchanged
p_si, sound_si, _ = turn.build(character_description=CDESC, visual_style="x",
                               orbit_direction="clockwise", end_on_closeup=True)
check("no speech when no line given",
      "<d>" not in p_si and "(S1)" not in p_si and sound_si == "N/A")
check("silent mode keeps the strict stillness rule",
      "change expression at any point" in p_si)

# close-up off => no hush clause to contradict
p_nh, _, _ = turn.build(character_description=CDESC, visual_style="x",
                        orbit_direction="clockwise", end_on_closeup=False,
                        spoken_line=LINE)
check("no hush clause without a close-up", "closes their mouth" not in p_nh)

# ------------------------------ interior pans, exterior arcs

def loc(space, **kw):
    return prompt_node.build(
        location_description="A grand salon with arched doors",
        visual_style="Cinematic", space=space, orbit_direction="clockwise",
        amplitude="large amplitude", speed="slow speed", **kw,
    )[0]

p_in, p_out = loc("interior"), loc("exterior")
check("interior pans instead of arcing",
      "pans right" in p_in and "arc shot" not in p_in)
check("interior holds position", "holds its position" in p_in)
check("exterior arcs instead of panning",
      "arc shot" in p_out and "pans" not in p_out)
check("both demand a full 360",
      "360-degree" in p_in and "360-degree" in p_out)
check("interior names the wall behind the camera",
      "behind the opening frame" in p_in)
check("exterior names the hidden far side",
      "hidden in the opening frame" in p_out)
p_ccw = loc("interior") if False else prompt_node.build(
    location_description="x", visual_style="y", space="interior",
    orbit_direction="counterclockwise", amplitude="large amplitude",
    speed="slow speed")[0]
check("counterclockwise pans left", "pans left" in p_ccw)

print()
if failures:
    print("FAILURES:", ", ".join(failures))
    sys.exit(1)
print("all orbit-node checks passed")
