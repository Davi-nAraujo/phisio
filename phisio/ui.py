"""Drawing: skeleton overlay on the video and the dark side dashboard.

Text is rendered with Pillow + a TTF font (OpenCV's Hershey fonts have no accents or degree sign);
shapes on the video use OpenCV. Colors are given in RGB and converted where OpenCV needs BGR.
"""
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import geometry as G

# palette (RGB)
P = {
    "bg": (10, 17, 33), "card": (19, 30, 54), "card2": (27, 42, 72), "border": (40, 58, 94),
    "text": (232, 238, 248), "sub": (140, 160, 192), "accent": (56, 189, 248), "accent2": (129, 140, 248),
    "ok": (52, 211, 153), "warn": (251, 191, 36), "bad": (248, 113, 113),
}


def bgr(rgb):
    return rgb[2], rgb[1], rgb[0]


# legacy BGR names still used by other modules
C = {"bg": bgr(P["bg"]), "text": bgr(P["text"]), "sub": bgr(P["sub"]), "primary": bgr(P["accent"]),
     "ok": bgr(P["ok"]), "warn": bgr(P["warn"]), "bad": bgr(P["bad"])}
LEVEL_RGB = [P["ok"], P["warn"], P["bad"]]
LEVEL_COLOR = [bgr(c) for c in LEVEL_RGB]
LEVEL_NAMES = ["OK", "ATENÇÃO", "RUIM"]
FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_W = 460

_FONT_DIRS = ["/usr/share/fonts/truetype/noto", "/usr/share/fonts/truetype/dejavu", "C:/Windows/Fonts",
              "/System/Library/Fonts/Supplemental", "/Library/Fonts"]
_FONT_NAMES = {False: ["NotoSans-Regular.ttf", "DejaVuSans.ttf", "segoeui.ttf", "Arial.ttf", "arial.ttf"],
               True: ["NotoSans-Bold.ttf", "DejaVuSans-Bold.ttf", "segoeuib.ttf", "Arial Bold.ttf", "arialbd.ttf"]}
_font_cache = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _font_cache:
        for d in _FONT_DIRS:
            for n in _FONT_NAMES[bold]:
                if (Path(d) / n).exists():
                    _font_cache[key] = ImageFont.truetype(str(Path(d) / n), size)
                    break
            if key in _font_cache:
                break
        else:
            _font_cache[key] = ImageFont.load_default(size)
    return _font_cache[key]


JOINT_LABEL = {"cotovelo": "Cotovelo", "joelho": "Joelho", "quadril": "Quadril", "ombro": "Ombro"}
SIDE_LABEL = {"D": "direito", "E": "esquerdo"}
EXERCISE_LABEL = {"agachamento": "Agachamento", "rosca_biceps": "Rosca bíceps", "elevacao_lateral": "Elevação lateral"}
METRIC_LABEL = {"pescoco": "Pescoço", "tronco": "Tronco", "ombros": "Ombros", "tronco_lateral": "Tronco lateral"}
FALL_LABEL = {"normal": "normal", "deitado": "deitado", "queda?": "queda?", "QUEDA": "QUEDA!"}


def joint_label(j):
    base, side = j.rsplit("_", 1)
    return f"{JOINT_LABEL.get(base, base)} {SIDE_LABEL.get(side, side)}"


def fmt(v, unit="°", nd=0):
    return f"{v:.{nd}f}{unit}" if isinstance(v, (int, float)) and math.isfinite(v) else "--"


# ---------------------------------------------------------------- text on numpy (BGR) images
def put(img, text, org, scale=0.55, color=C["text"], thick=1):
    """Text at baseline-left `org` (kept for compatibility); scale ~ OpenCV scale, color BGR."""
    size = max(10, int(scale * 30))
    pill(img, text, (org[0], org[1] - int(size * 0.9)), size, fg=color[::-1], bg=None, bold=thick > 1)


def pill(img, text, xy, size=22, fg=P["text"], bg=P["card"], bold=True, pad=(10, 5), radius=10, alpha=0.88,
         anchor="lt"):
    """Draw `text` (RGB colors) on a BGR numpy image, optionally inside a rounded box. Returns the box."""
    f = font(size, bold)
    l, t, r, b = f.getbbox(text)
    w, h = r - l + 2 * pad[0], b - t + 2 * pad[1]
    x, y = xy
    if anchor == "mt":
        x -= w // 2
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return (x, y, w, h)
    roi = img[y0:y1, x0:x1]
    patch = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    layer = Image.new("RGBA", patch.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    ox, oy = x - x0, y - y0
    if bg is not None:
        d.rounded_rectangle((ox, oy, ox + w - 1, oy + h - 1), radius, fill=(*bg, int(255 * alpha)))
    d.text((ox + pad[0] - l, oy + pad[1] - t), text, font=f, fill=(*fg, 255))
    patch = Image.alpha_composite(patch.convert("RGBA"), layer).convert("RGB")
    img[y0:y1, x0:x1] = cv2.cvtColor(np.asarray(patch), cv2.COLOR_RGB2BGR)
    return (x, y, w, h)


# ---------------------------------------------------------------- video overlay
LEFT = {G.L_SHOULDER, G.L_ELBOW, G.L_WRIST, G.L_HIP, G.L_KNEE, G.L_ANKLE, G.L_EAR, G.L_EYE}


def draw_skeleton(img, pts, color=None, thick=2, radius=4):
    """color=None: colored by body side (left = indigo, right = sky); otherwise one BGR color."""
    for a, b in G.SKELETON:
        if np.all(np.isfinite(pts[a])) and np.all(np.isfinite(pts[b])):
            c = color or bgr(P["accent2"] if (a in LEFT and b in LEFT) else P["accent"] if not (a in LEFT or b in LEFT)
                             else (200, 210, 230))
            cv2.line(img, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), c, thick, cv2.LINE_AA)
    for i, p in enumerate(pts):
        if np.all(np.isfinite(p)):
            c = color or bgr(P["accent2"] if i in LEFT else P["accent"])
            cv2.circle(img, (int(p[0]), int(p[1])), radius + 1, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(img, (int(p[0]), int(p[1])), radius, c, -1, cv2.LINE_AA)


def highlight_joint(img, pts, joint, value, color=None):
    """Thick limb segments + an arc at the vertex (like a protractor) + a value label."""
    color = color or bgr(P["ok"])
    a, b, c, kind = G.JOINTS[joint]
    if not all(np.all(np.isfinite(pts[i])) for i in (a, b, c)) or not math.isfinite(value):
        return
    pa, pb, pc = (np.asarray(pts[i], float) for i in (a, b, c))
    for p, q in ((pa, pb), (pb, pc)):
        cv2.line(img, tuple(map(int, p)), tuple(map(int, q)), color, 5, cv2.LINE_AA)
    # arc between the two segments; for flexion draw the bend (between BC and the extension of AB)
    u = pa - pb if kind == "interior" else pb - pa
    v = pc - pb
    r = int(max(18, min(60, 0.35 * min(np.linalg.norm(pa - pb), np.linalg.norm(pc - pb)))))
    a1, a2 = math.degrees(math.atan2(u[1], u[0])), math.degrees(math.atan2(v[1], v[0]))
    d = (a2 - a1 + 540) % 360 - 180
    overlay = img.copy()
    cv2.ellipse(overlay, tuple(map(int, pb)), (r, r), 0, a1, a1 + d, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    cv2.ellipse(img, tuple(map(int, pb)), (r, r), 0, a1, a1 + d, color, 2, cv2.LINE_AA)
    if kind == "flexion":   # dashed extension line = where 0 deg would be
        e = pb + (pb - pa) / max(np.linalg.norm(pb - pa), 1e-6) * r * 1.6
        cv2.line(img, tuple(map(int, pb)), tuple(map(int, e)), (230, 230, 230), 1, cv2.LINE_AA)
    cv2.circle(img, tuple(map(int, pb)), 8, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.circle(img, tuple(map(int, pb)), 6, color, -1, cv2.LINE_AA)
    pill(img, f"{value:.0f}°", (int(pb[0]) + r + 8, int(pb[1]) - 20), 26, fg=(255, 255, 255), bg=(15, 23, 42))


def banner(img, text, color):
    """Alert banner centered at the top. `color` is BGR (legacy API)."""
    pill(img, text, (img.shape[1] // 2, 18), 30, fg=(255, 255, 255), bg=color[::-1], pad=(22, 10), radius=16,
         alpha=0.95, anchor="mt")


def caption(img, text, remaining):
    """Large instruction + countdown at the bottom of the video, readable from a few meters away."""
    h, w = img.shape[:2]
    band = max(70, min(120, w // 11))
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - band), (w, h), bgr(P["bg"]), -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    count = f"{remaining:.0f}s"
    csize = int(band * 0.55)
    cw = int(font(csize, True).getlength("00s")) + 30
    size = int(band * 0.42)
    while font(size, True).getlength(text) > w - cw - 40 and size > 14:
        size -= 1
    pill(img, text, (16, h - band // 2 - size // 2 - 10), size, fg=P["text"], bg=None, pad=(6, 6))
    pill(img, count, (w - cw, h - band // 2 - csize // 2 - 12), csize, fg=P["accent"], bg=None, pad=(6, 6))


# ---------------------------------------------------------------- dashboard (Pillow)
class Panel:
    def __init__(self, height, width=PANEL_W):
        self.im = Image.new("RGB", (width, height), P["bg"])
        self.d = ImageDraw.Draw(self.im)
        self.w, self.h = width, height

    def text(self, xy, s, size=16, color=P["text"], bold=False, anchor="la"):
        self.d.text(xy, s, font=font(size, bold), fill=color, anchor=anchor)

    def card(self, y, h, title=None, x=16, w=None, fill=P["card"]):
        w = w or self.w - 2 * x
        self.d.rounded_rectangle((x, y, x + w, y + h), 14, fill=fill)
        if title:
            self.text((x + 16, y + 12), title.upper(), 12, P["sub"], True)
        return x + 16, y + (36 if title else 14)

    def chip(self, x, y, s, color, filled=True, size=14, pad=(10, 4)):
        f = font(size, True)
        w = int(f.getlength(s)) + 2 * pad[0]
        h = size + 2 * pad[1] + 4
        if filled:
            self.d.rounded_rectangle((x, y, x + w, y + h), h // 2, fill=color)
            self.text((x + pad[0], y + pad[1]), s, size, P["bg"], True)
        else:
            self.d.rounded_rectangle((x, y, x + w, y + h), h // 2, outline=color, width=1)
            self.text((x + pad[0], y + pad[1]), s, size, color, True)
        return x + w

    def bar(self, x, y, w, value, warn, bad, h=8):
        """Horizontal gauge 0..1.6*bad with green/amber/red zones and a marker at `value`."""
        top = 1.6 * bad
        sx = lambda v: x + int(w * min(max(v, 0), top) / top)  # noqa: E731
        for a, b, c in ((0, warn, P["ok"]), (warn, bad, P["warn"]), (bad, top, P["bad"])):
            self.d.rectangle((sx(a), y, sx(b), y + h), fill=tuple(int(0.35 * v + 0.65 * q) for v, q in zip(c, P["card"])))
        if math.isfinite(value):
            mx = sx(value)
            self.d.rounded_rectangle((mx - 3, y - 5, mx + 3, y + h + 5), 3, fill=P["text"])

    def spark(self, x, y, w, h, series, lines=(), color=P["accent"], lo=None, hi=None, window=10.0, extra=()):
        """Line plot of the last `window` seconds of (t, v) pairs, with optional horizontal guide lines.
        `extra`: more (series, color) drawn in the same box (e.g. the 3D angle next to the 2D one)."""
        self.d.rounded_rectangle((x, y, x + w, y + h), 8, fill=P["card2"])
        pts = [(t, v) for t, v in series if math.isfinite(v)]
        if not pts:
            self.text((x + w // 2, y + h // 2), "sem dados", 12, P["sub"], anchor="mm")
            return
        t1 = pts[-1][0]
        pts = [(t, v) for t, v in pts if t >= t1 - window]
        more = [([(t, v) for t, v in ser if math.isfinite(v) and t >= t1 - window], c) for ser, c in extra]
        vals = [v for _, v in pts] + [g for g, _ in lines] + [v for ser, _ in more for _, v in ser]
        lo = min(vals) if lo is None else lo
        hi = max(vals) if hi is None else hi
        if hi - lo < 1e-6:
            hi = lo + 1
        px = lambda t: x + 6 + (w - 12) * (t - (t1 - window)) / window  # noqa: E731
        py = lambda v: y + h - 6 - (h - 12) * (min(max(v, lo), hi) - lo) / (hi - lo)  # noqa: E731
        for g, c in lines:
            yy = py(g)
            for xx in range(x + 6, x + w - 6, 8):
                self.d.line((xx, yy, xx + 4, yy), fill=c, width=1)
        for ser, c in more:
            if len(ser) > 1:
                self.d.line([(px(t), py(v)) for t, v in ser], fill=c, width=2, joint="curve")
        if len(pts) > 1:
            self.d.line([(px(t), py(v)) for t, v in pts], fill=color, width=2, joint="curve")
        tx, ty = px(pts[-1][0]), py(pts[-1][1])
        self.d.ellipse((tx - 4, ty - 4, tx + 4, ty + 4), fill=color)

    def to_bgr(self):
        return cv2.cvtColor(np.asarray(self.im), cv2.COLOR_RGB2BGR)


MODE_TABS = ["Postura", "Ângulos", "Exercício", "Queda"]


def dashboard(height, s):
    """s: dict with everything the app wants to show (see app.py)."""
    p = Panel(height)
    # header
    p.text((20, 16), "Phisio", 30, P["text"], True)
    p.text((128, 30), "sensor de postura e movimento", 13, P["sub"])
    fps_col = P["ok"] if s["fps"] >= 15 else P["warn"]
    p.chip(p.w - 104, 22, f"{s['fps']:4.1f} FPS", fps_col, filled=False, size=13)
    # mode tabs
    x = 16
    for i, name in enumerate(MODE_TABS):
        active = MODE_TABS[i] == s["mode_tab"]
        x = p.chip(x, 66, f"{i + 1} {name}", P["accent"] if active else P["sub"], filled=active, size=14) + 8
    if not s["person"]:
        p.card(110, 120)
        p.text((p.w // 2, 160), "procurando pessoa...", 20, P["sub"], True, anchor="mm")
        p.text((p.w // 2, 192), "fique inteiro na imagem", 14, P["sub"], anchor="mm")
        main_end = 240
    else:
        main_end = {"Postura": _posture_card, "Ângulos": _angle_card, "Exercício": _reps_card,
                    "Queda": _fall_card}[s["mode_tab"]](p, 110, s)

    # always-on monitors
    y = main_end + 14
    half = (p.w - 48) // 2
    x0, ty = p.card(y, 76, "postura", x=16, w=half)
    p.chip(x0, ty, LEVEL_NAMES[s["posture_state"]], LEVEL_RGB[s["posture_state"]], size=15)
    fs = s["fall_state"]
    x1, ty = p.card(y, 76, "queda", x=32 + half, w=half)
    p.chip(x1, ty, FALL_LABEL.get(fs, fs), P["bad"] if fs == "QUEDA" else P["warn"] if fs != "normal" else P["ok"], size=15)

    # footer: toggles + keys
    y = height - 92
    x = 16
    for label, on in (("filtro 1€", s["filter"]), ("3D", s["depth"]), ("privacidade", s["privacy"]),
                      ("movimento", s["motion"])):
        x = p.chip(x, y, label, P["accent"] if on else P["border"], filled=on, size=12, pad=(8, 3)) + 6
    if s.get("gt") is not None:
        p.chip(p.w - 120, y - 34, f"rótulo: {LEVEL_NAMES[s['gt']]}", LEVEL_RGB[s["gt"]], size=12)
    p.text((18, height - 54), "1-4 modo · j junta · l lado · e exercício · r zera", 12, P["sub"])
    p.text((18, height - 34), "f filtro · d 3D · p privacidade · m movimento · q sair", 12, P["sub"])
    return p.to_bgr()


def _posture_card(p, y, s):
    x, ty = p.card(y, 300, f"postura · vista {s['view']}")
    st = s["posture_state"]
    p.chip(x, ty + 2, LEVEL_NAMES[st], LEVEL_RGB[st], size=26, pad=(16, 6))
    tot = sum(s["posture_time"]) or 1.0
    p.text((x + 200, ty + 6), "tempo na sessão", 12, P["sub"])
    bx, bw = x + 200, p.w - x - 216
    cx = bx
    for frac, c in zip((t / tot for t in s["posture_time"]), LEVEL_RGB):
        p.d.rectangle((cx, ty + 28, cx + int(bw * frac), ty + 38), fill=c)
        cx += int(bw * frac)
    yy = ty + 70
    for k, v in s["posture_metrics"].items():
        warn, bad = s["thresholds"][k]
        p.text((x, yy), METRIC_LABEL.get(k, k), 16, P["text"], True)
        p.text((p.w - 36, yy), fmt(v, nd=1), 16, P["text"], True, anchor="ra")
        p.bar(x, yy + 28, p.w - x - 36, v, warn, bad)
        yy += 58
    if not s["posture_metrics"]:
        p.text((x, yy), "sem métricas: mostre ombros e orelha", 14, P["sub"])
    return y + 300


def _angle_card(p, y, s):
    x, ty = p.card(y, 300, "ângulo articular")
    p.text((x, ty), joint_label(s["gonio_joint"]), 18, P["sub"], True)
    if s["depth"]:   # 2D and 3D side by side, so the difference is easy to see
        p.text((x, ty + 30), "2D", 14, P["accent"], True)
        p.text((x + 22, ty + 22), fmt(s["gonio_value"]), 54, P["accent"], True)
        p.text((x + 150, ty + 30), "3D", 14, P["accent2"], True)
        p.text((x + 172, ty + 22), fmt(s["gonio_3d"]), 54, P["accent2"], True)
    else:
        p.text((x - 4, ty + 22), fmt(s["gonio_value"]), 72, P["text"], True)
    col = p.w - 170 if not s["depth"] else p.w - 120
    for i, (lab, v) in enumerate((("mín", s["gonio_min"]), ("máx", s["gonio_max"]), ("ADM", s["gonio_rom"]))):
        p.text((col, ty + 6 + 30 * i), lab, 14, P["sub"])
        p.text((p.w - 36, ty + 4 + 30 * i), fmt(v), 18, P["accent"] if lab == "ADM" else P["text"], True, anchor="ra")
    extra = ((s["hist_angle3d"], P["accent2"]),) if s["depth"] else ()
    p.spark(x, ty + 140, p.w - x - 32, 108, s["hist_angle"], lo=0, hi=max(160, s.get("gonio_max", 0) or 0), extra=extra)
    p.text((x + 8, ty + 144), "últimos 10 s" + (" · azul 2D, roxo 3D" if s["depth"] else ""), 11, P["sub"])
    return y + 300


def _reps_card(p, y, s):
    x, ty = p.card(y, 300, f"exercício · {joint_label(s['rep_joint']).lower()}")
    p.text((x, ty), EXERCISE_LABEL.get(s["exercise"], s["exercise"]), 18, P["sub"], True)
    p.text((x - 4, ty + 22), str(s["reps"]), 72, P["text"], True)
    p.text((x + 26 + 44 * len(str(s["reps"])), ty + 66), "repetições", 16, P["sub"])
    p.chip(p.w - 150, ty + 4, f"fase: {s['rep_phase']}", P["accent"], filled=False, size=13)
    lo, hi = s["rep_thresholds"]
    p.spark(x, ty + 140, p.w - x - 32, 108, s["hist_rep"], lines=((lo, P["ok"]), (hi, P["warn"])), lo=0, hi=hi + 60)
    p.text((x + 8, ty + 144), "ângulo · limiares de repetição", 11, P["sub"])
    return y + 300


def _fall_card(p, y, s):
    x, ty = p.card(y, 300, "detecção de queda")
    fs = s["fall_state"]
    p.chip(x, ty + 2, FALL_LABEL.get(fs, fs), P["bad"] if fs == "QUEDA" else P["warn"] if fs != "normal" else P["ok"],
           size=26, pad=(16, 6))
    p.text((p.w - 36, ty + 6), f"quedas: {s['falls']}", 16, P["text"], True, anchor="ra")
    p.text((x, ty + 62), "velocidade de descida do quadril", 13, P["sub"])
    p.text((p.w - 36, ty + 58), f"{s['fall_speed']:+.2f} alt/s", 16, P["text"], True, anchor="ra")
    p.spark(x, ty + 88, p.w - x - 32, 120, s["hist_speed"], lines=((s["drop_speed"], P["bad"]),), lo=-1.0,
            hi=max(1.6, s["drop_speed"] * 1.5), color=P["warn"])
    p.text((x, ty + 220), "queda = descida rápida + tronco horizontal + 2 s parado", 12, P["sub"])
    return y + 300
