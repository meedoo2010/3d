import os
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

FRAME_W, FRAME_H = 960, 540
WINDOW_NAME = "RETROLENS Poke Python"
SWITCH_COOLDOWN = 0.9
PINCH_THRESHOLD = 0.35

FILTER_NAMES = [
    "dual-tone", "thermal", "sketch", "pixelate", "glitch", "invert",
    "red-channel", "edges", "blur", "cartoon", "rainbow-wave",
]

MODES = ["3D (Full 5 Jari)", "PERSEKTIF 2D"]

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

THUMB, INDEX, MIDDLE, RING, PINKY = 4, 8, 12, 16, 20
FINGERTIPS = [THUMB, INDEX, MIDDLE, RING, PINKY]


def f_dual_tone(img, t):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    c1 = np.array([160, 40, 255], np.float32)
    c2 = np.array([255, 200, 0], np.float32)
    a = (gray.astype(np.float32) / 255.0)[..., None]
    return (c1 * (1 - a) + c2 * a).astype(np.uint8)


def f_thermal(img, t):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def f_sketch(img, t):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def f_pixelate(img, t, block=14):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // block), max(1, h // block)),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def f_glitch(img, t):
    out = img.copy()
    h, w = out.shape[:2]
    rng = np.random.default_rng(int(t * 12))
    shift = int(rng.integers(6, 18))
    out[:, :, 2] = np.roll(out[:, :, 2], shift, axis=1)
    out[:, :, 0] = np.roll(out[:, :, 0], -shift, axis=1)
    for _ in range(8):
        y = int(rng.integers(0, h - 12))
        hh = int(rng.integers(4, 12))
        out[y:y + hh] = np.roll(out[y:y + hh], int(rng.integers(-40, 40)), axis=1)
    return out


def f_invert(img, t):
    return cv2.bitwise_not(img)


def f_red_channel(img, t):
    out = np.zeros_like(img)
    out[:, :, 2] = img[:, :, 2]
    return out


def f_edges(img, t):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 140)
    out = np.zeros_like(img)
    out[edges > 0] = (0, 255, 255)
    return out


def f_blur(img, t):
    return cv2.GaussianBlur(img, (0, 0), 14)


def f_cartoon(img, t):
    color = cv2.bilateralFilter(img, 9, 75, 75)
    color = (color // 48) * 48 + 24
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 7)
    return cv2.bitwise_and(color.astype(np.uint8), color.astype(np.uint8), mask=edges)


_grid_cache = {}


def f_rainbow_wave(img, t):
    h, w = img.shape[:2]
    if (h, w) not in _grid_cache:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        _grid_cache[(h, w)] = (xx, yy)
    xx, yy = _grid_cache[(h, w)]
    hue = ((xx + yy) / 6.0 + t * 90.0) % 180.0
    hsv = np.dstack([hue, np.full_like(hue, 255), np.full_like(hue, 255)]).astype(np.uint8)
    rainbow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.addWeighted(img, 0.35, rainbow, 0.65, 0)


FILTERS = {
    "dual-tone": f_dual_tone, "thermal": f_thermal, "sketch": f_sketch,
    "pixelate": f_pixelate, "glitch": f_glitch, "invert": f_invert,
    "red-channel": f_red_channel, "edges": f_edges, "blur": f_blur,
    "cartoon": f_cartoon, "rainbow-wave": f_rainbow_wave,
}


def hand_points(hand_lms, idxs):
    return [(int(hand_lms.landmark[i].x * FRAME_W),
             int(hand_lms.landmark[i].y * FRAME_H)) for i in idxs]


def is_pinky_thumb_touch(hand_lms):
    """لمسة الإبهام بالخنصر = أمر تغيير الفلتر."""
    lm = hand_lms.landmark
    size = np.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y) + 1e-6
    d = np.hypot(lm[THUMB].x - lm[PINKY].x, lm[THUMB].y - lm[PINKY].y)
    return (d / size) < PINCH_THRESHOLD


def order_quad(pts):
    """ترتيب 4 نقاط: أعلى-يسار، أعلى-يمين، أسفل-يمين، أسفل-يسار."""
    pts = np.array(pts, np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def build_region(hands, mode):
    """بيرجّع (mask, polygon) حسب الوضع، أو (None, None) لو الإيدين مش كفاية."""
    if len(hands) < 2:
        return None, None

    if mode == 0:
        pts = []
        for h in hands[:2]:
            pts += hand_points(h, FINGERTIPS)
        poly = cv2.convexHull(np.array(pts, np.int32))
    else:
        pts = []
        for h in hands[:2]:
            pts += hand_points(h, [THUMB, INDEX])
        poly = order_quad(pts).astype(np.int32)

    mask = np.zeros((FRAME_H, FRAME_W), np.uint8)
    cv2.fillConvexPoly(mask, poly, 255)
    return mask, poly


class Hand:
    """غلاف بسيط عشان باقي الكود يفضل يستخدم hand.landmark[i].x/.y"""
    def __init__(self, landmarks):
        self.landmark = landmarks


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("بنزّل موديل الإيدين لأول مرة...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("تم:", MODEL_PATH)


def draw_hand(img, hand):
    pts = [(int(p.x * FRAME_W), int(p.y * FRAME_H)) for p in hand.landmark]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (0, 200, 255), 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 3, (0, 220, 255), -1, cv2.LINE_AA)


def put_text(img, text, org, color, scale=0.55):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        raise SystemExit("مش قادر أفتح الكاميرا. تأكد إن مفيش برنامج تاني بيستخدمها.")

    ensure_model()
    with open(MODEL_PATH, "rb") as f:
        model_bytes = f.read()
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    mode = 0
    fidx = 0
    last_switch = 0.0
    t0 = time.time()
    last_ts = -1

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(cv2.resize(frame, (FRAME_W, FRAME_H)), 1)
            t = time.time() - t0

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = max(int(t * 1000), last_ts + 1)
            last_ts = ts
            res = landmarker.detect_for_video(mp_image, ts)
            hand_list = [Hand(lms) for lms in res.hand_landmarks]

            now = time.time()
            if now - last_switch > SWITCH_COOLDOWN:
                for h in hand_list:
                    if is_pinky_thumb_touch(h):
                        fidx = (fidx + 1) % len(FILTER_NAMES)
                        last_switch = now
                        break

            out = frame.copy()
            mask, poly = build_region(hand_list, mode)
            if mask is not None:
                filtered = FILTERS[FILTER_NAMES[fidx]](frame, t)
                m3 = cv2.merge([mask, mask, mask]).astype(bool)
                out[m3] = filtered[m3]
                cv2.polylines(out, [poly], True, (255, 255, 255), 2, cv2.LINE_AA)

            for h in hand_list:
                draw_hand(out, h)

            put_text(out, f"MODE: {MODES[mode]} [Tekan 'c' / Kopi 2 Tangan]",
                     (10, 24), (0, 220, 255), 0.6)
            put_text(out, f"FILTER: {FILTER_NAMES[fidx].upper()} "
                          f"[Sentuh Jempol-Kelingking / 'n' / 'p']",
                     (10, 50), (255, 255, 255), 0.5)
            if mask is None:
                put_text(out, "Tunjukkan 2 tangan ke kamera", (10, FRAME_H - 15),
                         (120, 255, 120), 0.55)

            cv2.imshow(WINDOW_NAME, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                mode = (mode + 1) % len(MODES)
            elif key == ord("n"):
                fidx = (fidx + 1) % len(FILTER_NAMES)
            elif key == ord("p"):
                fidx = (fidx - 1) % len(FILTER_NAMES)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
