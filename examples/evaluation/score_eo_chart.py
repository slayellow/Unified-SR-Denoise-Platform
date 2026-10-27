"""Score a denoise checkpoint on the meeting-room chart clips (EO Denoise v3, protocol P1).

Each clip is raw NV12 1920x1080 from capture_eo_raw.sh. For every selected frame the
denoiser runs on a native 1280x720 window around the chart, as on the board, and the
same measurements are taken on the camera input and on the output: flat-area noise of
gray patch P4, overshoot/undershoot at the top edges of P2 and P3, MTF50 of the slanted
black square, chroma noise of P3 and the chroma of the two smallest text rows (false
colour), and the residual at measured sensor hot pixels. Edge lines are fitted once on the
camera input and reused for the output; over/undershoot comes from frame-averaged edge
profiles (protocol P1.1; P1 fitted them per noisy frame and read a perfect denoiser as adding
halo). Per clip the frame medians are reported with output-minus-input overshoot, output/input
MTF50 and output/input noise. Guards off the chart: dot retention through the camera model and
texture ratio on clean images. See docs/modules/20261008-MOD-ChartScore.md.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=. python3 examples/evaluation/score_eo_chart.py \
        --checkpoint results/denoise_release_v1/checkpoint/best.pth --json out.json
"""

import argparse
import hashlib
import json
import os

import cv2
import numpy as np
import torch
import yaml
from scipy.ndimage import shift as subshift

from examples.evaluation.score_grid_artifact import load_model
from src.data.degradations import apply_camera_model, ycbcr601_to_bgr

W, H = 1920, 1080
FRAME_BYTES = W * H * 3 // 2
WIN_W, WIN_H = 1280, 720

# Chart geometry in mm from the page's top-left corner (results/denoise_release_v1/eval/kit/*.pdf).
CHARTS = {
    "z10": dict(
        square=(95.17, 97.67),
        side=16.0,
        centre=(148.5, 105.0),
        patches=[(119.5 + 14 * i, 132.0, 12.0, 10.0) for i in range(5)],
        small_text=(142.5, 111.8, 154.5, 117.2),
    ),
    "z01": dict(
        square=(41.5, 66.5),
        side=36.0,
        centre=(148.5, 105.0),
        patches=[(84.0 + 32 * i, 183.5, 28.0, 23.0) for i in range(5)],
        small_text=(136.5, 110.5, 160.5, 118.5),
    ),
}
DEFAULT_CLIPS = [
    "z10_a9=results/denoise_release_v1/eval/z10_a9.nv12",
    "z01_a9=results/denoise_release_v1/eval/z01_a9.nv12",
    "z10_a9_nr3doff=results/denoise_release_v1/eval/z10_a9_nr3doff.nv12",
]
DEFAULT_MANIFEST = "results/denoise_release_v1/eval/P1_clips.sha256"
DEFAULT_HOT_PIXELS = "results/denoise_release_v1/analysis/noise_census/hot_pixels_a0.json"
DEFAULT_CAMERA_CONFIG = "configs/data/bicubicpp/denoise.yaml"
DEFAULT_TEXTURE_ROOT = "/mnt/data2/jshong/SuperResolution/Pretrained_Dataset/val_denoise/HR"


# ---------- geometry ----------
def find_square(y: np.ndarray, side_mm: float) -> tuple:
    """Largest dark ~square blob tilted 5 +- 3 deg -> (centre_px, mm_per_px, contour); ValueError if none."""
    mask = (y < 80).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    best = None
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 400:
            continue
        cnt, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(cnt[0])
        tilt = min(abs(ang % 90 - 5), abs(ang % 90 - 85))
        if 0.85 < rw / rh < 1.15 and st[i, cv2.CC_STAT_AREA] / (rw * rh) > 0.9 and tilt < 3:
            if best is None or rw * rh > best[2]:
                best = ((cx, cy), side_mm / ((rw + rh) / 2), rw * rh, cnt[0])
    if best is None:
        raise ValueError("no tilted black square found: is the chart in the frame?")
    return best[0], best[1], best[3]


def mm_to_px(pt: tuple, c_px: tuple, c_mm: tuple, scale: float) -> tuple:
    return c_px[0] + (pt[0] - c_mm[0]) / scale, c_px[1] + (pt[1] - c_mm[1]) / scale


def box_px(box_mm: tuple, c_px: tuple, c_mm: tuple, scale: float, shrink: float = 0.0) -> tuple:
    x0, y0 = mm_to_px(box_mm[:2], c_px, c_mm, scale)
    x1, y1 = mm_to_px(box_mm[2:], c_px, c_mm, scale)
    dx, dy = (x1 - x0) * shrink / 2, (y1 - y0) * shrink / 2
    return int(x0 + dx), int(y0 + dy), int(x1 - dx), int(y1 - dy)


def patch_box(patch: tuple) -> tuple:
    cx, cy, w, h = patch
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


# ---------- measurements ----------
def edge_mtf(y: np.ndarray, contour: np.ndarray, axis: int) -> dict:
    """Slanted-edge MTF50 (cycles/px) and over/undershoot (%) of the square's left (axis=1) or top (axis=0) edge."""
    img = y.astype(np.float32) if axis == 1 else y.T.astype(np.float32)
    pts = contour[:, 0, :] if axis == 1 else contour[:, 0, ::-1]
    rows_all = pts[:, 1]
    span = rows_all.max() - rows_all.min()
    rows = np.arange(int(rows_all.min() + 0.25 * span), int(rows_all.max() - 0.25 * span))
    lefts = np.array([pts[pts[:, 1] == r][:, 0].min() if np.any(pts[:, 1] == r) else np.nan for r in rows], float)
    c0, c1 = int(np.nanmin(lefts)) - 24, int(np.nanmax(lefts)) + 24
    roi = img[rows[0] : rows[-1] + 1, c0:c1]
    rr, cc = np.mgrid[rows[0] : rows[-1] + 1, c0:c1]
    # Edge line from the threshold contour, the same for input and output: per-row
    # gradient centroids on a noisy input smear its profile and bias every ratio.
    ok = ~np.isnan(lefts)
    a, b = np.polyfit(rows[ok], lefts[ok], 1)
    d = (cc - (a * rr + b)) * np.cos(np.arctan(a))
    bins = np.arange(-20, 20.25, 0.25)
    idx = np.digitize(d.ravel(), bins)
    vals = roi.ravel()
    prof = np.array([vals[idx == k].mean() if np.any(idx == k) else np.nan for k in range(1, len(bins))])
    centres = bins[:-1] + 0.125
    ok = ~np.isnan(prof)
    prof = np.interp(centres, centres[ok], prof[ok])
    if prof[: len(prof) // 4].mean() > prof[-len(prof) // 4 :].mean():
        prof = prof[::-1]  # orient dark -> bright
    lo, hi = np.median(prof[centres < -12]), np.median(prof[centres > 12])
    e = (prof - lo) / (hi - lo)
    lsf = np.gradient(e) * np.hanning(len(e))
    mtf = np.abs(np.fft.rfft(lsf))
    mtf /= mtf[0]
    f = np.fft.rfftfreq(len(lsf), d=0.25)
    k = int(np.argmax(mtf < 0.5))
    mtf50 = f[k - 1] + (f[k] - f[k - 1]) * (mtf[k - 1] - 0.5) / (mtf[k - 1] - mtf[k]) if k > 0 else float("nan")
    over, under = esf_overshoot(e)
    return dict(mtf50=float(mtf50), over_pct=over, under_pct=under, esf=e)


ESF_CENTRES = np.arange(-20, 20, 0.25) + 0.125


def esf_overshoot(e: np.ndarray) -> tuple:
    """(over %, under %) of a normalised dark -> bright edge spread function sampled on ESF_CENTRES."""
    over = 100 * max(0.0, float(e[(ESF_CENTRES > 0) & (ESF_CENTRES < 12)].max()) - 1)
    under = 100 * max(0.0, -float(e[(ESF_CENTRES < 0) & (ESF_CENTRES > -12)].min()))
    return over, under


def _edge_strip(y: np.ndarray, box: tuple, half: int) -> tuple:
    x0, y0, x1, _ = box
    w = x1 - x0
    cols = np.arange(int(x0 + 0.1 * w), int(x1 - 0.1 * w))
    return cols, y[y0 - half - 4 : y0 + half + 4, cols].astype(np.float64).T


def _step_position(profile: np.ndarray) -> float:
    """Sub-pixel edge position of a 1-D profile: gradient centroid within 3 px of the strongest gradient."""
    g = np.abs(np.diff(profile))
    k = int(np.argmax(g))
    lo, hi = max(0, k - 3), min(len(g), k + 4)
    w = g[lo:hi]
    return float((w * (np.arange(lo, hi) + 0.5)).sum() / max(w.sum(), 1e-6))


def top_edge_line(y: np.ndarray, boxes: list, half: int = 12, blocks: int = 6) -> tuple:
    """(slope, offset) in image rows of a straight top edge shared by one or more patches.

    Each patch strip is split into column blocks; a block's mean profile gives one edge position,
    and one line is fitted through all of them (the chart's patches share one top edge).
    """
    xs, pos = [], []
    for box in boxes:
        cols, strip = _edge_strip(y, box, half)
        for part in np.array_split(np.arange(len(cols)), blocks):
            xs.append(cols[part].mean())
            pos.append(_step_position(strip[part].mean(0)) + box[1] - half - 4)
    a, b = np.polyfit(xs, pos, 1)
    return float(a), float(b)


def top_edge_profile(y: np.ndarray, box: tuple, half: int = 12, line: tuple = None) -> np.ndarray:
    """Mean column profile across a patch's top edge, every column shifted so the edge line sits at index half + 4.

    The line is fitted here, or the camera input's line is passed in for the output, so noise in
    single columns does not smear the profile and every frame's profile shares one edge index.
    """
    cols, strip = _edge_strip(y, box, half)
    a, b = top_edge_line(y, [box], half) if line is None else line
    pos = a * cols + b - (box[1] - half - 4)
    return np.array([subshift(r, (half + 4) - p, order=1, mode="nearest") for r, p in zip(strip, pos)]).mean(0)


def edge_overshoot(prof: np.ndarray, half: int = 12) -> tuple:
    """(overshoot %, undershoot %) of a top_edge_profile: paper above, darker patch below."""
    i = half + 4
    bright = np.median(prof[i - half : i - 6])
    dark = np.median(prof[i + 7 : i + half + 1])
    step = bright - dark
    if step <= 0:
        raise ValueError("patch edge has no paper-to-patch step; check the chart geometry")
    over = max(0.0, float(prof[i - 6 : i + 1].max() - bright)) / step
    under = max(0.0, float(dark - prof[i : i + 7].min())) / step
    return 100 * over, 100 * under


def top_edge_overshoot(y: np.ndarray, box: tuple, half: int = 12, line: tuple = None) -> tuple:
    """(overshoot %, undershoot %) at a patch's top edge of one image."""
    return edge_overshoot(top_edge_profile(y, box, half, line), half)


def _highpass_sd(plane: np.ndarray, box: tuple) -> float:
    x0, y0, x1, y1 = box
    p = plane[y0:y1, x0:x1].astype(np.float32)
    return float((p - cv2.GaussianBlur(p, (0, 0), 1.5)).std())


def flat_noise(y: np.ndarray, box: tuple) -> float:
    """Luma high-pass sd inside box (the caller shrinks the box away from the patch border)."""
    return _highpass_sd(y, box)


def chroma_noise(bgr: np.ndarray, box: tuple) -> float:
    """Mean of the Cr and Cb high-pass sd inside box."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return 0.5 * (_highpass_sd(ycc[..., 1], box) + _highpass_sd(ycc[..., 2], box))


def patch_lines(bgr: np.ndarray, chart: dict, c_px: tuple, scale: float) -> dict:
    """Top-edge line of patches P2 and P3, fitted once on the camera input and reused for the output.

    All five patches share one printed top edge; the high-contrast P0-P3 fix the line.
    """
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[..., 0]
    boxes = [box_px(patch_box(chart["patches"][i]), c_px, chart["square"], scale) for i in range(4)]
    line = top_edge_line(y, boxes)
    return {2: line, 3: line}


def measure(bgr: np.ndarray, chart: dict, c_px: tuple, scale: float, contour: np.ndarray, lines: dict) -> dict:
    """All P1 measurements on one image (window coordinates); lines from patch_lines() on the input."""
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[..., 0]
    c_mm = chart["square"]
    patches = [box_px(patch_box(p), c_px, c_mm, scale) for p in chart["patches"]]
    prof2 = top_edge_profile(y, patches[2], line=lines[2])
    prof3 = top_edge_profile(y, patches[3], line=lines[3])
    p2o, p2u = edge_overshoot(prof2)
    p3o, p3u = edge_overshoot(prof3)
    eh, ev = edge_mtf(y, contour, 1), edge_mtf(y, contour, 0)
    return dict(
        _prof_p2=prof2,
        _prof_p3=prof3,
        _esf_h=eh["esf"],
        _esf_v=ev["esf"],
        flat_noise_p4=flat_noise(y, box_px(patch_box(chart["patches"][4]), c_px, c_mm, scale, shrink=0.4)),
        over_p2=p2o,
        under_p2=p2u,
        over_p3=p3o,
        under_p3=p3u,
        mtf50_h=eh["mtf50"],
        mtf50_v=ev["mtf50"],
        square_over_h=eh["over_pct"],
        square_over_v=ev["over_pct"],
        chroma_noise_p3=chroma_noise(bgr, box_px(patch_box(chart["patches"][3]), c_px, c_mm, scale, shrink=0.4)),
        false_colour=chroma_noise(bgr, box_px(chart["small_text"], c_px, c_mm, scale)),
    )


def _luma(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)


def hot_residual(bgr: np.ndarray, xy: np.ndarray) -> float:
    """Median luma residual (pixel minus its 5x5 median) at the given (x, y) positions."""
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[..., 0]
    r = y.astype(np.float32) - cv2.medianBlur(y, 5).astype(np.float32)
    return float(np.median(r[xy[:, 1], xy[:, 0]]))


# ---------- guards (not on the chart) ----------
def dot_retention(run, camera_cfg: dict, sizes=(1, 2, 3), amps=(20, 60, 150), seeds: int = 5, level: float = 120.0) -> dict:
    """Bright square dots on flat gray through the camera model and the denoiser: output peak / source peak (luma).

    The same noise draw is rendered with and without the dot, so the difference isolates the dot.
    Also reports the camera input's own peak ratio ('camera_*') for reference.
    """
    g = float(ycbcr601_to_bgr(np.array([[[level, 128.0, 128.0]]], np.float32))[0, 0, 0])
    base = np.full((96, 96, 3), g, np.float32)
    c = 48
    out = {}
    for size in sizes:
        for amp in amps:
            dot = base.copy()
            dot[c : c + size, c : c + size] += amp
            src0, src1 = np.round(base).astype(np.uint8), np.clip(np.round(dot), 0, 255).astype(np.uint8)
            src_peak = float((_luma(src1) - _luma(src0)).max())
            ratios, cam = [], []
            for k in range(seeds):
                lr0 = apply_camera_model(src0, camera_cfg, rng=np.random.default_rng(k))[0]
                lr1 = apply_camera_model(src1, camera_cfg, rng=np.random.default_rng(k))[0]
                win = (slice(c - 3, c + size + 3), slice(c - 3, c + size + 3))
                cam.append(float((_luma(lr1) - _luma(lr0))[win].max()) / src_peak)
                ratios.append(float((_luma(run(lr1)) - _luma(run(lr0)))[win].max()) / src_peak)
            out[f"dot{size}px_{amp}"] = float(np.median(ratios))
            out[f"camera_dot{size}px_{amp}"] = float(np.median(cam))
    return out


def texture_ratio(run, root: str, n: int = 10, crop: int = 512) -> float:
    """Median Laplacian-variance ratio output/input on centre crops of clean images (1 = untouched, < 1 = smoothed)."""
    names = [m for m in sorted(os.listdir(root)) if os.path.splitext(m)[1].lower() in (".png", ".jpg", ".jpeg", ".bmp")][:n]
    if not names:
        raise FileNotFoundError(f"no images under {root}")
    ratios = []
    for name in names:
        img = cv2.imread(os.path.join(root, name), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        y0, x0 = max(0, (h - crop) // 2), max(0, (w - crop) // 2)
        img = np.ascontiguousarray(img[y0 : y0 + crop, x0 : x0 + crop])
        lap = lambda b: float(cv2.Laplacian(_luma(b), cv2.CV_32F).var())
        ratios.append(lap(run(img)) / max(lap(img), 1e-6))
    return float(np.median(ratios))


def chart_window(bgr: np.ndarray, chart: dict) -> tuple:
    """Full frame -> (x0, y0, square centre px, mm/px, square contour) of the 1280x720 window around the chart centre."""
    y_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[..., 0]
    c_px, scale, contour = find_square(y_full, chart["side"])
    cx, cy = mm_to_px(chart["centre"], c_px, chart["square"], scale)
    x0 = int(min(max(cx - WIN_W // 2, 0), bgr.shape[1] - WIN_W))
    y0 = int(min(max(cy - WIN_H // 2, 0), bgr.shape[0] - WIN_H))
    return x0, y0, c_px, scale, contour


def score_frame(bgr: np.ndarray, chart: dict, model_fn, hot_xy: np.ndarray = None) -> dict:
    """Camera input vs denoise output on the 1280x720 window around the chart centre of one full frame.

    hot_xy: full-frame (x, y) of sensor hot pixels; those inside the window add 'hot_residual'.
    """
    x0, y0, c_px, scale, contour = chart_window(bgr, chart)
    win = np.ascontiguousarray(bgr[y0 : y0 + WIN_H, x0 : x0 + WIN_W])
    c_win = (c_px[0] - x0, c_px[1] - y0)
    contour_win = contour - np.array([x0, y0], dtype=contour.dtype)
    out = model_fn(win)
    lines = patch_lines(win, chart, c_win, scale)
    cam = measure(win, chart, c_win, scale, contour_win, lines)
    dn = measure(out, chart, c_win, scale, contour_win, lines)
    if hot_xy is not None and len(hot_xy):
        xy = np.asarray(hot_xy) - np.array([x0, y0])
        xy = xy[(xy[:, 0] >= 3) & (xy[:, 0] < WIN_W - 3) & (xy[:, 1] >= 3) & (xy[:, 1] < WIN_H - 3)]
        if len(xy):
            cam["hot_residual"], dn["hot_residual"] = hot_residual(win, xy), hot_residual(out, xy)
    return dict(mm_per_px=scale, camera=cam, denoise=dn)


def _side_summary(frames: list, side: str) -> dict:
    """Frame medians of the scalar measurements; over/undershoot from the frame-mean edge profiles.

    Taking the extreme of each noisy frame profile would inflate the camera input's overshoot,
    so the profiles are averaged across frames first and measured once.
    """
    rows = [f[side] for f in frames]
    out = {k: float(np.median([r[k] for r in rows])) for k in rows[0] if not k.startswith("_")}
    out["over_p2"], out["under_p2"] = edge_overshoot(np.mean([r["_prof_p2"] for r in rows], 0))
    out["over_p3"], out["under_p3"] = edge_overshoot(np.mean([r["_prof_p3"] for r in rows], 0))
    out["square_over_h"], _ = esf_overshoot(np.mean([r["_esf_h"] for r in rows], 0))
    out["square_over_v"], _ = esf_overshoot(np.mean([r["_esf_v"] for r in rows], 0))
    return out


def summarize(frames: list) -> dict:
    """Camera and denoise summaries (_side_summary), plus output-minus-input overshoot and output/input ratios."""
    cam = _side_summary(frames, "camera")
    dn = _side_summary(frames, "denoise")
    ratio = lambda a, b: a / b if b else float("nan")
    derived = dict(
        added_over_p2=dn["over_p2"] - cam["over_p2"],
        added_over_p3=dn["over_p3"] - cam["over_p3"],
        added_under_p2=dn["under_p2"] - cam["under_p2"],
        added_under_p3=dn["under_p3"] - cam["under_p3"],
        added_square_over_h=dn["square_over_h"] - cam["square_over_h"],
        added_square_over_v=dn["square_over_v"] - cam["square_over_v"],
        mtf50_ratio_h=ratio(dn["mtf50_h"], cam["mtf50_h"]),
        mtf50_ratio_v=ratio(dn["mtf50_v"], cam["mtf50_v"]),
        noise_ratio_p4=ratio(dn["flat_noise_p4"], cam["flat_noise_p4"]),
        chroma_ratio_p3=ratio(dn["chroma_noise_p3"], cam["chroma_noise_p3"]),
        false_colour_ratio=ratio(dn["false_colour"], cam["false_colour"]),
    )
    if "hot_residual" in cam:
        derived["hot_residual_ratio"] = ratio(dn["hot_residual"], cam["hot_residual"])
    return dict(camera=cam, denoise=dn, derived=derived, mm_per_px=float(np.median([f["mm_per_px"] for f in frames])))


# ---------- clips and model ----------
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def select_frames(path: str, spec: str, count: int, min_frames: int) -> list:
    """Frame indices start::step (first `count`), checked against the clip length; ValueError when too few."""
    size = os.path.getsize(path)
    if size == 0 or size % FRAME_BYTES:
        raise ValueError(f"{path}: {size} bytes is not a whole number of NV12 1920x1080 frames")
    n = size // FRAME_BYTES
    start, step = (int(v) for v in spec.split("::"))
    idx = list(range(start, n, step))[:count]
    if len(idx) < max(min_frames, 1):
        raise ValueError(f"{path}: {len(idx)} frames selected by {spec}, protocol needs {min_frames}")
    return idx


def read_frame(path: str, i: int) -> np.ndarray:
    with open(path, "rb") as f:
        f.seek(i * FRAME_BYTES)
        nv12 = np.frombuffer(f.read(FRAME_BYTES), np.uint8).reshape(H * 3 // 2, W)
    return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)


def denoiser(model, device: str):
    """BGR uint8 -> BGR uint8 through the model in FP32, rounded to 8 bit like the deployed path."""

    def run(bgr):
        t = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1) / 255.0)[None]
        with torch.no_grad():
            o = model(t.to(device))[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        return cv2.cvtColor((o * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)

    return run


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="denoise checkpoint with its model config saved in it (every Trainer checkpoint)")
    parser.add_argument(
        "--clips", nargs="+", default=DEFAULT_CLIPS, help="name=path; name starts with the chart id (z10 / z01)"
    )
    parser.add_argument("--frames", default="20::10", help="start::step frame selection")
    parser.add_argument("--n_frames", type=int, default=10, help="frames per clip taken from the selection")
    parser.add_argument("--min_frames", type=int, default=10, help="reject a clip with fewer selected frames")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="sha256 list of the clips ('' skips the check)")
    parser.add_argument("--hot_pixels", default=DEFAULT_HOT_PIXELS, help="json with full-frame 'xy' of sensor hot pixels ('' skips)")
    parser.add_argument("--camera_config", default=DEFAULT_CAMERA_CONFIG, help="data config whose camera_model renders the dot guard ('' skips)")
    parser.add_argument("--texture_root", default=DEFAULT_TEXTURE_ROOT, help="clean images for the texture guard ('' skips)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", default="", help="write the result here")
    args = parser.parse_args(args)

    clips = {}
    for item in args.clips:
        name, path = item.split("=", 1)
        chart = name[:3]
        if chart not in CHARTS:
            raise ValueError(f"clip name {name!r} must start with one of {sorted(CHARTS)}")
        clips[name] = dict(
            path=path, chart=chart, frames=select_frames(path, args.frames, args.n_frames, args.min_frames)
        )
    if args.manifest:
        with open(args.manifest) as f:
            expected = {line.split()[1]: line.split()[0] for line in f if line.strip()}
        for name, c in clips.items():
            key = os.path.basename(c["path"])
            if expected.get(key) != (digest := sha256(c["path"])):
                raise ValueError(f"{c['path']}: sha256 {digest[:12]} does not match the manifest {args.manifest}")
            c["sha256"] = digest
    hot_xy = None
    if args.hot_pixels:
        with open(args.hot_pixels) as f:
            hot_xy = np.array(json.load(f)["xy"], dtype=int).reshape(-1, 2)
    camera_cfg = None
    if args.camera_config:
        with open(args.camera_config) as f:
            camera_cfg = yaml.safe_load(f)["camera_model"]
    if args.texture_root and not os.path.isdir(args.texture_root):
        raise FileNotFoundError(f"--texture_root {args.texture_root} is not a directory")

    model, _, epoch = load_model(args.checkpoint, args.device)
    run = denoiser(model, args.device)
    result = dict(
        protocol="P1.1",
        checkpoint=args.checkpoint,
        epoch=epoch,
        checkpoint_md5=hashlib.md5(open(args.checkpoint, "rb").read()).hexdigest(),
        clips={},
    )
    for name, c in clips.items():
        frames = [score_frame(read_frame(c["path"], i), CHARTS[c["chart"]], run, hot_xy) for i in c["frames"]]
        result["clips"][name] = dict(summarize(frames), frames=c["frames"], sha256=c.get("sha256"))
        d = result["clips"][name]
        print(
            f"{name:16s} noise {d['camera']['flat_noise_p4']:.2f} -> {d['denoise']['flat_noise_p4']:.2f} | "
            f"added over P2 {d['derived']['added_over_p2']:+.1f} P3 {d['derived']['added_over_p3']:+.1f} %p | "
            f"MTF50 ratio {d['derived']['mtf50_ratio_h']:.3f}/{d['derived']['mtf50_ratio_v']:.3f} | "
            f"chroma {d['derived']['chroma_ratio_p3']:.2f} | false colour {d['derived']['false_colour_ratio']:.2f} | "
            f"square added {d['derived']['added_square_over_h']:+.1f}/{d['derived']['added_square_over_v']:+.1f} %p | "
            f"hot residual {d['derived'].get('hot_residual_ratio', float('nan')):.2f}"
        )
    guards = {}
    if camera_cfg is not None:
        guards.update(dot_retention(run, camera_cfg))
    if args.texture_root:
        guards["texture_ratio"] = texture_ratio(run, args.texture_root)
    result["guards"] = guards
    if guards:
        print("guards " + " ".join(f"{k} {v:.2f}" for k, v in guards.items() if not k.startswith("camera_")))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
    print("Score Finish!")


if __name__ == "__main__":
    main()
