"""ChartScore (MOD-CHART): P1 chart measurements for EO Denoise v3.

TC ids follow docs/modules/20261008-MOD-ChartScore.md section 4.
"""

import hashlib
import json
import math
import os

import cv2
import numpy as np
import pytest
import yaml

from examples.evaluation.score_eo_chart import (
    CHARTS,
    DEFAULT_CLIPS,
    DEFAULT_HOT_PIXELS,
    DEFAULT_MANIFEST,
    DEFAULT_TEXTURE_ROOT,
    FRAME_BYTES,
    box_px,
    chart_window,
    dot_retention,
    edge_mtf,
    find_square,
    flat_noise,
    main,
    mm_to_px,
    patch_box,
    score_frame,
    summarize,
    texture_ratio,
    top_edge_overshoot,
)

SCALE = 0.166  # mm/px, the 10x / 3 m capture


def chart_frame(scale=SCALE, blur=0.8, noise=0.0, seed=0):
    """Synthetic 1920x1080 BGR frame of the 10x chart: paper, tilted black square, five gray patches."""
    chart = CHARTS["z10"]
    ss = 4
    h, w = 1080 * ss, 1920 * ss
    img = np.full((h, w), 160.0, np.float32)
    centre = np.array([960.0, 540.0])

    def to_px(mx, my):
        return (centre + (np.array([mx, my]) - np.array(chart["centre"])) / scale) * ss

    sq = to_px(*chart["square"])
    half = chart["side"] / 2 / scale * ss
    t = np.deg2rad(5.0)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    corners = (np.array([[-half, -half], [half, -half], [half, half], [-half, half]]) @ rot.T + sq).astype(np.int32)
    cv2.fillPoly(img, [corners], 20.0)
    for level, (cx, cy, pw, ph) in zip((30, 60, 100, 120, 145), chart["patches"]):
        x0, y0 = to_px(cx - pw / 2, cy - ph / 2)
        x1, y1 = to_px(cx + pw / 2, cy + ph / 2)
        img[int(y0) : int(y1), int(x0) : int(x1)] = level
    img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    bgr = np.repeat(img[..., None], 3, axis=2)
    if noise > 0:
        bgr = bgr + np.random.default_rng(seed).normal(0, noise, bgr.shape)
    return np.clip(np.round(bgr), 0, 255).astype(np.uint8), sq / ss


# ---------- MREQ-CHART-01 ----------
def test_TC_CHART_01_1_SquareFoundWithScale():
    bgr, sq = chart_frame()
    centre, scale, _ = find_square(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), CHARTS["z10"]["side"])
    assert abs(centre[0] - sq[0]) < 1.0 and abs(centre[1] - sq[1]) < 1.0
    assert scale == pytest.approx(SCALE, rel=0.02)


def test_TC_CHART_01_2_NoSquareRejected():
    with pytest.raises(ValueError, match="square"):
        find_square(np.full((1080, 1920), 160, np.uint8), 16.0)


# ---------- MREQ-CHART-02 ----------
def test_TC_CHART_02_1_FlatNoiseScalesWithSigma():
    rng = np.random.default_rng(1)
    box = (10, 10, 190, 190)
    flat = np.full((200, 200), 120.0, np.float32)
    n3 = flat_noise(flat + rng.normal(0, 3, flat.shape), box)
    n6 = flat_noise(flat + rng.normal(0, 6, flat.shape), box)
    assert n6 / n3 == pytest.approx(2.0, rel=0.05)
    assert flat_noise(flat, box) == pytest.approx(0.0, abs=1e-6)


def test_TC_CHART_02_2_OvershootOfKnownStep():
    y = np.full((100, 200), 160.0, np.float32)
    y[50:] = 100.0
    y[47:49] = 166.0  # +6 DN above a 60 DN step = 10 %
    over, under = top_edge_overshoot(y, (40, 50, 160, 90))
    assert over == pytest.approx(10.0, abs=2.0)
    assert under == pytest.approx(0.0, abs=1.0)


def test_TC_CHART_02_3_MTF50OfGaussianEdge():
    bgr, _ = chart_frame(blur=1.0)
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, _, contour = find_square(y, CHARTS["z10"]["side"])
    expected = np.sqrt(np.log(2) / (2 * np.pi**2)) / 1.0  # Gaussian MTF = 0.5
    for axis in (0, 1):
        assert edge_mtf(y, contour, axis)["mtf50"] == pytest.approx(expected, rel=0.10)


# ---------- MREQ-CHART-03 ----------
def test_TC_CHART_03_1_IdentityModelIsNeutral():
    frames = []
    for seed in range(3):
        bgr, _ = chart_frame(noise=4.0, seed=seed)
        frames.append(score_frame(bgr, CHARTS["z10"], lambda x: x))
    d = summarize(frames)["derived"]
    assert d["added_over_p2"] == pytest.approx(0.0, abs=1e-9)
    assert d["added_over_p3"] == pytest.approx(0.0, abs=1e-9)
    assert d["mtf50_ratio_h"] == pytest.approx(1.0) and d["mtf50_ratio_v"] == pytest.approx(1.0)
    assert d["noise_ratio_p4"] == pytest.approx(1.0)


def _sharpened(bgr):
    """Unsharp mask like the camera's Aperture: about 10-15 % overshoot at the chart edges."""
    f = bgr.astype(np.float32)
    return np.clip(np.round(f + 1.5 * (f - cv2.GaussianBlur(f, (0, 0), 1.0))), 0, 255).astype(np.uint8)


def test_TC_CHART_03_3_NoisyPerfectDenoiserIsNeutral():
    # Noise 10 DN, 10 frames: the scorer's own precision for a perfect denoiser was measured at
    # sd 0.7 %p (P2), 1.2 %p (P3), 0.4-0.7 %p (square) and 0.02 (MTF ratio) over 6 seed sets.
    clean = _sharpened(chart_frame(blur=0.9)[0])
    frames = []
    for seed in range(10):
        noise = np.random.default_rng(seed).normal(0, 10.0, clean.shape[:2])[..., None]
        noisy = np.clip(np.round(clean + noise), 0, 255).astype(np.uint8)
        x0, y0 = chart_window(noisy, CHARTS["z10"])[:2]
        oracle = lambda win, x0=x0, y0=y0: clean[y0 : y0 + win.shape[0], x0 : x0 + win.shape[1]]
        frames.append(score_frame(noisy, CHARTS["z10"], oracle))
    s = summarize(frames)
    d = s["derived"]
    assert s["camera"]["over_p2"] > 5.0  # the input's sharpening halo is seen through the noise (true 7.9 %)
    assert d["added_over_p2"] == pytest.approx(0.0, abs=2.0)
    assert d["added_over_p3"] == pytest.approx(0.0, abs=3.5)
    assert d["added_square_over_h"] == pytest.approx(0.0, abs=2.0)
    assert d["added_square_over_v"] == pytest.approx(0.0, abs=2.0)
    assert d["mtf50_ratio_h"] == pytest.approx(1.0, abs=0.06) and d["mtf50_ratio_v"] == pytest.approx(1.0, abs=0.06)


def test_TC_CHART_03_2_GeometryRoundTrip():
    chart = CHARTS["z10"]
    c_px = (500.0, 400.0)
    x, y = mm_to_px((chart["square"][0] + 16.6, chart["square"][1]), c_px, chart["square"], SCALE)
    assert (x, y) == pytest.approx((600.0, 400.0))
    x0, y0, x1, y1 = box_px(patch_box(chart["patches"][4]), c_px, chart["square"], SCALE, shrink=0.4)
    assert x1 > x0 and y1 > y0


# ---------- MREQ-CHART-04 ----------
def _clip(tmp_path, n_frames, name="z10_test.nv12"):
    p = tmp_path / name
    p.write_bytes(np.zeros(n_frames * FRAME_BYTES, np.uint8).tobytes())
    return p


def test_TC_CHART_04_1_TooFewFramesRejected(tmp_path):
    clip = _clip(tmp_path, 3)
    with pytest.raises(ValueError, match="frames selected"):
        main(
            [
                "--checkpoint",
                str(tmp_path / "missing.pth"),
                "--clips",
                f"z10_test={clip}",
                "--frames",
                "0::1",
                "--min_frames",
                "10",
                "--manifest",
                "",
            ]
        )


def test_TC_CHART_04_2_ManifestMismatchRejected(tmp_path):
    clip = _clip(tmp_path, 2)
    manifest = tmp_path / "clips.sha256"
    manifest.write_text(f"{hashlib.sha256(b'other').hexdigest()}  {clip.name}\n")
    with pytest.raises(ValueError, match="manifest"):
        main(
            [
                "--checkpoint",
                str(tmp_path / "missing.pth"),
                "--clips",
                f"z10_test={clip}",
                "--frames",
                "0::1",
                "--n_frames",
                "2",
                "--min_frames",
                "2",
                "--manifest",
                str(manifest),
            ]
        )


def test_TC_CHART_04_3_BadClipSizeRejected(tmp_path):
    clip = tmp_path / "z10_bad.nv12"
    clip.write_bytes(b"\0" * 1000)
    with pytest.raises(ValueError, match="NV12"):
        main(["--checkpoint", str(tmp_path / "missing.pth"), "--clips", f"z10_bad={clip}", "--manifest", ""])


# ---------- MREQ-CHART-05: guards ----------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _blur(bgr):
    return cv2.GaussianBlur(bgr, (0, 0), 2.0)


def test_TC_CHART_05_1_DotRetentionIdentityAndBlur():
    with open(os.path.join(REPO, "configs", "data", "bicubicpp", "denoise.yaml"), encoding="utf-8") as f:
        cam = yaml.safe_load(f)["camera_model"]
    ident = dot_retention(lambda x: x, cam, sizes=(1, 2), amps=(60,), seeds=2)
    blur = dot_retention(_blur, cam, sizes=(1, 2), amps=(60,), seeds=2)
    for k in ("dot1px_60", "dot2px_60"):
        assert ident[k] == pytest.approx(ident["camera_" + k])
        assert blur[k] < 0.5 * ident[k]
    assert ident["dot2px_60"] > ident["dot1px_60"] > 0.3  # the lens blurs a 1-px dot, the camera sharpens it back up


def test_TC_CHART_05_2_TextureRatio(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i}.png"), rng.integers(0, 256, (600, 600, 3), dtype=np.uint8))
    assert texture_ratio(lambda x: x, str(tmp_path)) == pytest.approx(1.0)
    assert texture_ratio(_blur, str(tmp_path)) < 0.5
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        texture_ratio(lambda x: x, str(tmp_path / "empty"))


def test_TC_CHART_05_3_HotPixelResidual():
    bgr, _ = chart_frame()
    x0, y0 = chart_window(bgr, CHARTS["z10"])[:2]
    xy = np.array([[x0 + 40 + 30 * i, y0 + 650] for i in range(8)])
    bgr = bgr.astype(np.int16)
    bgr[xy[:, 1], xy[:, 0]] += 40
    bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    frames = [score_frame(bgr, CHARTS["z10"], lambda w: w, xy)]
    assert frames[0]["camera"]["hot_residual"] == pytest.approx(40, abs=2)
    assert summarize(frames)["derived"]["hot_residual_ratio"] == pytest.approx(1.0)
    frames = [score_frame(bgr, CHARTS["z10"], lambda w: cv2.medianBlur(w, 3), xy)]
    assert summarize(frames)["derived"]["hot_residual_ratio"] < 0.2


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE_DN = "results/denoise_release_v1"


def numeric_leaves(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from numeric_leaves(v, f"{path}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from numeric_leaves(v, f"{path}/{i}")
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield path, node


def test_TC_CHART_06_1_ReleaseDenoiseReproducesP11(tmp_path, monkeypatch):
    """The shipped BicubicPPDenoise[v1] scores exactly what P1.1 recorded for it (EXP-09-1)."""
    ref_path = f"{RELEASE_DN}/analysis/p1_scores/P1.1/EXP-09-1.json"
    clips = [c.split("=", 1)[1] for c in DEFAULT_CLIPS]
    need = [
        f"{RELEASE_DN}/checkpoint/best.pth",
        f"{RELEASE_DN}/checkpoint/train_config.yaml",
        f"{RELEASE_DN}/checkpoint/data_config.yaml",
        ref_path,
        DEFAULT_MANIFEST,
        DEFAULT_HOT_PIXELS,
        *clips,
    ]
    missing = [p for p in need if not os.path.exists(os.path.join(REPO, p))]
    if missing or not os.path.isdir(DEFAULT_TEXTURE_ROOT):
        pytest.skip(f"not on this machine: {missing or DEFAULT_TEXTURE_ROOT}")
    monkeypatch.chdir(REPO)  # the clip, manifest and hot-pixel defaults are repo-relative
    main(
        [
            "--checkpoint",
            f"{RELEASE_DN}/checkpoint/best.pth",
            "--camera_config",
            f"{RELEASE_DN}/checkpoint/data_config.yaml",
            "--json",
            str(tmp_path / "p11.json"),
        ]
    )
    ours = dict(numeric_leaves(json.load(open(tmp_path / "p11.json"))))
    ref = dict(numeric_leaves(json.load(open(ref_path))))
    assert len(ref) == 161 and set(ours) == set(ref)
    differ = [k for k in ref if ours[k] != ref[k] and not (math.isnan(ours[k]) and math.isnan(ref[k]))]
    assert not differ, differ[:5]
